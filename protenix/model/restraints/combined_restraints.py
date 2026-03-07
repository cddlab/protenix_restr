"""CombinedRestraints: manages conformer and distance restraints during diffusion sampling.

During the reverse diffusion loop, after each denoising step, the predicted coordinates
x_denoised are refined via CG optimization to satisfy geometric restraints:
  - Conformer restraints: bond lengths, bond angles, chiral volumes (for ligands)
  - Distance restraints: COM distance between user-specified atom groups

Usage:
  # At data processing time (json_to_feature.py):
  combined_restr = CombinedRestraints.get_instance()
  combined_restr.set_config(restraints_config)
  combined_restr.make_bond(ai, aj, conf, global_indices)
  combined_restr.make_chiral(iatm, mol, conf, global_indices)
  combined_restr.make_angle_restraints(mol, conf, global_indices)
  ref_tensor = combined_restr.build_conformer_restraint_tensor(N_atom)
  feature_dict["ref_conformer_restraint"] = ref_tensor
  combined_restr.set_feats(feature_dict, atom_array)

  # At inference time (_main_inference_loop):
  combined_restr = CombinedRestraints.get_instance()
  combined_restr.setup_site(input_feature_dict)
  # then pass to sample_diffusion as combined_restraints=combined_restr
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from rdkit import Chem
from scipy import optimize

from .chiral_data import ChiralData, calc_chiral_vol
from .angle_restr_data import AngleData, get_angle_idxs
from .bond_restr_data import BondData
from .distance_restr_data import DistanceData


class CombinedRestraints:
    """Manages all restraints applied during diffusion sampling.

    Singleton pattern: call CombinedRestraints.get_instance() to obtain the shared instance.
    Reset with CombinedRestraints._instance = None before each new prediction job.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> CombinedRestraints:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.chiral_data: list[ChiralData] = []
        self.bond_data: list[BondData] = []
        self.angle_data: list[AngleData] = []
        self.distance_data: list[DistanceData] = []
        # sites[sid-1] is a list of setup callbacks for site sid (1-indexed; 0 = unconstrained)
        self.sites: list[list] = []
        # Maps global atom index -> site_id (1-indexed)
        self.site_registry: dict[int, int] = {}
        # Set after setup_site(); global atom indices included in CG optimization
        self.active_sites: list[int] = []
        # Config
        self.config: dict = {}
        self.verbose: bool = False
        self.method: str = "CG"
        self.max_iter: int = 100
        self.start_sigma: float = 1.0
        self.conformer_config: dict = {}
        self.bond_config: dict = {}
        self.angle_config: dict = {}
        self.chiral_config: dict = {}
        # CG state (set in minimize)
        self.nbatch: int = 0
        self.natoms: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_config(self, config: dict) -> None:
        self.config = config
        self.verbose = config.get("verbose", False)
        self.method = config.get("method", "CG")
        self.max_iter = int(config.get("max_iter", 100))
        self.start_sigma = float(config.get("start_sigma", 1.0))

        conformer_cfg = config.get("conformer_restraints_config", {})
        self.conformer_config = conformer_cfg
        self.bond_config = conformer_cfg.get("bond", {})
        self.angle_config = conformer_cfg.get("angle", {})
        self.chiral_config = conformer_cfg.get("chiral", {})

        distance_cfg = config.get("distance_restraints_config", [])
        for entry in distance_cfg:
            dist_restr = DistanceData()
            dist_restr.set_config(entry)
            self.distance_data.append(dist_restr)

    # ------------------------------------------------------------------
    # Factory helpers for data objects
    # ------------------------------------------------------------------

    def _create_bond_data(self, d: float) -> BondData:
        return BondData(
            -1, -1, d,
            slack=self.bond_config.get("slack", 0.0),
            w=self.bond_config.get("weight", 0.05),
        )

    def _create_angle_data(self, th0: float) -> AngleData:
        return AngleData(
            -1, -1, -1, th0,
            slack=self.angle_config.get("slack", math.radians(5.0)),
            w=self.angle_config.get("weight", 0.05),
        )

    def _create_chiral_data(self, chiral_vol: float) -> ChiralData:
        return ChiralData(
            -1, -1, -1, -1, chiral_vol,
            slack=self.chiral_config.get("slack", 0.05),
            w=self.chiral_config.get("weight", 0.05),
            fmax=self.chiral_config.get("f_max", -100.0),
        )

    # ------------------------------------------------------------------
    # Site registry (maps global atom idx -> site_id)
    # ------------------------------------------------------------------

    def register_site(self, global_atom_idx: int, callback) -> None:
        """Register a global atom index with a setup callback.

        The callback is called during setup_site() with the local index
        (position within active_sites) to resolve aid0/1/2/3 in data objects.
        """
        sid = self.site_registry.get(global_atom_idx, 0)
        if sid == 0:
            self.sites.append([callback])
            new_sid = len(self.sites)
            self.site_registry[global_atom_idx] = new_sid
        else:
            self.sites[sid - 1].append(callback)

    def get_sites(self, index: int) -> list:
        if index == 0:
            return []
        return self.sites[index - 1]

    # ------------------------------------------------------------------
    # Conformer restraint builders
    # ------------------------------------------------------------------

    def make_bond(
        self,
        rdkit_ai: int,
        rdkit_aj: int,
        conf,
        global_indices,
    ) -> None:
        """Register a bond length restraint.

        Args:
            rdkit_ai, rdkit_aj: atom indices in the H-removed RDKit mol.
            conf: RDKit conformer (of the H-removed mol).
            global_indices: array mapping mol atom index -> global AtomArray index.
        """
        crds = conf.GetPositions()
        d = np.linalg.norm(crds[rdkit_aj] - crds[rdkit_ai])
        bnd = self._create_bond_data(d)
        self.bond_data.append(bnd)

        global_ai = int(global_indices[rdkit_ai])
        global_aj = int(global_indices[rdkit_aj])
        self.register_site(global_ai, lambda x, b=bnd: b.setup(x, 0))
        self.register_site(global_aj, lambda x, b=bnd: b.setup(x, 1))

    def make_angle(
        self,
        rdkit_ai: int,
        rdkit_aj: int,
        rdkit_ak: int,
        conf,
        global_indices,
    ) -> None:
        """Register a bond angle restraint."""
        th0 = AngleData.calc_angle(rdkit_ai, rdkit_aj, rdkit_ak, conf)
        angl = self._create_angle_data(th0)
        self.angle_data.append(angl)

        global_ai = int(global_indices[rdkit_ai])
        global_aj = int(global_indices[rdkit_aj])
        global_ak = int(global_indices[rdkit_ak])
        self.register_site(global_ai, lambda x, a=angl: a.setup(x, 0))
        self.register_site(global_aj, lambda x, a=angl: a.setup(x, 1))
        self.register_site(global_ak, lambda x, a=angl: a.setup(x, 2))

    def make_angle_restraints(self, mol: Chem.Mol, conf, global_indices) -> None:
        """Register angle restraints for all 1-2-3 triples in mol."""
        idxs = get_angle_idxs(mol)
        for ai, aj, ak in idxs:
            self.make_angle(int(ai), int(aj), int(ak), conf, global_indices)

    def make_chiral_impl(
        self,
        rdkit_ai: int,
        rdkit_aj: list[int],
        conf,
        global_indices,
        invert: bool = False,
    ) -> None:
        """Register a chiral volume restraint for one set of 3 neighbors."""
        crds = conf.GetPositions()
        chiral_vol = calc_chiral_vol(crds, rdkit_ai, rdkit_aj)
        if invert:
            chiral_vol = -chiral_vol
        ch = self._create_chiral_data(chiral_vol)
        self.chiral_data.append(ch)

        global_ai = int(global_indices[rdkit_ai])
        global_ajs = [int(global_indices[j]) for j in rdkit_aj]
        self.register_site(global_ai, lambda x, c=ch: c.setup(x, 0))
        self.register_site(global_ajs[0], lambda x, c=ch: c.setup(x, 1))
        self.register_site(global_ajs[1], lambda x, c=ch: c.setup(x, 2))
        self.register_site(global_ajs[2], lambda x, c=ch: c.setup(x, 3))
        if self.verbose:
            print(f"chiral restr {rdkit_ai} - {rdkit_aj}: vol={chiral_vol:.2f}")

    def make_chiral(
        self,
        iatm: int,
        mol: Chem.Mol,
        conf,
        global_indices,
        invert: bool = False,
    ) -> None:
        """Register chiral volume restraints for all combinations of 3 neighbors."""
        nei_ind = ChiralData.get_nei_atoms(iatm, mol)
        for cand in itertools.combinations(nei_ind, 3):
            self.make_chiral_impl(iatm, list(cand), conf, global_indices, invert=invert)

    # ------------------------------------------------------------------
    # Feature tensor
    # ------------------------------------------------------------------

    def build_conformer_restraint_tensor(self, N_atom: int) -> torch.Tensor:
        """Build the ref_conformer_restraint tensor from site_registry.

        Returns:
            LongTensor of shape (N_atom,): 0 = no restraint, >0 = site_id.
        """
        t = torch.zeros(N_atom, dtype=torch.long)
        for global_idx, site_id in self.site_registry.items():
            if global_idx < N_atom:
                t[global_idx] = site_id
        return t

    # ------------------------------------------------------------------
    # Distance restraint setup
    # ------------------------------------------------------------------

    def set_feats(self, feats: dict, atom_array) -> None:
        """Resolve distance restraint atom selections to global indices."""
        atom_to_token_idx = feats.get("atom_to_token_idx")
        if atom_to_token_idx is None:
            return
        for dist_restr in self.distance_data:
            dist_restr.set_feats(atom_array, atom_to_token_idx)

    # ------------------------------------------------------------------
    # Setup (called before diffusion sampling)
    # ------------------------------------------------------------------

    def setup_site(self, feats: dict) -> None:
        """Resolve site indices and build active_sites list.

        Must be called after set_feats() (for distance restraints) and
        after ref_conformer_restraint is in feats (for conformer restraints).
        """
        self.reset_indices()

        # Reset local site lists for distance restraints (safe for multiple calls)
        for dist_restr in self.distance_data:
            dist_restr.target_local_sites1 = []
            dist_restr.target_local_sites2 = []

        self.active_sites = []

        # Add atoms from conformer restraints
        if "ref_conformer_restraint" in feats:
            feat_restr_tensor = feats["ref_conformer_restraint"]
            if feat_restr_tensor.dim() > 1:
                feat_restr_tensor = feat_restr_tensor[0]
            feat_restr = feat_restr_tensor.detach().cpu().numpy()
            for ind in range(len(feat_restr)):
                if int(feat_restr[ind]) != 0:
                    self.active_sites.append(ind)
        else:
            feat_restr = None

        # Add atoms from distance restraints
        for dist_restr in self.distance_data:
            self.active_sites += dist_restr.target_sites1
            self.active_sites += dist_restr.target_sites2

        if len(self.active_sites) == 0:
            return

        self.active_sites = sorted(set(self.active_sites))
        if self.verbose:
            print(f"active_sites: {len(self.active_sites)} atoms")

        # Build local index mappings
        for local_i, global_idx in enumerate(self.active_sites):
            # Map distance restraint global -> local
            for dist_restr in self.distance_data:
                if global_idx in set(dist_restr.target_sites1):
                    dist_restr.target_local_sites1.append(local_i)
                if global_idx in set(dist_restr.target_sites2):
                    dist_restr.target_local_sites2.append(local_i)

            # Resolve conformer restraint site callbacks
            if feat_restr is not None and global_idx < len(feat_restr):
                sid = int(feat_restr[global_idx])
                if sid != 0:
                    for callback in self.get_sites(sid):
                        callback(local_i)

        if self.verbose:
            for ch in self.chiral_data:
                if ch.is_valid():
                    print(f"chiral: {ch.aid0}-{ch.aid1}-{ch.aid2}-{ch.aid3}")

        print(
            f"[CombinedRestraints] active_sites={len(self.active_sites)}, "
            f"bonds={len(self.bond_data)}, angles={len(self.angle_data)}, "
            f"chirals={len(self.chiral_data)}, distances={len(self.distance_data)}"
        )

    def is_active(self) -> bool:
        return len(self.active_sites) > 0

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def minimize(
        self, batch_crds_in: torch.Tensor, istep: int, sigma_t: float
    ) -> None:
        """Run CG minimization on active atoms for one diffusion step.

        Args:
            batch_crds_in: coordinates tensor shape (N_batch, N_atom, 3), modified in-place.
            istep: current diffusion step index (for logging).
            sigma_t: current noise level; minimization is skipped if sigma_t > start_sigma.
        """
        if not self.is_active():
            return
        if sigma_t > self.start_sigma:
            return

        device = batch_crds_in.device
        crds = batch_crds_in.detach().cpu().numpy()
        crds = crds[:, self.active_sites, :]  # (N_batch, N_active, 3)
        self.nbatch = crds.shape[0]
        self.natoms = crds.shape[1]
        crds_flat = crds.reshape(-1)

        if self.verbose:
            print(f"[Restraints] minimizing step {istep}, sigma_t={sigma_t:.4f}")

        opt = optimize.minimize(
            self.calc,
            crds_flat,
            jac=self.grad,
            method=self.method,
            options={"maxiter": self.max_iter},
        )

        crds_out = opt.x.reshape(self.nbatch, self.natoms, 3)
        batch_crds_in[:, self.active_sites, :] = torch.tensor(
            crds_out, dtype=batch_crds_in.dtype
        ).to(device)

        if self.verbose:
            self.print_stat(crds_out)

    def finalize(self, batch_crds_in: torch.Tensor, istep: int) -> None:
        """Print final restraint statistics after diffusion."""
        if not self.is_active():
            return
        crds = batch_crds_in.detach().cpu().numpy()[:, self.active_sites, :]
        print(f"=== Restraint final stats (step {istep}) ===")
        self.print_stat(crds)

    # ------------------------------------------------------------------
    # Energy and gradient for CG optimizer
    # ------------------------------------------------------------------

    def calc(self, crds_in: np.ndarray) -> float:
        ene = 0.0
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)
        for i in range(self.nbatch):
            for ch in self.chiral_data:
                if ch.is_valid():
                    ene += ch.calc(crds[i])
            for b in self.bond_data:
                if b.is_valid():
                    ene += b.calc(crds[i])
            for a in self.angle_data:
                if a.is_valid():
                    ene += a.calc(crds[i])
            for d in self.distance_data:
                if d.is_valid():
                    ene += d.calc(crds[i])
        return ene

    def grad(self, crds_in: np.ndarray) -> np.ndarray:
        crds = crds_in.reshape(self.nbatch, self.natoms, 3)
        grad = np.zeros_like(crds)
        for i in range(self.nbatch):
            for ch in self.chiral_data:
                if ch.is_valid():
                    ch.grad(crds[i], grad[i])
            for b in self.bond_data:
                if b.is_valid():
                    b.grad(crds[i], grad[i])
            for a in self.angle_data:
                if a.is_valid():
                    a.grad(crds[i], grad[i])
            for d in self.distance_data:
                if d.is_valid():
                    d.grad(crds[i], grad[i])
        return grad.reshape(-1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_indices(self) -> None:
        """Reset all local indices (called before setup_site)."""
        for ch in self.chiral_data:
            ch.reset_indices()
        for b in self.bond_data:
            b.reset_indices()
        for a in self.angle_data:
            a.reset_indices()

    def print_stat(self, crds: np.ndarray) -> None:
        """Print per-batch restraint statistics."""
        for i in range(crds.shape[0]):
            if self.chiral_data:
                ch_ene = sum(c.calc(crds[i]) for c in self.chiral_data if c.is_valid())
                ch_sd = sum(c.calc_sd(crds[i]) for c in self.chiral_data if c.is_valid())
                print(f"  chiral E={ch_ene:.5f} rmsd={np.sqrt(ch_sd / max(1, len(self.chiral_data))):.5f}")
            if self.bond_data:
                b_ene = sum(b.calc(crds[i]) for b in self.bond_data if b.is_valid())
                b_sd = sum(b.calc_sd(crds[i]) for b in self.bond_data if b.is_valid())
                print(f"  bond  E={b_ene:.5f} rmsd={np.sqrt(b_sd / max(1, len(self.bond_data))):.5f}")
            if self.angle_data:
                a_ene = sum(a.calc(crds[i]) for a in self.angle_data if a.is_valid())
                a_sd = sum(a.calc_sd(crds[i]) for a in self.angle_data if a.is_valid())
                print(f"  angle E={a_ene:.5f} rmsd={np.sqrt(a_sd / max(1, len(self.angle_data))):.5f}")
            if self.distance_data:
                d_ene = sum(d.calc(crds[i]) for d in self.distance_data if d.is_valid())
                print(f"  dist  E={d_ene:.5f}")
