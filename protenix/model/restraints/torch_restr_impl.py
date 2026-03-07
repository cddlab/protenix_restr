"""GPU implementation of restraint energy and gradient calculations.

Uses PyTorch tensors for all computations, enabling GPU-accelerated optimization
via the torchmin library. Supports bond, angle, chiral, distance, and VdW restraints.

VdW restraints require torch_cluster (optional). If not installed, VdW is disabled.
"""
from __future__ import annotations

import numpy as np
import torch

from .bond_restr_data import BondData
from .angle_restr_data import AngleData
from .chiral_data import ChiralData
from .distance_restr_data import DistanceData

try:
    import torch_cluster
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

try:
    from torchmin.function import sf_value, ScalarFunction, de_value
    HAS_TORCHMIN = True
except ImportError:
    HAS_TORCHMIN = False


def _calculate_distances(atom_pos: torch.Tensor, atom_idx: torch.Tensor):
    """Compute pairwise distances and unit vectors for given atom index pairs."""
    dir_vec = atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]]
    dist = torch.norm(dir_vec, dim=1)
    safe_dist = torch.clamp(dist, min=1e-8)
    unit_vec = dir_vec / safe_dist.unsqueeze(1)
    return dist, unit_vec, dir_vec


class RestrTorchImpl:
    """GPU-based energy and gradient engine for all restraint types.

    All atom indices are *local* (i.e., positions within active_sites, not global
    atom indices). The flat index used in index_add_ is: local_idx + batch * natoms.

    Args:
        bond_data: list of BondData with resolved aid0/aid1.
        angle_data: list of AngleData with resolved aid0/aid1/aid2.
        chiral_data: list of ChiralData with resolved aid0-aid3.
        distance_data: list of DistanceData with resolved target_local_sites1/2.
        nbatch: number of diffusion samples in a chunk.
        natoms: number of active atoms (len(active_sites)).
        device: target device.
    """

    def __init__(
        self,
        bond_data: list[BondData],
        angle_data: list[AngleData],
        chiral_data: list[ChiralData],
        distance_data: list[DistanceData],
        nbatch: int,
        natoms: int,
        device,
    ):
        self.device = device
        self.nbatch = nbatch
        self.natoms = natoms
        self.use_vdw = False
        self.vdw_idx = None
        self.vdw_liglig_idx = None

        self._setup_bonds(bond_data, nbatch, natoms)
        self._setup_angles(angle_data, nbatch, natoms)
        self._setup_chirals(chiral_data, nbatch, natoms)
        self._setup_distance(distance_data, nbatch, natoms)

    # ------------------------------------------------------------------
    # Setup methods
    # ------------------------------------------------------------------

    def _setup_bonds(self, bond_data: list[BondData], nbatch: int, natoms: int) -> None:
        if len(bond_data) == 0 or bond_data[0].w <= 0.0:
            self.use_bonds = False
            return
        self.use_bonds = True

        data, r0s = [], []
        for ib in range(nbatch):
            for bond in bond_data:
                if not bond.is_valid():
                    continue
                data.append([bond.aid0 + ib * natoms, bond.aid1 + ib * natoms])
                r0s.append(bond.r0)

        self.bond_idx = torch.tensor(data, dtype=torch.long, device=self.device)
        self.bond_r0s = torch.tensor(r0s, dtype=torch.float32, device=self.device)
        self.bond_k = bond_data[0].w

    def _setup_angles(self, angle_data: list[AngleData], nbatch: int, natoms: int) -> None:
        if len(angle_data) == 0 or angle_data[0].w <= 0.0:
            self.use_angles = False
            return
        self.use_angles = True

        data, r0s = [], []
        for ib in range(nbatch):
            for angle in angle_data:
                if not angle.is_valid():
                    continue
                data.append([
                    angle.aid0 + ib * natoms,
                    angle.aid1 + ib * natoms,
                    angle.aid2 + ib * natoms,
                ])
                r0s.append(angle.th0)

        self.angle_idx = torch.tensor(data, dtype=torch.long, device=self.device)
        self.angle_r0s = torch.tensor(r0s, dtype=torch.float32, device=self.device)
        self.angle_k = angle_data[0].w

    def _setup_chirals(self, chiral_data: list[ChiralData], nbatch: int, natoms: int) -> None:
        if len(chiral_data) == 0 or chiral_data[0].w <= 0.0:
            self.use_chirals = False
            return
        self.use_chirals = True

        data, r0s = [], []
        for ib in range(nbatch):
            for ch in chiral_data:
                if not ch.is_valid():
                    continue
                data.append([
                    ch.aid0 + ib * natoms,
                    ch.aid1 + ib * natoms,
                    ch.aid2 + ib * natoms,
                    ch.aid3 + ib * natoms,
                ])
                r0s.append(ch.chiral_vol)

        self.chiral_idx = torch.tensor(data, dtype=torch.long, device=self.device)
        self.chiral_r0s = torch.tensor(r0s, dtype=torch.float32, device=self.device)
        self.chiral_k = chiral_data[0].w

    def _setup_distance(self, distance_data: list[DistanceData], nbatch: int, natoms: int) -> None:
        if len(distance_data) == 0:
            self.use_distance = False
            return
        self.use_distance = True

        self.distance_restraints = []
        for dist_restr in distance_data:
            if not dist_restr.is_valid():
                continue
            self.distance_restraints.append({
                "sites1": torch.tensor(
                    dist_restr.target_local_sites1, dtype=torch.long, device=self.device
                ),
                "sites2": torch.tensor(
                    dist_restr.target_local_sites2, dtype=torch.long, device=self.device
                ),
                "type": dist_restr.distance_restraint_type,
                "target_dist": dist_restr.target_distance,
                "target_dist1": dist_restr.target_distance1,
                "target_dist2": dist_restr.target_distance2,
                "num_sites1": len(dist_restr.target_local_sites1),
                "num_sites2": len(dist_restr.target_local_sites2),
            })

        if len(self.distance_restraints) == 0:
            self.use_distance = False

    def setup_vdw(
        self,
        nbatch: int,
        natoms: int,
        ligand_atoms: list[int],
        elems: torch.Tensor,
        config: dict,
    ) -> None:
        """Set up VdW (van der Waals) clash restraints between ligand and protein atoms.

        Args:
            nbatch: number of diffusion samples.
            natoms: number of active atoms.
            ligand_atoms: local indices (within active_sites) of ligand atoms.
            elems: element one-hot tensor of shape (N_active, 128), already sliced to
                   active_sites. Used to look up VdW radii via RDKit periodic table.
            config: VdW config dict with keys: weight, scale, dmax, ligand_only.
        """
        self.vdw_k = config.get("weight", None)
        if self.vdw_k is None or self.vdw_k <= 0 or not HAS_TORCH_CLUSTER:
            if not HAS_TORCH_CLUSTER and config.get("weight", 0) > 0:
                print("[RestrTorchImpl] torch_cluster not found; VdW disabled. "
                      "Install with: pip install torch_cluster")
            self.use_vdw = False
            return

        if len(ligand_atoms) == 0:
            self.use_vdw = False
            return

        self.use_vdw = True
        device = self.device

        all_atoms = list(range(natoms))
        prot_atoms = [i for i in all_atoms if i not in set(ligand_atoms)]

        self.ligand_idx = torch.tensor(ligand_atoms, dtype=torch.long, device=device)
        self.prot_idx = torch.tensor(prot_atoms, dtype=torch.long, device=device)

        self.lig_batch = torch.arange(nbatch, device=device).repeat_interleave(len(self.ligand_idx))
        self.prot_batch = torch.arange(nbatch, device=device).repeat_interleave(len(self.prot_idx))

        # Flat indices into (nbatch * natoms) tensor for each atom role
        self.lind_flat = self.ligand_idx.repeat(nbatch) + self.lig_batch * natoms
        self.pind_flat = self.prot_idx.repeat(nbatch) + self.prot_batch * natoms

        # VdW radii for all active atoms (repeated for each batch)
        vdwr = self._compute_vdw_radii(elems)  # (N_active,)
        self.vdwr = vdwr.repeat(nbatch)         # (nbatch * N_active,)

        self.vdw_scale = config.get("scale", 0.75)
        self.vdw_dmax = config.get("dmax", 5.0)
        self.vdw_lig_only = config.get("ligand_only", False)

        print(
            f"[RestrTorchImpl] VdW enabled: scale={self.vdw_scale}, "
            f"dmax={self.vdw_dmax}, ligand_only={self.vdw_lig_only}, "
            f"n_ligand={len(ligand_atoms)}, n_prot={len(prot_atoms)}"
        )

    def _compute_vdw_radii(self, elems_oh: torch.Tensor) -> torch.Tensor:
        """Convert element one-hot (N_active, 128) to VdW radii tensor via RDKit."""
        from rdkit import Chem
        peri = Chem.rdchem.GetPeriodicTable()
        elems = torch.argmax(elems_oh, dim=-1).cpu().numpy()  # (N_active,)

        def elem2rvdw(x: int) -> float:
            try:
                if x < 1 or x > 118:
                    return 0.0
                return peri.GetRvdw(int(x))
            except Exception:
                return 0.0

        vdwr = np.vectorize(elem2rvdw)(elems)
        return torch.tensor(vdwr, dtype=torch.float32, device=self.device)

    def update_vdw_idx(self, crds: torch.Tensor) -> None:
        """Recompute VdW contact pairs based on current coordinates.

        Called at the start of each minimize_gpu() step because atom positions
        change every diffusion step. Uses torch_cluster.radius() for fast
        GPU-based neighbor search.

        Args:
            crds: shape (nbatch, N_active, 3) on GPU.
        """
        if not self.use_vdw:
            return

        prot_crds = crds[:, self.prot_idx, :].reshape(-1, 3)
        lig_crds = crds[:, self.ligand_idx, :].reshape(-1, 3)

        # Protein-ligand contacts
        idx_j, idx_i = torch_cluster.radius(
            x=prot_crds,
            y=lig_crds,
            batch_x=self.prot_batch,
            batch_y=self.lig_batch,
            r=self.vdw_dmax,
        )
        if len(idx_i) > 0:
            idx_i_global = self.pind_flat[idx_i]
            idx_j_global = self.lind_flat[idx_j]
            self.vdw_idx = torch.stack([idx_i_global, idx_j_global], dim=1)
            lig_r = self.vdwr[self.vdw_idx[:, 1]]
            prot_r = self.vdwr[self.vdw_idx[:, 0]]
            self.vdw_r0s = (prot_r + lig_r) * self.vdw_scale
        else:
            self.vdw_idx = None

        # Ligand-ligand contacts
        idx_j_lig, idx_i_lig = torch_cluster.radius(
            x=lig_crds,
            y=lig_crds,
            batch_x=self.lig_batch,
            batch_y=self.lig_batch,
            r=self.vdw_dmax,
        )
        mask = idx_i_lig < idx_j_lig
        idx_i_lig, idx_j_lig = idx_i_lig[mask], idx_j_lig[mask]

        if len(idx_i_lig) > 0:
            idx_i_global = self.lind_flat[idx_i_lig]
            idx_j_global = self.lind_flat[idx_j_lig]
            self.vdw_liglig_idx = torch.stack([idx_i_global, idx_j_global], dim=1)

            r1 = self.vdwr[self.vdw_liglig_idx[:, 0]]
            r2 = self.vdwr[self.vdw_liglig_idx[:, 1]]
            self.vdw_liglig_r0s = (r1 + r2) * self.vdw_scale

            # Exclude bonded pairs from ligand-ligand VdW
            if self.use_bonds and self.bond_idx is not None and len(self.bond_idx) > 0:
                sorted_vdw = torch.sort(self.vdw_liglig_idx, dim=1)[0]
                sorted_bond = torch.sort(self.bond_idx, dim=1)[0]
                is_bond = (sorted_vdw.unsqueeze(1) == sorted_bond.unsqueeze(0)).all(dim=2).any(dim=1)
                self.vdw_liglig_idx = self.vdw_liglig_idx[~is_bond]
                self.vdw_liglig_r0s = self.vdw_liglig_r0s[~is_bond]

            if len(self.vdw_liglig_idx) == 0:
                self.vdw_liglig_idx = None
        else:
            self.vdw_liglig_idx = None

    # ------------------------------------------------------------------
    # Energy + gradient
    # ------------------------------------------------------------------

    def _calc_bond_grad(self, atom_pos: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        dist, unit_vec, _ = _calculate_distances(atom_pos, self.bond_idx)
        x = dist - self.bond_r0s
        pot = self.bond_k * x * x
        force = 2.0 * self.bond_k * x
        forcevec = unit_vec * force[:, None]
        grad.index_add_(0, self.bond_idx[:, 0], forcevec)
        grad.index_add_(0, self.bond_idx[:, 1], -forcevec)
        return pot.sum()

    def _calc_angle_grad(self, atom_pos: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        _, _, r21 = _calculate_distances(atom_pos, self.angle_idx[:, [0, 1]])
        _, _, r23 = _calculate_distances(atom_pos, self.angle_idx[:, [2, 1]])
        dotprod = (r23 * r21).sum(dim=1)
        norm21inv = 1.0 / torch.norm(r21, dim=1)
        norm23inv = 1.0 / torch.norm(r23, dim=1)
        cos_theta = torch.clamp(dotprod * norm21inv * norm23inv, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        delta = theta - self.angle_r0s
        pot = self.angle_k * delta * delta

        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        coef = torch.zeros_like(sin_theta)
        nz = sin_theta != 0
        coef[nz] = 2.0 * self.angle_k * delta[nz] / sin_theta[nz]

        f0 = coef[:, None] * (cos_theta[:, None] * r21 * norm21inv[:, None] - r23 * norm23inv[:, None]) * norm21inv[:, None]
        f2 = coef[:, None] * (cos_theta[:, None] * r23 * norm23inv[:, None] - r21 * norm21inv[:, None]) * norm23inv[:, None]
        f1 = -(f0 + f2)

        grad.index_add_(0, self.angle_idx[:, 0], f0)
        grad.index_add_(0, self.angle_idx[:, 1], f1)
        grad.index_add_(0, self.angle_idx[:, 2], f2)
        return pot.sum()

    def _calc_chiral_grad(self, atom_pos: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        a0 = atom_pos[self.chiral_idx[:, 0]]
        a1 = atom_pos[self.chiral_idx[:, 1]]
        a2 = atom_pos[self.chiral_idx[:, 2]]
        a3 = atom_pos[self.chiral_idx[:, 3]]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0
        vol = (v1 * torch.cross(v2, v3, dim=1)).sum(dim=1)
        delta = vol - self.chiral_r0s
        pot = self.chiral_k * delta * delta
        dE = (2.0 * self.chiral_k * delta)[:, None]
        f1 = torch.cross(v2, v3, dim=1) * dE
        f2 = torch.cross(v3, v1, dim=1) * dE
        f3 = torch.cross(v1, v2, dim=1) * dE
        fc = -(f1 + f2 + f3)
        grad.index_add_(0, self.chiral_idx[:, 0], fc)
        grad.index_add_(0, self.chiral_idx[:, 1], f1)
        grad.index_add_(0, self.chiral_idx[:, 2], f2)
        grad.index_add_(0, self.chiral_idx[:, 3], f3)
        return pot.sum()

    def _calc_distance_grad(self, atom_pos: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        crds = atom_pos.view(self.nbatch, self.natoms, 3)
        total_pot = torch.tensor(0.0, device=self.device)
        batch_offsets = torch.arange(self.nbatch, device=self.device) * self.natoms

        for restr in self.distance_restraints:
            com1 = crds[:, restr["sites1"], :].mean(dim=1)  # (nbatch, 3)
            com2 = crds[:, restr["sites2"], :].mean(dim=1)
            vec = com2 - com1
            dist = torch.norm(vec, dim=1)

            delta = torch.zeros_like(dist)
            rtype = restr["type"]
            if rtype == "harmonic":
                delta = dist - restr["target_dist"]
            elif rtype in ("flat-bottomed", "flat-bottomed1"):
                mask = dist < restr["target_dist1"]
                delta[mask] = dist[mask] - restr["target_dist1"]
            if rtype in ("flat-bottomed", "flat-bottomed2"):
                mask = dist > restr["target_dist2"]
                delta = torch.where(mask, dist - restr["target_dist2"], delta)

            total_pot = total_pot + delta.pow(2).sum()
            safe_dist = torch.clamp(dist, min=1e-8)
            grad_com = (2.0 * delta / safe_dist).unsqueeze(1) * vec  # (nbatch, 3)

            # Distribute gradient equally to all atoms in each group
            n1, n2 = restr["num_sites1"], restr["num_sites2"]
            s1_global = (restr["sites1"].unsqueeze(0) + batch_offsets.unsqueeze(1)).view(-1)
            s2_global = (restr["sites2"].unsqueeze(0) + batch_offsets.unsqueeze(1)).view(-1)
            src1 = (-grad_com / n1).unsqueeze(1).expand(-1, n1, -1).reshape(-1, 3)
            src2 = (grad_com / n2).unsqueeze(1).expand(-1, n2, -1).reshape(-1, 3)
            grad.index_add_(0, s1_global, src1)
            grad.index_add_(0, s2_global, src2)

        return total_pot

    def _calc_vdw_grad(self, atom_pos: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        total_pot = torch.tensor(0.0, device=self.device)

        if self.vdw_idx is not None:
            dist, unit_vec, _ = _calculate_distances(atom_pos, self.vdw_idx)
            x = dist - self.vdw_r0s
            flag = x < 0
            pot = self.vdw_k * x * x * flag
            force = (2.0 * self.vdw_k * x * flag)[:, None] * unit_vec
            if not self.vdw_lig_only:
                grad.index_add_(0, self.vdw_idx[:, 0], force)
            grad.index_add_(0, self.vdw_idx[:, 1], -force)
            total_pot = total_pot + pot.sum()

        if self.vdw_liglig_idx is not None:
            dist, unit_vec, _ = _calculate_distances(atom_pos, self.vdw_liglig_idx)
            x = dist - self.vdw_liglig_r0s
            flag = x < 0
            pot = self.vdw_k * x * x * flag
            force = (2.0 * self.vdw_k * x * flag)[:, None] * unit_vec
            grad.index_add_(0, self.vdw_liglig_idx[:, 0], force)
            grad.index_add_(0, self.vdw_liglig_idx[:, 1], -force)
            total_pot = total_pot + pot.sum()

        return total_pot

    def grad(self, crds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total gradient and energy for all restraints.

        Args:
            crds: shape (nbatch, N_active, 3) or (nbatch * N_active * 3,) flat.

        Returns:
            (gradient tensor same shape as input, scalar energy tensor)
        """
        atom_pos = crds.reshape(-1, 3)
        g = torch.zeros_like(atom_pos)
        f = torch.tensor(0.0, device=self.device)

        if self.use_bonds:
            f = f + self._calc_bond_grad(atom_pos, g)
        if self.use_angles:
            f = f + self._calc_angle_grad(atom_pos, g)
        if self.use_chirals:
            f = f + self._calc_chiral_grad(atom_pos, g)
        if self.use_distance:
            f = f + self._calc_distance_grad(atom_pos, g)
        if self.use_vdw and (self.vdw_idx is not None or self.vdw_liglig_idx is not None):
            f = f + self._calc_vdw_grad(atom_pos, g)

        return g.reshape(-1), f


if HAS_TORCHMIN:
    class MyScalarFunc(ScalarFunction):
        """Adapter between RestrTorchImpl and torchmin optimizer.

        torchmin requires a ScalarFunction with closure() and dir_evaluate().
        closure() returns energy + gradient for the current position.
        dir_evaluate() evaluates energy + gradient at x + t*d (line search).
        """

        def __init__(self, impl: RestrTorchImpl, x_shape: torch.Size):
            super().__init__(lambda x: x, x_shape)
            self.impl = impl

        def closure(self, x):
            g, f = self.impl.grad(x)
            return sf_value(f=f.detach(), grad=g.detach(), hessp=None, hess=None)

        def dir_evaluate(self, x, t, d):
            x_new = (x + d.mul(t)).detach()
            g, f = self.impl.grad(x_new)
            return de_value(f=float(f), grad=g)

else:
    class MyScalarFunc:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torchmin is required for GPU restraints. "
                "Install with: pip install torchmin"
            )
