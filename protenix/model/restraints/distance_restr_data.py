from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .selection import AtomSelector


@dataclass
class DistanceData:
    """COM distance restraint between two atom groups.

    Penalty types:
      harmonic       - (d - target)^2
      flat-bottomed  - penalizes d < d1 AND d > d2
      flat-bottomed1 - penalizes d < d1 only
      flat-bottomed2 - penalizes d > d2 only
    """

    atom_selection1: str = None
    atom_selection2: str = None
    target_distance: float = None   # for harmonic
    target_distance1: float = None  # for flat-bottomed, flat-bottomed1
    target_distance2: float = None  # for flat-bottomed, flat-bottomed2
    distance_restraint_type: str = None
    target_sites1: list = field(default_factory=list)  # global atom indices
    target_sites2: list = field(default_factory=list)
    target_local_sites1: list = field(default_factory=list)  # local indices within active_sites
    target_local_sites2: list = field(default_factory=list)
    calc_method: str = "unfixed-absolute"
    run_restr: bool = False

    def set_config(self, config: dict) -> None:
        self.atom_selection1 = config.get("atom_selection1")
        self.atom_selection2 = config.get("atom_selection2")
        self.calc_method = config.get("calc_method", "unfixed-absolute")

        if "harmonic" in config:
            self.target_distance = float(config["harmonic"]["target_distance"])
            self.distance_restraint_type = "harmonic"
        elif "flat-bottomed" in config:
            self.target_distance1 = float(config["flat-bottomed"]["target_distance1"])
            self.target_distance2 = float(config["flat-bottomed"]["target_distance2"])
            assert self.target_distance1 <= self.target_distance2, (
                "target_distance1 must be <= target_distance2"
            )
            self.distance_restraint_type = "flat-bottomed"
        elif "flat-bottomed1" in config:
            self.target_distance1 = float(config["flat-bottomed1"]["target_distance1"])
            self.distance_restraint_type = "flat-bottomed1"
        elif "flat-bottomed2" in config:
            self.target_distance2 = float(config["flat-bottomed2"]["target_distance2"])
            self.distance_restraint_type = "flat-bottomed2"
        else:
            raise ValueError("No valid distance restraint type found in config.")

        self.run_restr = (
            self.atom_selection1 is not None
            and self.atom_selection2 is not None
            and self.distance_restraint_type is not None
        )

    def set_feats(self, atom_array, atom_to_token_idx: torch.Tensor) -> None:
        """Resolve atom selections to global indices using the full AtomArray.

        Args:
            atom_array: Biotite AtomArray (unpadded, with label_asym_id annotation).
            atom_to_token_idx: shape (N_atom,) or (1, N_atom), maps atom → token index.
        """
        if not self.run_restr:
            return

        # Handle optional batch dimension
        if atom_to_token_idx.dim() > 1:
            atom_to_token_idx = atom_to_token_idx[0]

        selector1 = AtomSelector(self.atom_selection1)
        selector2 = AtomSelector(self.atom_selection2)

        self.target_sites1 = []
        self.target_sites2 = []

        n_atoms = len(atom_array)
        for i in range(n_atoms):
            chain = atom_array.label_asym_id[i]
            token_idx = int(atom_to_token_idx[i].item())
            candidate = {
                "chain": chain,
                "resid": token_idx + 1,  # 1-indexed, matches selection syntax
                "index": i,
            }
            if selector1.matches(candidate):
                self.target_sites1.append(i)
            if selector2.matches(candidate):
                self.target_sites2.append(i)

        assert len(self.target_sites1) > 0, (
            f"atom_selection1 '{self.atom_selection1}' matched no atoms"
        )
        assert len(self.target_sites2) > 0, (
            f"atom_selection2 '{self.atom_selection2}' matched no atoms"
        )
        print(f"[DistanceRestr] group1={len(self.target_sites1)} atoms, "
              f"group2={len(self.target_sites2)} atoms")

    def _calculate_com_vector(self, crds: np.ndarray) -> np.ndarray:
        com1 = np.mean(crds[self.target_local_sites1, :], axis=0)
        com2 = np.mean(crds[self.target_local_sites2, :], axis=0)
        return com2 - com1

    def calc(self, crds: np.ndarray) -> float:
        com_vector = self._calculate_com_vector(crds)
        dist = np.linalg.norm(com_vector)
        delta = 0.0
        rtype = self.distance_restraint_type
        if rtype == "harmonic":
            delta = dist - self.target_distance
        elif rtype in ("flat-bottomed", "flat-bottomed1") and dist < self.target_distance1:
            delta = dist - self.target_distance1
        elif rtype in ("flat-bottomed", "flat-bottomed2") and dist > self.target_distance2:
            delta = dist - self.target_distance2
        return delta ** 2

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        com_vector = self._calculate_com_vector(crds)
        dist = np.linalg.norm(com_vector)
        if dist < 1e-8:
            return
        delta = 0.0
        rtype = self.distance_restraint_type
        if rtype == "harmonic":
            delta = dist - self.target_distance
        elif rtype in ("flat-bottomed", "flat-bottomed1") and dist < self.target_distance1:
            delta = dist - self.target_distance1
        elif rtype in ("flat-bottomed", "flat-bottomed2") and dist > self.target_distance2:
            delta = dist - self.target_distance2
        if abs(delta) < 1e-9:
            return
        coeff = 2 * delta
        grad_com = coeff * com_vector / dist
        grad_atom1 = -grad_com / len(self.target_local_sites1)
        grad_atom2 = grad_com / len(self.target_local_sites2)
        grad[self.target_local_sites1, :] += grad_atom1
        grad[self.target_local_sites2, :] += grad_atom2

    def is_valid(self) -> bool:
        return self.run_restr

    def distance(self, crds: np.ndarray) -> float:
        return np.linalg.norm(self._calculate_com_vector(crds))

    def print(self, crds: np.ndarray) -> None:
        print(
            f"COM distance '{self.atom_selection1}' - '{self.atom_selection2}':"
            f" {self.distance(crds):.3f} Ang"
        )

    def calc_sd(self, crds: np.ndarray) -> float:
        dist = self.distance(crds)
        delta = 0.0
        rtype = self.distance_restraint_type
        if rtype == "harmonic":
            delta = dist - self.target_distance
        elif rtype in ("flat-bottomed", "flat-bottomed1") and dist < self.target_distance1:
            delta = dist - self.target_distance1
        elif rtype in ("flat-bottomed", "flat-bottomed2") and dist > self.target_distance2:
            delta = dist - self.target_distance2
        return delta ** 2
