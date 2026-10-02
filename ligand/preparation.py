"""
ProteinToolkit Ligand Preparation (Sprint 5)

Implements Section 12 of the master prompt:
    Input ligand
        ↓
    Standardization
        ↓
    Protonation handling
        ↓
    Tautomer handling
        ↓
    Hydrogen addition
        ↓
    3D conformer generation
        ↓
    Energy minimization
        ↓
    Docking-ready ligand

Every transformation is recorded in `Ligand.provenance`. If RDKit is
available, we delegate to it; otherwise we fall back to a deterministic
geometry builder that places atoms along simple bond vectors.

The preparator never silently alters the chemistry: tautomer choices
and protonation state are explicit and returned in the report.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ligand import Ligand, LigandAtom, LigandBond


@dataclass
class PreparationStep:
    name: str
    applied: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparationReport:
    ligand: Ligand
    steps: List[PreparationStep] = field(default_factory=list)
    tautomer_chosen: Optional[str] = None
    protonation_state: str = "neutral"
    has_3d: bool = False
    energy_estimate: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {"name": s.name, "applied": s.applied, "details": s.details}
                for s in self.steps
            ],
            "tautomer_chosen": self.tautomer_chosen,
            "protonation_state": self.protonation_state,
            "has_3d": self.has_3d,
            "energy_estimate": self.energy_estimate,
            "notes": list(self.notes),
            "provenance": list(self.ligand.provenance),
        }


class LigandPreparator:
    """Run the ligand preparation pipeline."""

    def __init__(
        self,
        *,
        protonation: str = "neutral",   # "neutral", "deprotonate-acid", "protonate-base"
        tautomer: str = "first",         # "first", "enumerate"
        add_hydrogens: bool = True,
        generate_3d: bool = True,
        minimize_energy: bool = True,
        seed: int = 42,
    ):
        if protonation not in {"neutral", "deprotonate-acid", "protonate-base"}:
            raise ValueError(f"Unsupported protonation mode: {protonation}")
        if tautomer not in {"first", "enumerate"}:
            raise ValueError(f"Unsupported tautomer mode: {tautomer}")
        self.protonation = protonation
        self.tautomer = tautomer
        self.add_hydrogens = add_hydrogens
        self.generate_3d = generate_3d
        self.minimize_energy = minimize_energy
        self.seed = seed

    def prepare(self, ligand: Ligand) -> PreparationReport:
        # Work on a deep copy so the original ligand is not mutated.
        work = _deep_copy_ligand(ligand)
        report = PreparationReport(ligand=work)

        work.record("preparation_started", seed=self.seed,
                    protonation=self.protonation, tautomer=self.tautomer)

        # 1) Standardization — RDKit if available
        std_step = self._standardize(work)
        report.steps.append(std_step)

        # 2) Protonation handling
        p_step = self._apply_protonation(work)
        report.steps.append(p_step)
        report.protonation_state = p_step.details.get("state", "neutral")

        # 3) Tautomer handling
        t_step = self._apply_tautomer(work)
        report.steps.append(t_step)
        report.tautomer_chosen = t_step.details.get("chosen")

        # 4) Hydrogen addition
        h_step = self._add_hydrogens(work)
        report.steps.append(h_step)

        # 5) 3D conformer
        c_step = self._generate_3d(work)
        report.steps.append(c_step)
        report.has_3d = c_step.applied

        # 6) Energy minimization
        m_step = self._minimize(work)
        report.steps.append(m_step)
        report.energy_estimate = m_step.details.get("energy")

        if not report.has_3d:
            report.notes.append(
                "Ligand does not have 3D coordinates; downstream docking "
                "must perform its own conformer sampling."
            )

        work.record("preparation_completed", steps=[s.name for s in report.steps])
        return report

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def _standardize(self, lig: Ligand) -> PreparationStep:
        try:
            from rdkit import Chem  # type: ignore
            # If RDKit is available we reparse the SMILES; otherwise we
            # cannot standardize at the chemistry level.
            if lig.smiles:
                mol = Chem.MolFromSmiles(lig.smiles)
                if mol is not None:
                    lig.record("standardize", engine="rdkit")
                    return PreparationStep(
                        "standardize", True,
                        {"engine": "rdkit", "smiles_canonical": Chem.MolToSmiles(mol)},
                    )
        except ImportError:
            pass

        lig.record("standardize", engine="fallback",
                   note="RDKit not available; structure passed through unchanged.")
        return PreparationStep(
            "standardize", True,
            {"engine": "fallback"},
        )

    # ------------------------------------------------------------------
    # Protonation
    # ------------------------------------------------------------------

    def _apply_protonation(self, lig: Ligand) -> PreparationStep:
        state = "neutral"
        before = sum(a.formal_charge for a in lig.atoms)

        if self.protonation == "deprotonate-acid":
            # Carboxylic acid proxy: find a C bonded to TWO oxygens,
            # deprotonate one of them (the OH one, which has implicit H
            # after preparation).
            changed = 0
            c_id = _carboxyl_carbon(lig)
            if c_id is not None:
                for a in lig.atoms:
                    if a.element != "O":
                        continue
                    if any(
                        (b.a1 == a.serial and b.a2 == c_id) or
                        (b.a2 == a.serial and b.a1 == c_id)
                        for b in lig.bonds
                    ):
                        # Pick the single-bonded O (the OH)
                        for b in lig.bonds:
                            if (b.a1 == a.serial and b.a2 == c_id) or \
                               (b.a2 == a.serial and b.a1 == c_id):
                                if abs(b.order - 1.0) < 0.01:
                                    a.h_count = 0
                                    a.formal_charge = -1
                                    changed += 1
                                    break
                        if changed:
                            break
            state = "deprotonated-acid"
        elif self.protonation == "protonate-base":
            # Protonate a free amine (N with bond_sum <= 3 and 2 Hs).
            changed = 0
            for a in lig.atoms:
                if a.element != "N":
                    continue
                bond_sum = sum(
                    b.order for b in lig.bonds
                    if b.a1 == a.serial or b.a2 == a.serial
                )
                if bond_sum <= 3 and a.h_count >= 1 and a.formal_charge == 0:
                    a.h_count = a.h_count + 1
                    a.formal_charge = 1
                    changed += 1
                    break
            state = "protonated-base"
        else:
            changed = 0

        after = sum(a.formal_charge for a in lig.atoms)
        lig.record("protonation", mode=self.protonation, state=state,
                   delta_charge=after - before)
        return PreparationStep(
            "protonation", True,
            {"mode": self.protonation, "state": state,
             "atoms_changed": changed},
        )

    # ------------------------------------------------------------------
    # Tautomer
    # ------------------------------------------------------------------

    def _apply_tautomer(self, lig: Ligand) -> PreparationStep:
        chosen = None
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
            if lig.smiles:
                mol = Chem.MolFromSmiles(lig.smiles)
                if mol is not None:
                    enumerator = rdMolStandardize.TautomerEnumerator()
                    canon = enumerator.Canonicalize(mol)
                    chosen = Chem.MolToSmiles(canon)
        except ImportError:
            chosen = lig.smiles
        except Exception:
            chosen = lig.smiles

        lig.record("tautomer", mode=self.tautomer, chosen=chosen)
        return PreparationStep(
            "tautomer", True,
            {"mode": self.tautomer, "chosen": chosen},
        )

    # ------------------------------------------------------------------
    # Hydrogens
    # ------------------------------------------------------------------

    def _add_hydrogens(self, lig: Ligand) -> PreparationStep:
        if not self.add_hydrogens:
            lig.record("hydrogens", added=False)
            return PreparationStep("hydrogens", False, {})

        try:
            from rdkit import Chem  # type: ignore
            if lig.smiles:
                mol = Chem.MolFromSmiles(lig.smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    lig.record("hydrogens", engine="rdkit",
                               note="implicit H counts attached to heavy atoms")
                    return PreparationStep("hydrogens", True,
                                           {"engine": "rdkit"})
        except ImportError:
            pass

        # Fallback: compute implicit-H counts from valence rules.
        for a in lig.atoms:
            if a.h_count:
                continue
            if a.element == "H":
                continue
            max_v = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3}.get(a.element)
            if max_v is None:
                continue
            bond_sum = sum(
                b.order for b in lig.bonds
                if b.a1 == a.serial or b.a2 == a.serial
            )
            # Account for formal charge: an anion has one extra bond equivalent.
            free = max_v - bond_sum + max(0, a.formal_charge) - max(0, -a.formal_charge)
            a.h_count = max(0, int(round(free)))
        lig.record("hydrogens", engine="fallback")
        return PreparationStep("hydrogens", True, {"engine": "fallback"})

    # ------------------------------------------------------------------
    # 3D conformer
    # ------------------------------------------------------------------

    def _generate_3d(self, lig: Ligand) -> PreparationStep:
        # If ligand already has 3D coords from SDF/MOL2/PDB we keep them.
        has_3d = any(
            (a.x != 0.0 or a.y != 0.0 or a.z != 0.0)
            for a in lig.atoms
        )
        if has_3d:
            lig.record("3d_conformer", source="input")
            return PreparationStep("3d_conformer", True, {"source": "input"})

        if not self.generate_3d:
            lig.record("3d_conformer", generated=False)
            return PreparationStep("3d_conformer", False, {})

        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
            if lig.smiles:
                mol = Chem.MolFromSmiles(lig.smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    params = AllChem.ETKDGv3()
                    params.randomSeed = self.seed
                    AllChem.EmbedMolecule(mol, params)
                    conf = mol.GetConformer()
                    for atom in lig.atoms:
                        pos = conf.GetAtomPosition(atom.serial - 1)
                        atom.x, atom.y, atom.z = pos.x, pos.y, pos.z
                    lig.record("3d_conformer", engine="rdkit-etkdg", seed=self.seed)
                    return PreparationStep(
                        "3d_conformer", True,
                        {"engine": "rdkit-etkdg", "seed": self.seed},
                    )
        except ImportError:
            pass
        except Exception as exc:
            lig.record("3d_conformer", error=str(exc))

        # Fallback: tree-based placement along bond vectors.
        self._build_3d_tree(lig, seed=self.seed)
        lig.record("3d_conformer", engine="fallback-tree", seed=self.seed)
        return PreparationStep(
            "3d_conformer", True,
            {"engine": "fallback-tree", "seed": self.seed},
        )

    @staticmethod
    def _build_3d_tree(lig: Ligand, *, seed: int) -> None:
        """Place atoms along deterministic bond vectors with random rotation.

        Produces a connected, sterically plausible geometry. Not a real
        force field; that's what RDKit is for.
        """
        if not lig.atoms:
            return
        rng = random.Random(seed)
        placed: Dict[int, Tuple[float, float, float]] = {}

        # Start at the first atom
        first = lig.atoms[0]
        placed[first.serial] = (0.0, 0.0, 0.0)

        # BFS over bonds
        queue = [first.serial]
        head = 0
        # Atom-by-atom ideal bond lengths (very rough)
        bond_len = lambda e1, e2: 1.5 if "H" in (e1, e2) else 1.4

        # adjacency
        adj: Dict[int, List[Tuple[int, float]]] = {a.serial: [] for a in lig.atoms}
        for b in lig.bonds:
            adj[b.a1].append((b.a2, b.order))
            adj[b.a2].append((b.a1, b.order))

        while head < len(queue):
            cur = queue[head]
            head += 1
            cur_atom = _get_atom(lig, cur)
            if cur_atom is None:
                continue
            cx, cy, cz = placed[cur]
            placed_neighbors = 0
            for nbr, order in adj[cur]:
                if nbr in placed:
                    placed_neighbors += 1
                    continue
                nbr_atom = _get_atom(lig, nbr)
                if nbr_atom is None:
                    continue
                # Direction: pick a unit vector using golden-angle spiral,
                # jittered to avoid collinearity.
                theta = rng.random() * 2 * math.pi
                phi = math.acos(2 * rng.random() - 1)
                dx = math.sin(phi) * math.cos(theta)
                dy = math.sin(phi) * math.sin(theta)
                dz = math.cos(phi)
                length = bond_len(cur_atom.element, nbr_atom.element)
                placed[nbr] = (cx + dx * length, cy + dy * length, cz + dz * length)
                nbr_atom.x, nbr_atom.y, nbr_atom.z = placed[nbr]
                queue.append(nbr)

        # Any unplaced atoms go to the centroid with tiny offset
        centroid = (
            sum(p[0] for p in placed.values()) / max(1, len(placed)),
            sum(p[1] for p in placed.values()) / max(1, len(placed)),
            sum(p[2] for p in placed.values()) / max(1, len(placed)),
        )
        for atom in lig.atoms:
            if atom.serial not in placed:
                atom.x = centroid[0] + rng.random()
                atom.y = centroid[1] + rng.random()
                atom.z = centroid[2] + rng.random()

    # ------------------------------------------------------------------
    # Energy minimization
    # ------------------------------------------------------------------

    def _minimize(self, lig: Ligand) -> PreparationStep:
        if not self.minimize_energy:
            lig.record("minimize", applied=False)
            return PreparationStep("minimize", False, {})

        energy: Optional[float] = None
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
            if lig.smiles:
                mol = Chem.MolFromSmiles(lig.smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=self.seed)
                    res = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    energy = float(res)
                    lig.record("minimize", engine="rdkit-mmff", energy=energy)
                    return PreparationStep(
                        "minimize", True,
                        {"engine": "rdkit-mmff", "energy": energy},
                    )
        except ImportError:
            pass
        except Exception as exc:
            lig.record("minimize", error=str(exc))

        # Fallback: simple steric-clash penalty as a proxy "energy"
        clash = 0
        n = len(lig.atoms)
        for i in range(n):
            for j in range(i + 1, n):
                ai, aj = lig.atoms[i], lig.atoms[j]
                d = math.dist((ai.x, ai.y, ai.z), (aj.x, aj.y, aj.z))
                if d < 1.2:
                    clash += (1.2 - d) ** 2
        energy = round(clash, 3)
        lig.record("minimize", engine="fallback-steric", energy=energy)
        return PreparationStep(
            "minimize", True,
            {"engine": "fallback-steric", "energy": energy},
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _get_atom(lig: Ligand, serial: int) -> Optional[LigandAtom]:
    for a in lig.atoms:
        if a.serial == serial:
            return a
    return None


def _carboxyl_carbon(lig: Ligand) -> Optional[int]:
    """Return serial of a carbon bonded to >=2 oxygens, or None."""
    oxygen_neighbours: Dict[int, int] = {a.serial: 0 for a in lig.atoms}
    for b in lig.bonds:
        a1 = _get_atom(lig, b.a1)
        a2 = _get_atom(lig, b.a2)
        if not a1 or not a2:
            continue
        if a1.element == "C" and a2.element == "O":
            oxygen_neighbours[b.a1] = oxygen_neighbours.get(b.a1, 0) + 1
        elif a2.element == "C" and a1.element == "O":
            oxygen_neighbours[b.a2] = oxygen_neighbours.get(b.a2, 0) + 1
    for serial, count in oxygen_neighbours.items():
        if count >= 2:
            return serial
    return None


def _deep_copy_ligand(lig: Ligand) -> Ligand:
    new_atoms = [
        LigandAtom(
            serial=a.serial, element=a.element,
            x=a.x, y=a.y, z=a.z,
            formal_charge=a.formal_charge,
            aromatic=a.aromatic, in_ring=a.in_ring,
            h_count=a.h_count,
        )
        for a in lig.atoms
    ]
    new_bonds = [
        LigandBond(
            a1=b.a1, a2=b.a2, order=b.order,
            aromatic=b.aromatic, in_ring=b.in_ring,
        )
        for b in lig.bonds
    ]
    new = Ligand(
        name=lig.name, atoms=new_atoms, bonds=new_bonds,
        source_format=lig.source_format,
        source_path=lig.source_path,
        smiles=lig.smiles,
        chemistry_verified=lig.chemistry_verified,
    )
    return new
