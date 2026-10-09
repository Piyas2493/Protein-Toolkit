# Redocking benchmark

Success = heavy-atom RMSD <= 2.0 A to the crystal pose. Seeds: 42, 1, 2. Vina exhaustiveness and pose count as in the run's JSON.

## Native-box mode (box centered on the crystal ligand)

| PDB | Ligand | Heavy atoms | Rot. bonds | Top-1 RMSD (A) per seed | Best-of-N RMSD (A) per seed | Top-1 Vina score |
|---|---|---|---|---|---|---|
| 3HS4 | AZM | 13 | 2 | 3.31, 3.27, 3.41 | 2.45, 1.40, 1.41 | -5.92 |
| 3PTB | BEN | 9 | 1 | 0.40, 0.43, 0.41 | 0.38, 0.38, 0.41 | -6.07 |
| 1STP | BTN | 16 | 5 | 0.87, 0.81, 0.86 | 0.87, 0.81, 0.86 | -6.31 |
| 181L | BNZ | 6 | 0 | 0.95, 0.88, 0.87 | 0.82, 0.88, 0.75 | -5.47 |

**Top-1 success: 9/12 runs** (75%); best-of-N success: 11/12.

## Pipeline mode (pocket from HETATM records)

The pocket is built from the structure's HETATM records, which include the crystal ligand itself: a small center-to-ligand distance here is by construction, not site finding.

| PDB | Pocket found | Pocket center to ligand (A) | Top-1 RMSD (A) | Best-of-N RMSD (A) | Pose label |
|---|---|---|---|---|---|
| 3HS4 | P2 | 0.1 | 5.26 | 2.44 | MODERATE |
| 3PTB | P2 | 0.2 | 0.35 | 0.35 | MODERATE |
| 1STP | P1 | 1.0 | 0.83 | 0.83 | STRONG |
| 181L | P1 | 13.2 | 12.30 | 12.27 | WEAK |

## Blind mode (all HETATM removed; cavity detection)

The detector sees protein atoms only, so this is the honest test of whether the site is found.

| PDB | Pocket found | Pocket center to ligand (A) | Top-1 RMSD (A) | Best-of-N RMSD (A) | Pose label |
|---|---|---|---|---|---|
| 3HS4 | P15 | 6.3 | 3.38 | 3.38 | WEAK |
| 3PTB | P5 | 8.9 | 2.81 | 2.81 | WEAK |
| 1STP | P1 | 9.1 | 5.99 | 5.88 | MODERATE |
| 181L | P1 | 11.3 | 17.26 | 13.89 | WEAK |
