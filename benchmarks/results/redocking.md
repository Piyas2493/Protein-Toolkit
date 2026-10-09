# Redocking benchmark

Success = heavy-atom RMSD <= 2.0 A to the crystal pose. Seeds: 42, 1, 2. Vina exhaustiveness and pose count as in the run's JSON.

Receptor preparation: 3HS4 pdb2pqr+meeko pH 7.4, 448 polar H; 3PTB pdb2pqr+meeko pH 7.4, 383 polar H; 1STP pdb2pqr+meeko pH 7.4, 213 polar H; 181L pdb2pqr+meeko pH 7.4, 325 polar H.

## Native-box mode (box centered on the crystal ligand)

| PDB | Ligand | Heavy atoms | Rot. bonds | Top-1 RMSD (A) per seed | Best-of-N RMSD (A) per seed | Top-1 Vina score |
|---|---|---|---|---|---|---|
| 3HS4 | AZM | 13 | 2 | 5.38, 5.36, 5.24 | 1.45, 1.79, 1.37 | -6.47 |
| 3PTB | BEN | 9 | 1 | 0.39, 0.39, 0.39 | 0.39, 0.39, 0.39 | -5.99 |
| 1STP | BTN | 16 | 5 | 0.62, 0.61, 0.68 | 0.62, 0.61, 0.68 | -7.47 |
| 181L | BNZ | 6 | 0 | 0.86, 0.87, 0.88 | 0.32, 0.31, 0.79 | -5.47 |

**Top-1 success: 9/12 runs** (75%); best-of-N success: 12/12.

## Pipeline mode (pocket from HETATM records)

The pocket is built from the structure's HETATM records, which include the crystal ligand itself: a small center-to-ligand distance here is by construction, not site finding.

| PDB | Pocket found | Pocket center to ligand (A) | Top-1 RMSD (A) | Best-of-N RMSD (A) | Pose label |
|---|---|---|---|---|---|
| 3HS4 | P2 | 0.1 | 5.37 | 2.26 | STRONG |
| 3PTB | P2 | 0.2 | 0.40 | 0.40 | WEAK |
| 1STP | P1 | 1.0 | 0.60 | 0.60 | STRONG |
| 181L | P1 | 13.2 | 12.28 | 11.73 | WEAK |

## Blind mode (all HETATM removed; cavity detection)

The detector sees protein atoms only, so this is the honest test of whether the site is found.

| PDB | Pocket found | Pocket center to ligand (A) | Top-1 RMSD (A) | Best-of-N RMSD (A) | Pose label |
|---|---|---|---|---|---|
| 3HS4 | P15 | 6.3 | 1.33 | 1.33 | MODERATE |
| 3PTB | P5 | 8.9 | 2.55 | 2.55 | WEAK |
| 1STP | P1 | 9.1 | 5.05 | 5.05 | WEAK |
| 181L | P1 | 11.3 | 17.45 | 13.51 | WEAK |
