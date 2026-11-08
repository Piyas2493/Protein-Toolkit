# Active-site validation (structure-based catalytic-site detector)

Textbook catalytic residues of classic proteases, numbered as in the deposited file. `legacy` = any residues of the right types within 8 A (centroids); `strict` = needs the conserved motif, uses the residues it marks, side-chain atoms, connected cluster within 5 A.

## Development set (optimistic: used to design `strict`)

| Detector | Right system reported | Catalytic residues recovered | Structures with false extra systems |
|---|---|---|---|
| legacy | 7/7 | 9/17 | 7/7 |
| strict | 7/7 | 17/17 | 0/7 |

| PDB | Enzyme | Truth | strict result | legacy result |
|---|---|---|---|---|
| 3PTB | bovine trypsin | ASP102, HIS57, SER195 | 3/3 (ASP102, HIS57, SER195) | 2/3 (ASP102, HIS57, SER214; false: Aspartic Protease, Cysteine Protease Dyad) |
| 5CHA | bovine chymotrypsin A | ASP102, HIS57, SER195 | 3/3 (ASP102, HIS57, SER195) | 2/3 (ASP102, HIS57, SER214; false: Aspartic Protease, Cysteine Protease Dyad) |
| 1SBT | subtilisin BPN' | ASP32, HIS64, SER221 | 3/3 (ASP32, HIS64, SER221) | 0/3 (ASP36, HIS39, SER38; false: Aspartic Protease) |
| 9PAP | papain; Cys25 is stored as OCS (modified) | CYS25, HIS159 | 2/2 (CYS25, HIS159) | 2/2 (CYS25, HIS159; false: Aspartic Protease) |
| 5PEP | porcine pepsin | ASP215, ASP32 | 2/2 (ASP215, ASP32) | 0/2 (ASP280, ASP281; false: Cysteine Protease Dyad, Serine Protease Triad) |
| 8TLN | thermolysin | GLU143, HIS142, HIS146 | 3/3 (GLU143, HIS142, HIS146) | 3/3 (GLU143, HIS142, HIS146; false: Aspartic Protease, Serine Protease Triad) |
| 1HVR | HIV-1 protease (dimer) | ASP25 | 1/1 (ASP25) | 0/1 (ASP29, ASP30; false: Cysteine Protease Dyad) |

## Held-out set

| Detector | Right system reported | Catalytic residues recovered | Structures with false extra systems |
|---|---|---|---|
| legacy | 8/8 | 10/18 | 8/8 |
| strict | 7/8 | 16/18 | 0/8 |

| PDB | Enzyme | Truth | strict result | legacy result |
|---|---|---|---|---|
| 3EST | porcine pancreatic elastase | ASP102, HIS57, SER195 | 3/3 (ASP102, HIS57, SER195) | 2/3 (ASP102, HIS57, SER214; false: Aspartic Protease, Cysteine Protease Dyad) |
| 1PPB | human alpha-thrombin | ASP102, HIS57, SER195 | 3/3 (ASP102, HIS57, SER195) | 2/3 (ASP102, HIS57, SER214; false: Aspartic Protease, Cysteine Protease Dyad) |
| 4APE | endothiapepsin; literature Asp35/Asp219, but THIS file uses pepsin-style numbering (35 and 219 are Ser/Thr) | ASP215, ASP32 | 2/2 (ASP215, ASP32) | 0/2 (ASP12, ASP8; false: Serine Protease Triad) |
| 1HSG | HIV-1 protease + indinavir (dimer) | ASP25 | 1/1 (ASP25) | 0/1 (ASP29, ASP30; false: Cysteine Protease Dyad) |
| 1CP3 | caspase-3 + inhibitor; Cys163-His121 are 5.95 A apart here | CYS163, HIS121 | 0/2 (not reported) | 2/2 (CYS163, HIS121; false: Aspartic Protease, Metalloprotease, Serine Protease Triad) |
| 2ACT | actinidin; Cys25 stored as CSD (partly development, see docstring) | CYS25, HIS162 | 2/2 (CYS25, HIS162) | 2/2 (CYS25, HIS162; false: Aspartic Protease) |
| 1PPN | papain, unmodified Cys25 | CYS25, HIS159 | 2/2 (CYS25, HIS159) | 2/2 (CYS25, HIS159; false: Aspartic Protease) |
| 1AST | astacin | GLU93, HIS92, HIS96 | 3/3 (GLU93, HIS92, HIS96) | 0/3 (GLU103, HIS104, HIS148; false: Aspartic Protease) |

Limits: 15 structures, all proteases of the four covered systems; truth written from the literature; the detector marks a PLAUSIBLE catalytic cluster and says nothing about enzymes outside those four systems. 4APE's truth was corrected after inspecting the file (its score with the literature numbering would be 0/2).
