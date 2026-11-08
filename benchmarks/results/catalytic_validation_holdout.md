# Catalytic rule validation against UniProt EC subclasses (held-out draw)

2000 reviewed sequences: Serine Protease Triad (250), Cysteine Protease Dyad (250), Aspartic Protease (250), Metalloprotease (250), neg:other hydrolase (EC 3.1/3.2/3.5/3.6) (250), neg:oxidoreductase (EC 1) (250), neg:transferase (EC 2) (250), neg:lyase/isomerase/ligase (EC 4-6) (250).

`composition` = the old rule (enough His/Ser/Asp... residues anywhere); `motif` = the current engine (needs a conserved active-site motif). Recall is on the rule's own protease type; false-hit rates are on sequences that are NOT that type.

| Rule | Engine | Recall | Fires on non-peptidases | Fires on other protease types | Precision |
|---|---|---|---|---|---|
| Serine Protease Triad | composition | 100% | 99% | 99% | 13% |
| Cysteine Protease Dyad | composition | 100% | 98% | 83% | 14% |
| Aspartic Protease | composition | 100% | 100% | 100% | 13% |
| Metalloprotease | composition | 100% | 99% | 96% | 13% |
| Serine Protease Triad | motif | 35% | 0% | 0% | 95% |
| Cysteine Protease Dyad | motif | 39% | 0% | 0% | 97% |
| Aspartic Protease | motif | 45% | 0% | 1% | 90% |
| Metalloprotease | motif | 89% | 1% | 1% | 92% |

Downstream: share of sequences told they are a **zinc metalloprotease** with a **Zn2+** ligand (function / ligand), by true class. Only the real metalloproteases should score high.

| True class | n | old (composition rule + function keyed to any Hydrolase) | new (motif rule + function keyed to Metalloprotease) |
|---|---|---|---|
| Serine Protease Triad | 250 | 100% / 100% | 0% / 0% |
| Cysteine Protease Dyad | 250 | 100% / 100% | 0% / 0% |
| Aspartic Protease | 250 | 100% / 100% | 9% / 9% |
| Metalloprotease | 250 | 100% / 100% | 89% / 89% |
| neg:other hydrolase (EC 3.1/3.2/3.5/3.6) | 250 | 100% / 100% | 2% / 2% |
| neg:oxidoreductase (EC 1) | 250 | 100% / 100% | 3% / 3% |
| neg:transferase (EC 2) | 250 | 100% / 100% | 2% / 2% |
| neg:lyase/isomerase/ligase (EC 4-6) | 250 | 100% / 100% | 2% / 2% |

| Engine | Non-peptidases with any catalytic hit | Mean rules fired per non-peptidase |
|---|---|---|
| composition | 100% | 3.96 |
| motif | 2% | 0.02 |
