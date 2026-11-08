# Catalytic rule validation against UniProt EC subclasses

3200 reviewed sequences: Serine Protease Triad (400), Cysteine Protease Dyad (400), Aspartic Protease (400), Metalloprotease (400), neg:other hydrolase (EC 3.1/3.2/3.5/3.6) (400), neg:oxidoreductase (EC 1) (400), neg:transferase (EC 2) (400), neg:lyase/isomerase/ligase (EC 4-6) (400).

`composition` = the old rule (enough His/Ser/Asp... residues anywhere); `motif` = the current engine (needs a conserved active-site motif). Recall is on the rule's own protease type; false-hit rates are on sequences that are NOT that type.

| Rule | Engine | Recall | Fires on non-peptidases | Fires on other protease types | Precision |
|---|---|---|---|---|---|
| Serine Protease Triad | composition | 100% | 100% | 99% | 13% |
| Cysteine Protease Dyad | composition | 100% | 99% | 89% | 13% |
| Aspartic Protease | composition | 100% | 100% | 100% | 13% |
| Metalloprotease | composition | 100% | 99% | 96% | 13% |
| Serine Protease Triad | motif | 56% | 1% | 0% | 95% |
| Cysteine Protease Dyad | motif | 41% | 0% | 0% | 97% |
| Aspartic Protease | motif | 47% | 1% | 1% | 92% |
| Metalloprotease | motif | 84% | 1% | 1% | 93% |

Downstream: share of sequences told they are a **zinc metalloprotease** with a **Zn2+** ligand (function / ligand), by true class. Only the real metalloproteases should score high.

| True class | n | old (composition rule + function keyed to any Hydrolase) | new (motif rule + function keyed to Metalloprotease) |
|---|---|---|---|
| Serine Protease Triad | 400 | 100% / 100% | 0% / 0% |
| Cysteine Protease Dyad | 400 | 100% / 100% | 0% / 0% |
| Aspartic Protease | 400 | 100% / 100% | 8% / 8% |
| Metalloprotease | 400 | 100% / 100% | 85% / 85% |
| neg:other hydrolase (EC 3.1/3.2/3.5/3.6) | 400 | 100% / 100% | 2% / 2% |
| neg:oxidoreductase (EC 1) | 400 | 100% / 100% | 3% / 3% |
| neg:transferase (EC 2) | 400 | 100% / 100% | 2% / 2% |
| neg:lyase/isomerase/ligase (EC 4-6) | 400 | 100% / 100% | 2% / 2% |

| Engine | Non-peptidases with any catalytic hit | Mean rules fired per non-peptidase |
|---|---|---|
| composition | 100% | 3.98 |
| motif | 2% | 0.02 |
