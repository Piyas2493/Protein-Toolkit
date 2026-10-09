# Screening enrichment benchmark (ESR1, 3ERT chain A)

50 actives (ChEMBL binding assays, pChEMBL >= 7) vs 50 size-matched
weak binders (pChEMBL <= 5); 0 ligand(s) failed to dock and are excluded.
Receptor preparation: pdb2pqr/PROPKA + meeko at pH 7.4 (413 polar H, template Gasteiger charges).

| Metric | Value |
|---|---|
| ROC AUC, best Vina score | **0.60** (95% CI 0.49-0.70) |
| ROC AUC, heavy-atom count only (control) | 0.50 |
| Enrichment factor, top 10% | 1.6 (random = 1.0) |
| Spearman, score vs pChEMBL (actives only) | 0.04 |

Pocket chosen per ligand: {'P1': 100}

| Truth | Top-pose label | Count |
|---|---|---|
| active | MODERATE | 16 |
| active | STRONG | 34 |
| weak | MODERATE | 18 |
| weak | STRONG | 29 |
| weak | WEAK | 3 |
