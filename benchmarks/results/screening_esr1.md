# Screening enrichment benchmark (ESR1, 3ERT chain A)

50 actives (ChEMBL binding assays, pChEMBL >= 7) vs 50 size-matched
weak binders (pChEMBL <= 5); 0 ligand(s) failed to dock and are excluded.

| Metric | Value |
|---|---|
| ROC AUC, best Vina score | **0.58** (95% CI 0.46-0.68) |
| ROC AUC, heavy-atom count only (control) | 0.50 |
| Enrichment factor, top 10% | 1.4 (random = 1.0) |
| Spearman, score vs pChEMBL (actives only) | 0.02 |

Pocket chosen per ligand: {'P1': 100}

| Truth | Top-pose label | Count |
|---|---|---|
| active | MODERATE | 18 |
| active | STRONG | 32 |
| weak | MODERATE | 22 |
| weak | STRONG | 23 |
| weak | WEAK | 5 |
