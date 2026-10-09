# Screening enrichment benchmark (CA2, 1CA2 chain A)

49 actives (ChEMBL binding assays, pChEMBL >= 7) vs 49 size-matched
weak binders (pChEMBL <= 5); 2 ligand(s) failed to dock and are excluded.

| Metric | Value |
|---|---|
| ROC AUC, best Vina score | **0.58** (95% CI 0.47-0.69) |
| ROC AUC, heavy-atom count only (control) | 0.50 |
| Enrichment factor, top 10% | 1.1 (random = 1.0) |
| Spearman, score vs pChEMBL (actives only) | -0.06 |
| Primary-sulfonamide fraction, actives / weak | 86% / 6% |
| ROC AUC, 'has a primary sulfonamide' alone (control) | 0.90 |
| ROC AUC, Vina score within the 45 sulfonamides | 0.52 |

Pocket chosen per ligand: {'P1': 98}

| Truth | Top-pose label | Count |
|---|---|---|
| active | MODERATE | 28 |
| active | STRONG | 16 |
| active | WEAK | 5 |
| weak | MODERATE | 17 |
| weak | STRONG | 26 |
| weak | WEAK | 6 |
