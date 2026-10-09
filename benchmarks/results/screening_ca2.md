# Screening enrichment benchmark (CA2, 1CA2 chain A)

49 actives (ChEMBL binding assays, pChEMBL >= 7) vs 49 size-matched
weak binders (pChEMBL <= 5); 2 ligand(s) failed to dock and are excluded.
Receptor preparation: pdb2pqr/PROPKA + meeko at pH 7.4 (443 polar H, template Gasteiger charges).

| Metric | Value |
|---|---|
| ROC AUC, best Vina score | **0.49** (95% CI 0.38-0.61) |
| ROC AUC, heavy-atom count only (control) | 0.50 |
| Enrichment factor, top 10% | 0.7 (random = 1.0) |
| Spearman, score vs pChEMBL (actives only) | 0.03 |
| Primary-sulfonamide fraction, actives / weak | 86% / 6% |
| ROC AUC, 'has a primary sulfonamide' alone (control) | 0.90 |
| ROC AUC, Vina score within the 45 sulfonamides | 0.48 |

Pocket chosen per ligand: {'P1': 98}

| Truth | Top-pose label | Count |
|---|---|---|
| active | MODERATE | 27 |
| active | STRONG | 21 |
| active | WEAK | 1 |
| weak | MODERATE | 12 |
| weak | STRONG | 33 |
| weak | WEAK | 4 |
