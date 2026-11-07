# Novelty-aware confidence (EC classifier: enzyme_classifier_plm.pkl)

15725 held-out sequences (no cluster shared with training), compared with
14642 training families. Tiers split held-out sequences by similarity
to the nearest training family: novel < 0.0185 <= related
< 0.0626 <= familiar. Adjusted figures are out-of-fold
(grouped by sequence cluster).

| Metric | Raw confidence | Novelty-adjusted |
|---|---|---|
| Expected calibration error | 0.0096 | 0.0098 |
| Brier (is the prediction right?) | 0.0503 | 0.0516 |
| ROC AUC (is the prediction right?) | 0.922 | 0.917 |

| Tier | n | Accuracy | Mean raw conf. | Mean adj. conf. | ECE raw | ECE adj. |
|---|---|---|---|---|---|---|
| novel | 3141 | 82.7% | 0.86 | 0.83 | 0.032 | 0.016 |
| related | 9439 | 93.6% | 0.94 | 0.94 | 0.014 | 0.018 |
| familiar | 3145 | 98.3% | 0.98 | 0.98 | 0.010 | 0.010 |
