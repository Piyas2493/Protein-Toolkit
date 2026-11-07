# Novelty-aware confidence (EC classifier)

15725 held-out sequences (no cluster shared with training), compared with
14642 training families. Tiers split held-out sequences by similarity
to the nearest training family: novel < 0.0185 <= related
< 0.0626 <= familiar. Adjusted figures are out-of-fold
(grouped by sequence cluster).

| Metric | Raw confidence | Novelty-adjusted |
|---|---|---|
| Expected calibration error | 0.0445 | 0.0188 |
| Brier (is the prediction right?) | 0.1704 | 0.1648 |
| ROC AUC (is the prediction right?) | 0.804 | 0.813 |

| Tier | n | Accuracy | Mean raw conf. | Mean adj. conf. | ECE raw | ECE adj. |
|---|---|---|---|---|---|---|
| novel | 3141 | 45.6% | 0.61 | 0.45 | 0.154 | 0.049 |
| related | 9439 | 65.0% | 0.69 | 0.66 | 0.038 | 0.015 |
| familiar | 3145 | 89.6% | 0.85 | 0.88 | 0.044 | 0.038 |
