# EC classifier on ESM-1b embeddings

Cluster-held-out split (same as the XGBoost model): 62900 train / 15725 test sequences.
Model chosen on a cluster-grouped validation split inside train; temperature fitted there; test used once.

| Model | Validation accuracy |
|---|---|
| logreg C=0.01 | 92.5% |
| logreg C=0.1 | 91.8% |
| mlp-1024 (5 epochs) | 94.3% |

Chosen: **mlp-1024 (5 epochs)**, temperature 1.758.

| Metric (held-out test) | ESM-1b classifier | XGBoost on sequence features |
|---|---|---|
| Accuracy | **92.4%** | 66.1% |
| Macro F1 | 0.921 | |
| Mean confidence | 0.933 | |
| Expected calibration error (after temperature) | 0.010 | 0.044 |
| Expected calibration error (before temperature) | 0.043 | |
| Brier score | 0.115 | |

| Class | Recall |
|---|---|
| EC1 | 94% |
| EC2 | 92% |
| EC3 | 94% |
| EC4 | 82% |
| EC5 | 92% |
| EC6 | 98% |
