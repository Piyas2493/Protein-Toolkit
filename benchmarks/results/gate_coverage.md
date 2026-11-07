# Triage gate coverage on held-out EC data

Target recall 0.95, delta 0.05; 15725 held-out rows in 3659 clusters; 200 random cluster splits per setting. A valid gate holds in >= 95% of splits for the recall it bounds.

| Calibrate on | Gate | Cluster-averaged recall held | Row-weighted recall held | Mean threshold | Mean compute saved |
|---|---|---|---|---|---|
| 20% of clusters | row-level | 15.5% | 75.0% | 0.394 | 9.4% |
| 20% of clusters | cluster-aware | 94.5% | 100.0% | 0.349 | 4.8% |
| 50% of clusters | row-level | 1.5% | 67.0% | 0.399 | 9.9% |
| 50% of clusters | cluster-aware | 92.5% | 100.0% | 0.357 | 5.5% |
