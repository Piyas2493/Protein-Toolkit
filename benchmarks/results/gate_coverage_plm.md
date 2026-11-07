# Triage gate coverage on held-out EC data

Target recall 0.95, delta 0.05; 15725 held-out rows in 3659 clusters; 200 random cluster splits per setting. A valid gate holds in >= 95% of splits for the recall it bounds.

| Calibrate on | Gate | Cluster-averaged recall held | Row-weighted recall held | Mean threshold | Mean compute saved |
|---|---|---|---|---|---|
| 20% of clusters | row-level | 20.5% | 62.5% | 0.667 | 8.1% |
| 20% of clusters | cluster-aware | 93.0% | 100.0% | 0.567 | 4.6% |
| 50% of clusters | row-level | 4.0% | 62.5% | 0.668 | 8.0% |
| 50% of clusters | cluster-aware | 88.5% | 100.0% | 0.578 | 5.1% |
