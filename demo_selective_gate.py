"""Self-contained demonstration of the risk-controlled gate.

Simulates a triage stage: true enzymes tend to score higher than
non-enzymes, but the two distributions overlap (realistic). The gate is
fitted on a calibration draw to guarantee >= 95% recall of true enzymes,
then checked on 200 fresh independent draws it never saw. Reports how often
the guarantee held and how much structural compute the gate saved.
"""
import numpy as np
from ml.selective_gate import RiskControlledGate

rng = np.random.default_rng(0)

def draw(n_pos=400, n_neg=1600):
    # scores in [0,1]; positives shifted higher but overlapping
    pos = np.clip(rng.beta(6, 3, n_pos), 0, 1)
    neg = np.clip(rng.beta(3, 6, n_neg), 0, 1)
    scores = np.concatenate([pos, neg])
    positive = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    return scores, positive

TARGET, DELTA = 0.95, 0.05

cal_scores, cal_pos = draw()
gate = RiskControlledGate(target_recall=TARGET, delta=DELTA).fit(cal_scores, cal_pos)
print("Calibration:", gate.report_.summary())

held = [gate.evaluate(*draw()) for _ in range(200)]
recalls = np.array([h["realized_recall"] for h in held])
savings = np.array([h["compute_saving"] for h in held])
held_rate = np.mean(recalls >= TARGET)

print(f"\nOver 200 fresh test draws the gate never saw:")
print(f"  realized recall: mean {recalls.mean():.3f}, min {recalls.min():.3f}")
print(f"  guarantee (recall >= {TARGET:.0%}) held in {held_rate:.1%} of draws "
      f"(target confidence 1 - delta = {1-DELTA:.0%})")
print(f"  structural compute saved: mean {savings.mean():.1%} of sequences skipped")

# Contrast: a naive 'keep top 50%' gate ignores the miss rate entirely.
naive_recalls = []
for _ in range(200):
    s, p = draw()
    thr = np.quantile(s, 0.5)
    keep = s >= thr
    naive_recalls.append((keep & p).sum() / p.sum())
naive_recalls = np.array(naive_recalls)
print(f"\nNaive 'keep top 50%' gate for comparison:")
print(f"  realized recall: mean {naive_recalls.mean():.3f}, "
      f"below {TARGET:.0%} in {np.mean(naive_recalls < TARGET):.0%} of draws "
      f"(no guarantee, user cannot set the miss rate)")
