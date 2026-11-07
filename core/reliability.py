"""
Reliability notes: let the toolkit's outputs quote its own measured accuracy.

Instead of a generic "this is only a prediction" disclaimer, a report can say
what the committed benchmarks actually measured for the component it used
(docking and screening, EC classification). The numbers are read from
benchmarks/results/*.json at call time, so they cannot drift from the
benchmarks; if a file is missing the note is simply omitted and the caller
falls back to its generic disclaimer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "results"
SUCCESS_RMSD = 2.0


def _load(name: str) -> Optional[dict]:
    try:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def docking_notes() -> List[str]:
    """What benchmarks/ measured for pose reproduction and compound ranking."""
    notes = []
    red = _load("redocking.json")
    if red:
        s = red["summary"]
        notes.append(
            f"Pose reproduction with a known site: {s['native_top1_success']}/"
            f"{s['native_runs']} top-ranked poses within {SUCCESS_RMSD:g} A of the "
            f"crystal pose ({len(red['complexes'])} complexes).")
        blind = [c["blind"] for c in red["complexes"] if c.get("blind")]
        if blind:
            found = sum(1 for b in blind if b.get("pocket_found")
                        and b["pocket_dist"] <= 4.0)
            notes.append(
                f"Blind site finding: the detected pocket was within 4 A of the "
                f"true site in {found}/{len(blind)} complexes; give a known site "
                "or box rather than relying on pocket detection.")
    parts = []
    for target in ("ca2", "esr1"):
        d = _load(f"screening_{target}.json")
        if d:
            lo, hi = d["roc_auc_95ci"]
            parts.append(f"{d['pdb']} ROC AUC {d['roc_auc']:.2f} "
                         f"(95% CI {lo:.2f}-{hi:.2f}), top-10% enrichment "
                         f"{d['ef_top10pct']:.1f}x")
    if parts:
        notes.append(
            "Ranking potent vs weak binders of one target (measured actives vs "
            "size-matched weak binders; random = AUC 0.50, enrichment 1.0x): "
            + "; ".join(parts) + ". Treat docking ranks as weak evidence of potency.")
    return notes


def ec_notes() -> List[str]:
    """What benchmarks/ measured for the EC classifier, by novelty tier."""
    d = _load("novelty_calibration.json")
    if not d:
        return []
    t = d["tiers"]
    return [
        f"EC class accuracy on {d['n_heldout']} held-out sequences from unseen "
        f"clusters: {d['accuracy']:.0%} overall, "
        + ", ".join(f"{t[k]['accuracy']:.0%} for {k}" for k in ("novel", "related", "familiar"))
        + " sequences (by similarity to the nearest training family)."]


def format_notes(notes: List[str], title: str = "Measured reliability") -> str:
    if not notes:
        return ""
    return title + ":\n" + "\n".join(f"  - {n}" for n in notes)
