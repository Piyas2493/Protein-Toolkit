"""
Sequence-first batch triage.

Runs the sequence-only half of the pipeline (stats, motifs, catalytic
residues, rule-based + ML enzyme classification, combined confidence)
on every record in a FASTA file, with no structure or docking required.
Ranks results so you can screen thousands of sequences cheaply and only
run structure analysis / docking on the handful that triage well.

Usage:
    python triage.py sequences.fasta --out triage.csv --top 20

Risk-controlled gate (optional):
    Instead of a hand-picked --top cutoff, you can let the gate choose the
    cutoff for you so that a target fraction of correctly-classifiable
    enzymes is kept, with a distribution-free confidence guarantee. First
    export calibration data from a held-out evaluation:

        python -m ml.evaluation --gate-calibration outputs/gate_calib.csv

    then triage with a target miss rate:

        python triage.py sequences.fasta --out triage.csv \\
            --gate-calibration outputs/gate_calib.csv \\
            --target-recall 0.95 --delta 0.05

    Every row gets a `passes_gate` flag and the chosen `gate_threshold`; with
    a gate and no --top, only passing rows are written.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.fasta import FASTAParser
from core.validator import SequenceValidator
from core.statistics import SequenceStatistics
from core.motifs import MotifDatabase
from core.catalytic import CatalyticPredictor
from core.enzymes_classifier import EnzymeClassifier
from core.scoring import ConfidenceScorer
from ml.novelty import load_for_model
from ml.predictor import EnzymePredictor

DEFAULT_ML_MODEL = ROOT / "ml" / "models" / "enzyme_classifier_xgb.pkl"


def triage_one(record, motif_db, catalytic_predictor, enzyme_classifier,
                scorer, ml_predictor, novelty=None):
    seq = SequenceValidator.clean(record.sequence)
    validation = SequenceValidator.validate(seq)
    if not validation["valid"]:
        return {
            "header": record.header, "length": len(seq),
            "valid": False, "error": "unsupported amino acids",
        }

    stats = SequenceStatistics(seq)
    motif_hits = motif_db.search(seq)
    catalytic_hits = catalytic_predictor.predict(seq)
    enzyme_predictions = enzyme_classifier.classify(record.header, motif_hits)

    ml_class, ml_confidence = None, None
    if ml_predictor is not None:
        try:
            result = ml_predictor.predict_sequence(seq)
            ml_class = result["prediction"]
            ml_confidence = max(result["probabilities"].values())
        except Exception:
            pass

    confidence = scorer.calculate(
        motif_hits=motif_hits, catalytic_hits=catalytic_hits,
        enzyme_predictions=enzyme_predictions, statistics=stats,
    )

    extra = {}
    if novelty is not None and ml_confidence:
        nv = novelty.assess(seq, ml_confidence)
        extra = {"nn_similarity": nv["nn_similarity"], "novelty": nv["novelty"],
                 "adj_confidence": nv["adj_confidence"]}

    return {
        **extra,
        "header": record.header,
        "length": len(seq),
        "valid": True,
        "motifs": len(motif_hits),
        "catalytic_hits": len(catalytic_hits),
        "rule_enzyme_class": enzyme_predictions[0]["class"]
            if enzyme_predictions else "",
        "ml_enzyme_class": ml_class or "",
        "ml_confidence": round(ml_confidence, 4) if ml_confidence else "",
        "triage_score": confidence["confidence"],
    }


def run_triage(fasta_path: str, ml_model_path: str | None = DEFAULT_ML_MODEL):
    records = FASTAParser.load(fasta_path)

    motif_db = MotifDatabase()
    catalytic_predictor = CatalyticPredictor()
    enzyme_classifier = EnzymeClassifier()
    scorer = ConfidenceScorer()

    ml_predictor = None
    if ml_model_path and Path(ml_model_path).exists():
        ml_predictor = EnzymePredictor(str(ml_model_path))

    novelty = load_for_model(ml_model_path) if ml_predictor is not None else None
    if ml_predictor is not None and novelty is None:
        print("Note: no novelty model built for this classifier (python -m "
              "ml.novelty --build), so nn_similarity / novelty / adj_confidence "
              "are left blank.", file=sys.stderr)

    rows = [
        triage_one(r, motif_db, catalytic_predictor, enzyme_classifier,
                   scorer, ml_predictor, novelty)
        for r in records
    ]
    rows.sort(key=lambda r: r.get("triage_score", -1), reverse=True)
    return rows


def _fit_gate(calibration_csv: str, target_recall: float, delta: float):
    """Fit a RiskControlledGate on exported (ml_confidence, correct)
    calibration pairs. Returns the fitted gate, or raises on a bad file."""
    import numpy as np
    from ml.selective_gate import RiskControlledGate

    scores, correct, clusters = [], [], []
    with open(calibration_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "ml_confidence" not in reader.fieldnames \
                or "correct" not in reader.fieldnames:
            raise ValueError(
                "calibration file needs columns 'ml_confidence' and 'correct'")
        clustered = "cluster" in reader.fieldnames
        for row in reader:
            try:
                s, c = float(row["ml_confidence"]), float(row["correct"]) >= 0.5
            except (TypeError, ValueError):
                continue
            if clustered and not row["cluster"]:
                continue
            scores.append(s)
            correct.append(c)
            clusters.append(row["cluster"] if clustered else None)
    if not scores:
        raise ValueError("no usable rows in calibration file")

    if not clustered:
        print("Warning: the calibration file has no 'cluster' column, so rows are "
              "treated as independent. For homologous sequences that makes the "
              "guarantee over-confident; re-export with `python -m "
              "ml.evaluation --gate-calibration` after running "
              "`python -m ml.training.cluster_sequences`.", file=sys.stderr)

    gate = RiskControlledGate(target_recall=target_recall, delta=delta)
    gate.fit(np.asarray(scores), np.asarray(correct, dtype=bool),
             groups=np.asarray(clusters) if clustered else None)
    return gate


def apply_gate(rows, gate):
    """Annotate each valid row with passes_gate / gate_threshold. A valid row
    with no ml_confidence (ML unavailable) is kept and flagged 'unscored',
    never silently dropped."""
    threshold = gate.threshold_
    for row in rows:
        if not row.get("valid"):
            continue
        row["gate_threshold"] = round(float(threshold), 4)
        conf = row.get("ml_confidence")
        if conf == "" or conf is None:
            row["passes_gate"] = "unscored"
        else:
            row["passes_gate"] = bool(float(conf) >= threshold)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fasta", help="FASTA file with one or more sequences")
    parser.add_argument("--out", default="triage.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=None,
                        help="Only write the top N by triage score")
    parser.add_argument("--model", default=str(DEFAULT_ML_MODEL),
                        help="ML enzyme classifier path (skip with --no-ml)")
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--gate-calibration", default=None,
                        help="CSV of held-out (ml_confidence, correct) pairs "
                             "from `python -m ml.evaluation --gate-calibration`. "
                             "Enables the risk-controlled gate.")
    parser.add_argument("--target-recall", type=float, default=0.95,
                        help="Gate: fraction of correctly-classifiable enzymes "
                             "to keep (default 0.95)")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Gate: confidence level is 1 - delta (default 0.05)")
    args = parser.parse_args(argv)

    # Fit the gate first and fail closed: a requested gate that can't be
    # built must not quietly turn into unfiltered output.
    gate = None
    if args.gate_calibration:
        try:
            gate = _fit_gate(args.gate_calibration, args.target_recall, args.delta)
        except (OSError, ValueError) as exc:
            print(f"Cannot build the gate from {args.gate_calibration}: {exc}",
                  file=sys.stderr)
            return 2
        print(gate.report_.summary())

    rows = run_triage(args.fasta, None if args.no_ml else args.model)
    n_scored = len(rows)
    if gate is not None:
        rows = apply_gate(rows, gate)

    # A gate with no explicit --top keeps only passing rows (plus unscored).
    if gate is not None and args.top is None:
        rows = [r for r in rows
                if not r.get("valid") or r.get("passes_gate") in (True, "unscored")]
    if args.top:
        rows = rows[:args.top]

    if not rows:
        print("No sequences found." if not n_scored else
              f"The gate removed all {n_scored} sequence(s).")
        return 1

    fieldnames = ["header", "length", "valid", "motifs", "catalytic_hits",
                  "rule_enzyme_class", "ml_enzyme_class", "ml_confidence",
                  "nn_similarity", "novelty", "adj_confidence",
                  "triage_score", "passes_gate", "gate_threshold", "error"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Triaged {len(rows)} sequence(s) -> {args.out}")
    print(f"Top candidate: {rows[0]['header']!r} "
          f"(triage_score={rows[0].get('triage_score', '-')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
