"""
Sequence-first batch triage.

Runs the sequence-only half of the pipeline (stats, motifs, catalytic
residues, rule-based + ML enzyme classification, combined confidence)
on every record in a FASTA file, with no structure or docking required.
Ranks results so you can screen thousands of sequences cheaply and only
run structure analysis / docking on the handful that triage well.

Usage:
    python triage.py sequences.fasta --out triage.csv --top 20
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
from ml.predictor import EnzymePredictor

DEFAULT_ML_MODEL = ROOT / "ml" / "models" / "enzyme_classifier_xgb.pkl"


def triage_one(record, motif_db, catalytic_predictor, enzyme_classifier,
                scorer, ml_predictor):
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

    return {
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

    rows = [
        triage_one(r, motif_db, catalytic_predictor, enzyme_classifier,
                   scorer, ml_predictor)
        for r in records
    ]
    rows.sort(key=lambda r: r.get("triage_score", -1), reverse=True)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", help="FASTA file with one or more sequences")
    parser.add_argument("--out", default="triage.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=None,
                        help="Only write the top N by triage score")
    parser.add_argument("--model", default=str(DEFAULT_ML_MODEL),
                        help="ML enzyme classifier path (skip with --no-ml)")
    parser.add_argument("--no-ml", action="store_true")
    args = parser.parse_args(argv)

    rows = run_triage(args.fasta, None if args.no_ml else args.model)
    if args.top:
        rows = rows[:args.top]

    if not rows:
        print("No sequences found.")
        return 1

    fieldnames = ["header", "length", "valid", "motifs", "catalytic_hits",
                  "rule_enzyme_class", "ml_enzyme_class", "ml_confidence",
                  "triage_score", "error"]
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
