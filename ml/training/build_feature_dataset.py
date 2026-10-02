"""
Build the extended feature CSV from cached sequences.

Reads ml/training/enzyme_sequences.csv (raw sequences written by
fetch_enzyme_dataset) and emits ml/training/enzyme_features.csv
with whatever schema `FeatureExtractor.extract(..., include_dipeptides=...)`
produces.

This avoids re-hitting UniProt when we add new feature types.

Usage:
    python -m ml.training.build_feature_dataset \
        --in  ml/training/enzyme_sequences.csv \
        --out ml/training/enzyme_features.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.feature_extractor import FeatureExtractor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_in = ROOT / "ml" / "training" / "enzyme_sequences.csv"
    default_out = ROOT / "ml" / "training" / "enzyme_features.csv"
    parser.add_argument("--in", dest="in_path", default=str(default_in))
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument(
        "--no-dipeptides", action="store_true",
        help="Disable dipeptide features (match the original schema).",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: sequence CSV not found: {in_path}", file=sys.stderr)
        return 2

    extractor = FeatureExtractor()
    include_dipeptides = not args.no_dipeptides

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with in_path.open(newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        rows = list(reader)
        if not rows:
            print("ERROR: sequence CSV is empty.", file=sys.stderr)
            return 2
        # Discover column names from the first row, in deterministic order
        feature_keys = sorted(extractor.extract(
            "MKTLLLTLVVVTIVCLDLGYTFQPQNGQFICTTAG",
            include_dipeptides=include_dipeptides,
        ).keys())
        cols = ["ID", "Label"] + feature_keys
        writer = csv.DictWriter(fout, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            seq = row.get("Sequence", "").strip()
            if not seq:
                continue
            feats = extractor.extract(
                seq, include_dipeptides=include_dipeptides
            )
            writer.writerow({"ID": row["ID"], "Label": row["Label"],
                             **feats})
            written += 1
    print(f"Wrote {written} rows ({'extended' if include_dipeptides else 'base'} features) -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
