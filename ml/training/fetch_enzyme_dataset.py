"""
Fetch a real UniProt enzyme dataset for retraining.

Strategy:
    - For each EC top-level class (EC1..EC6), query UniProt's
      /uniprotkb/search endpoint with `(ec:N.*) AND reviewed:true`,
      paging through results (UniProt caps a single page at 500 rows)
      via the cursor in the response `Link` header until `per_class`
      valid rows are kept or there are no more pages.
    - Use the TSV format with fields: id, sequence, ec.
    - Filter non-standard residues and length out-of-range.
    - Run `FeatureExtractor.extract(seq)` on each.
    - Save CSV at ml/training/enzyme_dataset.csv.

Usage:
    python -m ml.training.fetch_enzyme_dataset \
        --per-class 90 --out ml/training/enzyme_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.feature_extractor import FeatureExtractor

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_FIELDS = "id,sequence,ec"

USER_AGENT = "ProteinToolkit/1.0 (+contact: local)"

PAGE_SIZE = 500  # UniProt's hard per-request cap
NEXT_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def _build_query(ec_top: int) -> str:
    return f"(ec:{ec_top}.*) AND (reviewed:true)"


def _first_page_url(ec_top: int) -> str:
    return UNIPROT_SEARCH + "?" + urllib.parse.urlencode({
        "format": "tsv",
        "fields": DEFAULT_FIELDS,
        "query": _build_query(ec_top),
        "size": str(PAGE_SIZE),
    })


def _fetch_page(url: str) -> tuple[list[dict], str | None]:
    """Fetch one TSV page. Returns (rows, next_page_url_or_None)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/tab-separated-values",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read().decode("utf-8", errors="replace")
                link = r.headers.get("Link")
            next_url = None
            if link:
                m = NEXT_LINK_RE.search(link)
                next_url = m.group(1) if m else None
            return _parse_tsv(data), next_url
        except (urllib.error.HTTPError,
                urllib.error.URLError, OSError) as e:
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/3 after {wait}s ({e})",
                  file=sys.stderr)
            time.sleep(wait)
    return [], None


def _parse_tsv(data: str) -> list[dict]:
    """Parse UniProt TSV (header line + tab rows)."""
    reader = csv.DictReader(io.StringIO(data), delimiter="\t")
    rows = []
    for row in reader:
        ec = (row.get("EC number") or "").strip()
        seq = (row.get("Sequence") or "").strip().upper()
        acc = (row.get("Entry") or row.get("Entry Name") or "").strip()
        if not acc or not seq or not ec:
            continue
        first_ec = ec.split(";")[0].strip()
        top = first_ec.split(".")[0]
        if not top.isdigit():
            continue
        top_int = int(top)
        rows.append({
            "id": acc,
            "sequence": seq,
            "ec_label": f"EC{top_int}",
        })
    return rows


def _clean_sequence(seq: str) -> str | None:
    seq = "".join(c for c in seq if c.isalpha())
    if len(seq) < 60 or len(seq) > 4000:
        return None
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    non_standard = sum(1 for c in seq if c not in standard)
    if non_standard / max(len(seq), 1) > 0.02:
        return None
    return seq


def fetch_dataset(
    per_class: int,
    *,
    only: set[int] | None = None,
    existing_raw: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Fetch per_class rows per EC class. Returns (dataset_with_features,
    raw_sequences).

    A failure on a single EC class does NOT abort the run. We log it,
    continue with whatever classes we got, and emit the partial result.

    Args:
        per_class: target rows per EC class.
        only: optional set of EC top numbers to fetch. If None, fetches all.
        existing_raw: optional list of already-fetched raw sequence rows
            to seed the result.
    """
    extractor = FeatureExtractor()
    raw = list(existing_raw) if existing_raw else []
    # Re-derive features for any merged-in rows and seed per-label seen-ID
    # sets so re-fetching a class already in `existing_raw` tops it up
    # instead of duplicating rows already present.
    dataset = []
    seen_by_label: dict[str, set[str]] = {}
    for row in raw:
        seen_by_label.setdefault(row["Label"], set()).add(row["ID"])
        dataset.append({"ID": row["ID"], "Label": row["Label"],
                        **extractor.extract(row["Sequence"])})
    failures: list[tuple[int, str]] = []
    classes = sorted(only) if only is not None else list(range(1, 7))
    # ponytail: hard cap on pages so a huge --per-class can't loop forever;
    # raise if you legitimately need more than ~60000 scanned rows/class.
    max_pages = 120
    for ec_top in classes:
        label = f"EC{ec_top}"
        try:
            print(f"Fetching {label} ...", file=sys.stderr)
            seen = seen_by_label.setdefault(label, set())
            kept = len(seen)
            url: str | None = _first_page_url(ec_top)
            pages = 0
            while url and kept < per_class and pages < max_pages:
                rows, url = _fetch_page(url)
                pages += 1
                print(f"  page {pages}: raw rows {len(rows)}",
                      file=sys.stderr)
                for row in rows:
                    if kept >= per_class:
                        break
                    if row["id"] in seen:
                        continue
                    if row["ec_label"] != label:
                        # `ec:N.*` matches any EC number on the entry, not
                        # just its first one — a multi-EC protein (e.g.
                        # "1.1.1.1; 5.3.1.9") shows up in more than one
                        # class's query. Only keep it under its true
                        # primary class to avoid the same sequence getting
                        # contradictory labels.
                        continue
                    seq = _clean_sequence(row["sequence"])
                    if seq is None:
                        continue
                    seen.add(row["id"])
                    try:
                        features = extractor.extract(seq)
                    except Exception as e:
                        print(f"  skip {row['id']}: {e}", file=sys.stderr)
                        continue
                    dataset.append({"ID": row["id"], "Label": label,
                                    **features})
                    raw.append({"ID": row["id"], "Label": label,
                                "Sequence": seq})
                    kept += 1
                time.sleep(0.34)
            print(f"  accepted: {kept} ({pages} page(s))", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED on {label}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            failures.append((ec_top, str(e)))
    if failures:
        print(f"\nFinished with {len(failures)} failed class(es): "
              f"{failures}", file=sys.stderr)
    return dataset, raw


def write_csv(dataset: list[dict], out_path: Path) -> None:
    if not dataset:
        raise RuntimeError("Empty dataset — nothing to write.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_keys = sorted(k for k in dataset[0].keys()
                          if k not in ("ID", "Label"))
    cols = ["ID", "Label"] + feature_keys
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)
    print(f"Wrote {len(dataset)} rows -> {out_path}", file=sys.stderr)


def write_sequences(
    sequences: list[dict], out_path: Path
) -> None:
    """Persist raw sequences to a sidecar CSV so future feature
    additions don't require re-hitting UniProt."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ID", "Label", "Sequence"]
        )
        writer.writeheader()
        for row in sequences:
            writer.writerow(row)
    print(f"Wrote {len(sequences)} sequences -> {out_path}",
          file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument(
        "--out",
        default=str(ROOT / "ml" / "training" / "enzyme_dataset.csv"),
    )
    parser.add_argument(
        "--seq-out",
        default=str(ROOT / "ml" / "training" / "enzyme_sequences.csv"),
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated EC top numbers to fetch (e.g., '6' or '5,6').",
    )
    parser.add_argument(
        "--merge-with", type=str, default=None,
        help="Existing sequence CSV to merge into the new one.",
    )
    args = parser.parse_args(argv)
    only: set[int] | None = None
    if args.only:
        only = {int(x) for x in args.only.split(",") if x.strip()}
    existing = None
    if args.merge_with:
        existing_path = Path(args.merge_with)
        if existing_path.exists():
            with existing_path.open(newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
            print(f"Loaded {len(existing)} existing sequence rows from "
                  f"{existing_path}", file=sys.stderr)
    dataset, raw = fetch_dataset(
        args.per_class, only=only, existing_raw=existing,
    )
    if not dataset:
        print("ERROR: nothing fetched.", file=sys.stderr)
        return 2
    write_csv(dataset, Path(args.out))
    write_sequences(raw, Path(args.seq_out))
    return 0


if __name__ == "__main__":
    main()
