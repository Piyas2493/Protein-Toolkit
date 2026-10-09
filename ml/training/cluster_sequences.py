"""
Greedy k-mer sequence clustering, used to keep homologs on the same side
of a train/test split.

A random row-level split puts near-identical sequences (isoforms,
orthologs, paralogs) in both train and test, so a classifier can score
well by recognizing a training neighbor rather than by generalizing. This
groups sequences so that a cluster is never split.

Algorithm (CD-HIT-like): visit sequences longest-first; a sequence joins
the cluster of the representative sharing the most 5-mers with it if the
shared fraction of ITS 5-mers is >= `threshold`, otherwise it founds a
new cluster.

What the threshold means: if two sequences are aligned end to end at
identity p, a 5-mer survives with probability ~p^5, so a shared-5-mer
fraction f corresponds to roughly p = f^(1/5):
    f = 0.05  ->  p ~ 0.55      f = 0.10  ->  p ~ 0.63
Chance sharing between unrelated ~400-residue sequences is ~0.05
expected 5-mers, i.e. negligible. This is a coarser, k-mer proxy for
identity-based clustering — it will NOT catch remote homologs at the
30-40% identity level that alignment-based tools (CD-HIT, MMseqs2) do.
Use those, if available, for a stricter split.

Usage:
    python -m ml.training.cluster_sequences
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

K = 5


def kmers(seq: str, k: int = K) -> set:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def max_containment(seq: str, index: dict, rep_sizes: dict):
    """(rep_id, shared_fraction_of_seq_kmers) for the best-matching
    representative in the inverted index, or (None, 0.0)."""
    km = kmers(seq)
    if not km:
        return None, 0.0
    hits = Counter()
    for kmer in km:
        for rep in index.get(kmer, ()):
            hits[rep] += 1
    if not hits:
        return None, 0.0
    rep, shared = hits.most_common(1)[0]
    return rep, shared / len(km)


def cluster(sequences: dict, threshold: float = 0.05) -> dict:
    """sequences: {id: sequence}. Returns {id: cluster_id}."""
    index: dict = defaultdict(list)
    rep_sizes: dict = {}
    assignment: dict = {}
    order = sorted(sequences, key=lambda i: len(sequences[i]), reverse=True)
    for n, sid in enumerate(order):
        seq = sequences[sid]
        rep, frac = max_containment(seq, index, rep_sizes)
        if rep is not None and frac >= threshold:
            assignment[sid] = assignment[rep]
            continue
        assignment[sid] = sid          # founds its own cluster
        km = kmers(seq)
        rep_sizes[sid] = len(km)
        for kmer in km:
            index[kmer].append(sid)
        if n and n % 10000 == 0:
            print(f"  clustered {n}/{len(order)}", file=sys.stderr)
    return assignment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default=str(
        ROOT / "ml" / "training" / "enzyme_sequences.csv"))
    parser.add_argument("--out", default=str(
        ROOT / "ml" / "training" / "enzyme_clusters.csv"))
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    with open(args.seqs, newline="", encoding="utf-8") as f:
        sequences = {row["ID"]: row["Sequence"] for row in csv.DictReader(f)}
    print(f"Clustering {len(sequences)} sequences "
          f"(k={K}, threshold={args.threshold}) ...", file=sys.stderr)

    assignment = cluster(sequences, args.threshold)
    n_clusters = len(set(assignment.values()))
    sizes = Counter(assignment.values())
    print(f"{n_clusters} clusters; largest {max(sizes.values())}; "
          f"singletons {sum(1 for s in sizes.values() if s == 1)}",
          file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Cluster"])
        for sid, cid in assignment.items():
            writer.writerow([sid, cid])
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
