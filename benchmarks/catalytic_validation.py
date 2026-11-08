"""
Catalytic-rule validation: when the rule layer says "serine protease", is it one?

core/catalytic.py names a catalytic system (serine / cysteine / aspartic /
metalloprotease). Ground truth for that is in UniProt: an enzyme's EC subclass
says which protease type it is (3.4.21 serine, 3.4.22 cysteine, 3.4.23 aspartic,
3.4.24 metallo-). So this fetches reviewed sequences of each type, plus negatives
that are NOT peptidases (other hydrolases, transferases, oxidoreductases, ...),
and measures, per rule:

  recall     of its own protease type
  false-hit  rate on the negatives  (and on the other protease types)

It scores two engines on the same sequences: the old composition-only rule
(fires whenever the protein merely contains enough His/Ser/Asp... residues) and
the current engine in core/catalytic.py (needs a conserved active-site motif).

Usage:
    python -m benchmarks.catalytic_validation fetch
    python -m benchmarks.catalytic_validation evaluate
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATABASE_RULES = ROOT / "database" / "catalytic_sites.json"
DATA = Path(__file__).parent / "data" / "catalytic_validation.csv"
HOLDOUT = Path(__file__).parent / "data" / "catalytic_validation_holdout.csv"
RESULTS = Path(__file__).parent / "results"
PER_CLASS = 400
SEED = 0

# label -> UniProt query
POSITIVES = {
    "Serine Protease Triad": "(reviewed:true) AND (ec:3.4.21.*)",
    "Cysteine Protease Dyad": "(reviewed:true) AND (ec:3.4.22.*)",
    "Aspartic Protease": "(reviewed:true) AND (ec:3.4.23.*)",
    "Metalloprotease": "(reviewed:true) AND (ec:3.4.24.*)",
}
NEGATIVES = {
    "neg:other hydrolase (EC 3.1/3.2/3.5/3.6)": "(reviewed:true) AND (ec:3.1.* OR ec:3.2.* OR ec:3.5.* OR ec:3.6.*) NOT (ec:3.4.*)",
    "neg:oxidoreductase (EC 1)": "(reviewed:true) AND (ec:1.*) NOT (ec:3.4.*)",
    "neg:transferase (EC 2)": "(reviewed:true) AND (ec:2.*) NOT (ec:3.4.*)",
    "neg:lyase/isomerase/ligase (EC 4-6)": "(reviewed:true) AND (ec:4.* OR ec:5.* OR ec:6.*) NOT (ec:3.4.*)",
}


def _fetch(query: str, want: int, rng: random.Random):
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": query, "format": "tsv", "fields": "accession,sequence,ec", "size": 500})
    rows = []
    while url and len(rows) < 3 * want:
        req = urllib.request.Request(url, headers={"User-Agent": "ProteinToolkit-benchmark/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            text, link = r.read().decode(), r.headers.get("Link", "")
        rows += [l.split("\t") for l in text.strip().split("\n")[1:]]
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    rows = [r for r in rows if len(r) >= 3 and 50 <= len(r[1]) <= 2000]
    rng.shuffle(rows)
    return rows[:want]


def fetch(holdout: bool = False) -> None:
    """The motifs were chosen while looking at the primary set, so a second
    draw (different seed, accessions already used excluded) is kept apart to
    check them out of sample."""
    rng = random.Random(SEED + (1 if holdout else 0))
    seen = set()
    if holdout:
        seen = {r["accession"] for r in csv.DictReader(open(DATA, encoding="utf-8"))}
    out = HOLDOUT if holdout else DATA
    want = 250 if holdout else PER_CLASS
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["accession", "truth", "ec", "sequence"])
        for table in (POSITIVES, NEGATIVES):
            for label, q in table.items():
                rows = [r for r in _fetch(q, 4 * want if holdout else want, rng) if r[0] not in seen][:want]
                for acc, seq, ec in rows:
                    w.writerow([acc, label, ec, seq])
                print(f"  {label}: {len(rows)}", file=sys.stderr)


def composition_only_hits(sequence: str):
    """The OLD engine, kept here as the baseline: a rule fires when the
    sequence has at least the required count of each residue type."""
    from core.catalytic import AA3, DATABASE

    rules = json.load(open(DATABASE, encoding="utf-8"))
    counts = Counter(AA3.get(x) for x in sequence.upper() if x in AA3)
    return [r["name"] for r in rules
            if all(counts[res] >= n for res, n in Counter(r["residues"]).items())]


def zinc_claims(rows, rule_hits_per_row, function_class):
    """Share of sequences, per truth class, that the downstream predictors tell
    a user is a ZINC metalloprotease (function) and binds Zn2+ (ligand).
    `function_class` is the enzyme_class the function/ligand rules key on:
    "Hydrolase" was the old, too-generic value; "Metalloprotease" is current."""
    import copy

    from core.enzymes_classifier import EnzymeClassifier
    from core.function_prediction import FunctionPredictor
    from core.ligand_prediction import LigandPredictor
    from core.motifs import MotifDatabase

    rules = json.load(open(DATABASE_RULES, encoding="utf-8"))
    by_name = {r["name"]: r for r in rules}
    motifs, enzymes = MotifDatabase(), EnzymeClassifier()
    functions, ligands = FunctionPredictor(), LigandPredictor()
    for rule in functions.rules:
        if rule["name"] == "Zinc Metalloprotease":
            rule["enzyme_class"] = function_class
    for lig in ligands.database:
        if lig["name"] == "Zn2+":
            lig["enzyme_classes"] = [function_class]

    out = {}
    for r, names in zip(rows, rule_hits_per_row):
        mh = motifs.search(r["sequence"])
        ch = [copy.deepcopy(by_name[n]) for n in names]
        ep = enzymes.classify("", mh)
        fp = functions.predict(mh, ep, ch)
        lp = ligands.predict(mh, ep, fp, ch)
        slot = out.setdefault(r["truth"], [0, 0, 0])
        slot[0] += 1
        slot[1] += any(f["function"] == "Zinc Metalloprotease" for f in fp)
        slot[2] += any(l["ligand"] == "Zn2+" for l in lp)
    return {k: {"n": v[0], "zinc_function": v[1] / v[0], "zn_ligand": v[2] / v[0]} for k, v in out.items()}


def evaluate(path: Path = DATA) -> dict:
    from core.catalytic import CatalyticPredictor

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    engine = CatalyticPredictor()
    rules = [r["name"] for r in json.load(open(ROOT / "database" / "catalytic_sites.json", encoding="utf-8"))]
    hits = {"composition": [], "motif": []}
    for r in rows:
        hits["composition"].append(set(composition_only_hits(r["sequence"])))
        hits["motif"].append({h["name"] for h in engine.predict(r["sequence"])})

    out = {"n": len(rows), "classes": dict(Counter(r["truth"] for r in rows)), "engines": {}}
    for eng, hs in hits.items():
        per_rule = {}
        for rule in rules:
            pos = [i for i, r in enumerate(rows) if r["truth"] == rule]
            neg = [i for i, r in enumerate(rows) if r["truth"].startswith("neg:")]
            other = [i for i, r in enumerate(rows) if r["truth"] in rules and r["truth"] != rule]
            per_rule[rule] = {
                "recall": sum(rule in hs[i] for i in pos) / len(pos),
                "false_hit_on_non_peptidases": sum(rule in hs[i] for i in neg) / len(neg),
                "false_hit_on_other_protease_types": sum(rule in hs[i] for i in other) / len(other),
            }
            fired = sum(rule in h for h in hs)
            tp = sum(rule in hs[i] for i in pos)
            per_rule[rule]["precision_vs_all_sequences"] = tp / fired if fired else None
        # how many different catalytic systems does a typical sequence get?
        out["engines"][eng] = {
            "per_rule": per_rule,
            "mean_rules_fired_per_non_peptidase": sum(
                len(hs[i]) for i, r in enumerate(rows) if r["truth"].startswith("neg:"))
            / max(1, sum(r["truth"].startswith("neg:") for r in rows)),
            "pct_non_peptidases_with_any_hit": 100 * sum(
                bool(hs[i]) for i, r in enumerate(rows) if r["truth"].startswith("neg:"))
            / max(1, sum(r["truth"].startswith("neg:") for r in rows)),
        }
    out["downstream_zinc_claims"] = {
        "old (composition rule + function keyed to any Hydrolase)": zinc_claims(rows, hits["composition"], "Hydrolase"),
        "new (motif rule + function keyed to Metalloprotease)": zinc_claims(rows, hits["motif"], "Metalloprotease"),
    }
    stem = "catalytic_validation" + ("_holdout" if path == HOLDOUT else "")
    out["data"] = Path(path).name
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stem}.json").write_text(json.dumps(out, indent=2))
    (RESULTS / f"{stem}.md").write_text(to_markdown(out), encoding="utf-8")
    return out


def to_markdown(o: dict) -> str:
    lines = ["# Catalytic rule validation against UniProt EC subclasses"
             + (" (held-out draw)" if "holdout" in o.get("data", "") else ""), "",
             f"{o['n']} reviewed sequences: " + ", ".join(f"{k} ({v})" for k, v in o["classes"].items()) + ".", "",
             "`composition` = the old rule (enough His/Ser/Asp... residues anywhere); "
             "`motif` = the current engine (needs a conserved active-site motif). "
             "Recall is on the rule's own protease type; false-hit rates are on sequences "
             "that are NOT that type.", "",
             "| Rule | Engine | Recall | Fires on non-peptidases | Fires on other protease types | Precision |",
             "|---|---|---|---|---|---|"]
    for eng, e in o["engines"].items():
        for rule, v in e["per_rule"].items():
            p = "n/a" if v["precision_vs_all_sequences"] is None else f"{v['precision_vs_all_sequences']:.0%}"
            lines.append(f"| {rule} | {eng} | {v['recall']:.0%} | {v['false_hit_on_non_peptidases']:.0%} | "
                         f"{v['false_hit_on_other_protease_types']:.0%} | {p} |")
    z = o.get("downstream_zinc_claims")
    if z:
        truths = list(next(iter(z.values())))
        lines += ["", "Downstream: share of sequences told they are a **zinc metalloprotease** with a **Zn2+** ligand "
                  "(function / ligand), by true class. Only the real metalloproteases should score high.", "",
                  "| True class | n | " + " | ".join(z) + " |", "|---|---|" + "---|" * len(z)]
        for t in truths:
            cells = " | ".join(f"{v[t]['zinc_function']:.0%} / {v[t]['zn_ligand']:.0%}" for v in z.values())
            lines.append(f"| {t} | {next(iter(z.values()))[t]['n']} | {cells} |")
    lines += ["", "| Engine | Non-peptidases with any catalytic hit | Mean rules fired per non-peptidase |",
              "|---|---|---|"]
    for eng, e in o["engines"].items():
        lines.append(f"| {eng} | {e['pct_non_peptidases_with_any_hit']:.0f}% | "
                     f"{e['mean_rules_fired_per_non_peptidase']:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["fetch", "evaluate"])
    parser.add_argument("--holdout", action="store_true",
                        help="the second, disjoint draw used to check the motifs out of sample")
    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.holdout)
    else:
        print(to_markdown(evaluate(HOLDOUT if args.holdout else DATA)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
