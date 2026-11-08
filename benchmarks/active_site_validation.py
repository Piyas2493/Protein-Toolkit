"""
Active-site validation: does the structure-based detector find the real catalytic residues?

For classic enzymes with textbook catalytic residues, run the spatial catalytic-site
detector (structure/catalytic_geometry.py) and score three things:

  system reported   did it name the right catalytic system?
  residues correct  how many of the textbook catalytic residues did it return?
  false systems     other systems it reported that the enzyme does not have

Two detectors on the same structures: `legacy` (any residues of the required types
whose centroids are within 8 A) and `strict` (needs the conserved motif in the
sequence, uses the residues the motif marks, side-chain functional atoms, connected
cluster within 5 A, modified residues read through MODRES).

Two sets:
  development   seven structures looked at while designing `strict`; its numbers are
                optimistic.
  held-out      eight structures chosen after the design. One caveat is recorded in
                its notes: the MODRES handling was added after actinidin (2ACT)
                failed, so 2ACT is partly development; and the endothiapepsin truth
                was first written in literature numbering and corrected after the
                file showed pepsin-style numbering (the two scores are given).

Truth residues use the numbering OF THE DEPOSITED FILE, written from the literature
before running. Only the four systems in database/catalytic_sites.json are covered.

Usage:
    python -m benchmarks.active_site_validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).parent / "results"

DEVELOPMENT = [
    ("3PTB", "Serine Protease Triad", {"HIS57", "ASP102", "SER195"}, "bovine trypsin"),
    ("5CHA", "Serine Protease Triad", {"HIS57", "ASP102", "SER195"}, "bovine chymotrypsin A"),
    ("1SBT", "Serine Protease Triad", {"ASP32", "HIS64", "SER221"}, "subtilisin BPN'"),
    ("9PAP", "Cysteine Protease Dyad", {"CYS25", "HIS159"}, "papain; Cys25 is stored as OCS (modified)"),
    ("5PEP", "Aspartic Protease", {"ASP32", "ASP215"}, "porcine pepsin"),
    ("8TLN", "Metalloprotease", {"HIS142", "GLU143", "HIS146"}, "thermolysin"),
    ("1HVR", "Aspartic Protease", {"ASP25"}, "HIV-1 protease (dimer)"),
]
HELD_OUT = [
    ("3EST", "Serine Protease Triad", {"HIS57", "ASP102", "SER195"}, "porcine pancreatic elastase"),
    ("1PPB", "Serine Protease Triad", {"HIS57", "ASP102", "SER195"}, "human alpha-thrombin"),
    ("4APE", "Aspartic Protease", {"ASP32", "ASP215"},
     "endothiapepsin; literature Asp35/Asp219, but THIS file uses pepsin-style numbering (35 and 219 are Ser/Thr)"),
    ("1HSG", "Aspartic Protease", {"ASP25"}, "HIV-1 protease + indinavir (dimer)"),
    ("1CP3", "Cysteine Protease Dyad", {"CYS163", "HIS121"}, "caspase-3 + inhibitor; Cys163-His121 are 5.95 A apart here"),
    ("2ACT", "Cysteine Protease Dyad", {"CYS25", "HIS162"}, "actinidin; Cys25 stored as CSD (partly development, see docstring)"),
    ("1PPN", "Cysteine Protease Dyad", {"CYS25", "HIS159"}, "papain, unmodified Cys25"),
    ("1AST", "Metalloprotease", {"HIS92", "GLU93", "HIS96"}, "astacin"),
]


def score(detector, structure, system, truth):
    hits = detector.predict(structure)
    by = {h["name"]: {r.split(":")[1] for r in h["residues_found"]} for h in hits}
    got = by.get(system)
    return {
        "system_reported": got is not None,
        "residues_correct": len(truth & (got or set())),
        "residues_total": len(truth),
        "false_systems": sorted(n for n in by if n != system),
        "reported_residues": sorted(got or []),
    }


def run_set(entries, detectors):
    from structure.fetch import fetch_pdb
    from structure.parser import parse_pdb

    rows = []
    for pdb, system, truth, note in entries:
        structure = parse_pdb(str(fetch_pdb(pdb)))
        row = {"pdb": pdb, "system": system, "truth": sorted(truth), "note": note}
        for name, det in detectors.items():
            row[name] = score(det, structure, system, truth)
        rows.append(row)
    return rows


def summarise(rows, mode):
    return {
        "systems_reported": sum(r[mode]["system_reported"] for r in rows),
        "structures": len(rows),
        "residues_correct": sum(r[mode]["residues_correct"] for r in rows),
        "residues_total": sum(r[mode]["residues_total"] for r in rows),
        "structures_with_false_systems": sum(bool(r[mode]["false_systems"]) for r in rows),
    }


def to_markdown(out):
    lines = ["# Active-site validation (structure-based catalytic-site detector)", "",
             "Textbook catalytic residues of classic proteases, numbered as in the deposited file. "
             "`legacy` = any residues of the right types within 8 A (centroids); `strict` = needs the "
             "conserved motif, uses the residues it marks, side-chain atoms, connected cluster within 5 A.", ""]
    for set_name in ("development", "held_out"):
        lines += [f"## {'Development set (optimistic: used to design `strict`)' if set_name == 'development' else 'Held-out set'}", "",
                  "| Detector | Right system reported | Catalytic residues recovered | Structures with false extra systems |",
                  "|---|---|---|---|"]
        for mode in ("legacy", "strict"):
            s = out[set_name]["summary"][mode]
            lines.append(f"| {mode} | {s['systems_reported']}/{s['structures']} | "
                         f"{s['residues_correct']}/{s['residues_total']} | {s['structures_with_false_systems']}/{s['structures']} |")
        lines += ["", "| PDB | Enzyme | Truth | strict result | legacy result |", "|---|---|---|---|---|"]
        for r in out[set_name]["rows"]:
            def cell(m):
                x = r[m]
                got = ", ".join(x["reported_residues"]) or "not reported"
                extra = f"; false: {', '.join(x['false_systems'])}" if x["false_systems"] else ""
                return f"{x['residues_correct']}/{x['residues_total']} ({got}{extra})"
            lines.append(f"| {r['pdb']} | {r['note']} | {', '.join(r['truth'])} | {cell('strict')} | {cell('legacy')} |")
        lines.append("")
    lines += ["Limits: 15 structures, all proteases of the four covered systems; truth written from the "
              "literature; the detector marks a PLAUSIBLE catalytic cluster and says nothing about enzymes outside "
              "those four systems. 4APE's truth was corrected after inspecting the file (its score with the "
              "literature numbering would be 0/2).", ""]
    return "\n".join(lines)


def main() -> int:
    from structure.catalytic_geometry import SpatialCatalyticDetector

    detectors = {"strict": SpatialCatalyticDetector(strict=True),
                 "legacy": SpatialCatalyticDetector(strict=False)}
    out = {}
    for name, entries in (("development", DEVELOPMENT), ("held_out", HELD_OUT)):
        rows = run_set(entries, detectors)
        out[name] = {"rows": rows, "summary": {m: summarise(rows, m) for m in detectors}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "active_site_validation.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "active_site_validation.md").write_text(to_markdown(out), encoding="utf-8")
    print(to_markdown(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
