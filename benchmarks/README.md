# Benchmarks

Everything here regenerates the numbers quoted in the top-level README and
`paper/paper.md`. Results are written to `benchmarks/results/` (committed,
so you can diff a re-run against them). Docking needs the Vina binary at
`bin/vina.exe` (see `bin/README.md`); network access is needed on first run
to fetch PDB entries (RCSB) and, for `--build`, ChEMBL.

## 1. Redocking — does the docking stack reproduce crystal poses?

```bash
python -m benchmarks.redocking --seeds 42 1 2
```

Four protein-ligand complexes (3HS4, 3PTB, 1STP, 181L), each ligand
re-docked into its own protein. Success = top-ranked pose within 2.0 Å
heavy-atom RMSD of the crystal pose (symmetry-aware, no re-alignment).
Three modes:

- *native box*: centered on the crystal ligand. Tests the engine and
  receptor preparation.
- *pipeline*: the toolkit's own pocket detection. **Not blind:** the
  detector builds pockets from the structure's HETATM records, which include
  the ligand being redocked, so a small pocket-to-ligand distance here is by
  construction.
- *blind*: the same, with every HETATM removed first, so cavity detection
  must find the site from protein atoms alone. This is the real test of
  site finding.

Output: `results/redocking.{json,md}`.

Four complexes is a smoke test of correctness, not a docking benchmark. A
publishable comparison needs a standard set (PDBbind core, CASF-2016, Astex
Diverse) and a baseline engine run through the same harness.

## 2. Screening enrichment — can it rank actives above weak binders?

```bash
python -m benchmarks.screening --target ca2 --build   # once per target: samples ChEMBL
python -m benchmarks.screening --target ca2           # docks the committed set
python -m benchmarks.screening --target esr1
```

Two targets: carbonic anhydrase II (1CA2, a zinc metalloenzyme) and
estrogen receptor α (3ERT, no metal). For each, 50 ChEMBL actives (binding
assays, pChEMBL ≥ 7) against 50 size-matched measured weak binders
(pChEMBL ≤ 5). Reports ROC AUC with a bootstrap interval, a
heavy-atom-count-only control AUC, top-10% enrichment factor, and the
toolkit's pose labels cross-tabulated against truth; for CA II also a
primary-sulfonamide control (its zinc-binding group). The docking site is
the pocket built from the structure's HETATM records (the Zn ion in 1CA2,
the bound ligand in 3ERT): known-site docking, not blind site finding.
`data/*_screen_set.tsv` are committed; `--build` is only needed to resample.
Per-ligand scores are in `results/screening_<target>_ligands.csv`.

Not DUD-E/LIT-PCBA: those provide property-matched *decoys* at scale. This
set uses measured weak binders instead (harder, same chemotypes), but it is
two targets and ~100 ligands each, so AUC intervals are wide.

## 3. EC classifier — accuracy without sequence leakage

```bash
python -m ml.training.cluster_sequences            # writes ml/training/enzyme_clusters.csv
python -m ml.training.train_xgb_enzyme_classifier  # cluster-aware split
python -m ml.evaluation                            # accuracy, Brier, ECE, reliability plot
```

Sequences are grouped by 5-mer containment (≥ 5% of a sequence's 5-mers
shared with a cluster representative ≈ 55% identity over an end-to-end
alignment) and whole clusters go to either train or test. This is a coarse
proxy for CD-HIT/MMseqs2 clustering and will not catch remote homologs at
30–40% identity; use those tools for a stricter split.

`enzyme_clusters.csv` is gitignored (regenerable, 1.8 MB). Without it the
trainers fall back to a random split and warn — that number is inflated.
