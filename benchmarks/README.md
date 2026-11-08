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

Receptors are protonated at pH 7.4 (`pdb2pqr`/PROPKA) and typed/charged by
meeko before docking (`docking.preparation.write_prepared_receptor_pdbqt`);
the route actually used (`pdb2pqr+meeko`, or the `legacy` no-hydrogen
fallback) is stored per complex in the JSON and printed in the markdown, so
a run that silently fell back is visible. The first published numbers used
the legacy receptor; both are reported in the top-level README.

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

## 4. Triage gate coverage — does the recall guarantee hold on unseen clusters?

```bash
python -m ml.evaluation --gate-calibration outputs/gate_calib.csv   # needs the cluster file (section 3)
python -m benchmarks.gate_coverage
```

Splits the EC classifier's held-out rows by cluster into a calibration part
(20% or 50% of clusters) and a deployment part, fits the triage gate on the
former, and checks its realized recall on the latter over 200 random splits.
Compares the row-level gate (rows treated as independent) with the
cluster-aware gate (one random correct prediction per cluster) and reports
both cluster-averaged and row-weighted recall. A valid 95% gate should hold in
at least 95% of splits for the recall it bounds (cluster-averaged, for the
cluster-aware gate). Output: `results/gate_coverage.{json,md}`.

The deployment part is a finite sample of clusters, so its recall scatters
around the population value; the measured coverage includes that noise.

## 5. Novelty-aware confidence — does similarity to the training data predict accuracy?

```bash
python -m ml.novelty --build     # needs the trained model, features and cluster file (section 3)
```

For each held-out sequence (none shares a cluster with training) it measures
the 5-mer overlap with the nearest training family, then fits a 3-parameter
logistic on (model confidence, log similarity) to predict whether the
prediction is right. Quality is reported out-of-fold, grouped by sequence
cluster, in `results/novelty_calibration.{json,md}`: calibration error, Brier
score and AUC with and without the adjustment, and accuracy per novelty tier.

What it does not show: the held-out set only spans the similarity range it
spans (extrapolating to similarity ~0 or ~1 is unvalidated), similarity is a
k-mer proxy that misses remote homology, and the fit describes this one model
and must be rebuilt whenever the classifier is retrained.

## 6. ESM-1b embedding classifier — accuracy and calibration

```bash
python -m ml.training.embed_esm1b_dataset --weights /path/to/esm1b_t33_650M_UR50S.pt   # resumable, ~85 min on an 8 GB GPU
python -m ml.training.train_plm_classifier --weights /path/to/esm1b_t33_650M_UR50S.pt
python -m ml.novelty --build --model ml/models/enzyme_classifier_plm.pkl
python -m ml.evaluation --model ml/models/enzyme_classifier_plm.pkl
```

Mean ESM-1b (layer 33) embeddings of every sequence feed a classifier trained on
the same cluster-aware split as the XGBoost model, so the test sets are
identical. The classifier (logistic regression vs a one-hidden-layer MLP, with
its early-stopping epoch) is chosen on a cluster-grouped validation split
carved from the training data, one temperature is fitted there so that
confidence matches accuracy, and the test set is used once. Output:
`results/plm_classifier.{json,md}` and
`results/novelty_calibration_enzyme_classifier_plm.{json,md}`.

What it does not show: the language model was pretrained (without labels) on
UniRef50, which contains relatives of nearly every test sequence, and it finds
remote homologs the 5-mer clustering cannot, so accuracy on the held-out set
overstates performance on sequences with no relatives (it is ~78% in the least
similar fifth). Training is single-seed; the two runs differ by about 2 points.
ESM-1b is MIT-licensed (Meta).

## 7. Pose plausibility — does the pose label predict whether the pose is right?

```bash
python -m benchmarks.pose_validation select      # sample complexes from RCSB by rule (network)
python -m benchmarks.pose_validation dock        # dock each ligand into its own site; resumable
python -m benchmarks.pose_validation_analysis    # AUCs, bootstrap intervals, calibration
python -m benchmarks.pose_validation_analysis fit    # write database/pose_plausibility.json
```

Complexes are chosen by rule (X-ray, <= 2.0 A, one protein chain, one bound
small molecule that is not a common additive or ion, 120-350 Da, 8-28 heavy
atoms, with a ring; fixed random seed; sampled from the first 10,000 matches),
not by hand. Each ligand is docked back into its own site and every pose gets
its true RMSD to the crystal pose (correct = within 2 A). Output:
`results/pose_validation.{json,md}`, per-pose rows in
`results/pose_validation_poses.csv`. The decision rule for adopting the fitted
model (leave-one-complex-out AUC more than 0.05 above the rule-based label's)
was fixed before the results were seen.

What it does not show: 30 complexes is small (see the bootstrap intervals);
the 8 of 38 selected entries that failed to dock (missing files, ligand not on
the protein chain) are dropped, which may favour easy cases; and a correct
pose is not a binding prediction.

## 8. Catalytic-system rules — when it says "serine protease", is it one?

```bash
python -m benchmarks.catalytic_validation fetch               # primary set, from UniProt
python -m benchmarks.catalytic_validation fetch --holdout     # a second, disjoint draw
python -m benchmarks.catalytic_validation evaluate            # writes results/catalytic_validation.*
python -m benchmarks.catalytic_validation evaluate --holdout  # the out-of-sample numbers
```

UniProt EC subclasses give ground truth: 3.4.21 serine, 3.4.22 cysteine,
3.4.23 aspartic, 3.4.24 metallo-proteases, against non-peptidase hydrolases,
oxidoreductases, transferases and lyases/isomerases/ligases as negatives. The
old composition-only rule and the current motif engine are both scored on the
same sequences, along with how often the downstream predictors then claim a zinc
metalloprotease with a Zn2+ ligand. The motifs were screened on the primary
set, so its numbers are optimistic; the held-out draw (accessions disjoint from
the primary set) is the figure to quote.

What it does not show: recall is partial because the motifs cover only some
protease families; the labels come from EC annotation, which is itself
imperfect; and the standalone HEXXH motif entry in `database/motifs.json` is
looser than the catalytic rules and still produces a 2-9% false zinc claim.

## 9. Active-site detection — does it find the real catalytic residues?

```bash
python -m benchmarks.active_site_validation
```

Runs the structure-based catalytic-site detector (`structure/catalytic_geometry.py`)
on 15 classic proteases with textbook catalytic residues (numbering of the
deposited file, written from the literature before running) and scores the right
system, the catalytic residues recovered and any false extra systems, for the
`strict` detector and the `legacy` geometry-only one it replaced. Seven structures
were looked at while designing `strict` (their numbers are optimistic); eight were
chosen afterwards. Output: `results/active_site_validation.{json,md}`.

What it does not show: only four protease systems are covered, so it says nothing
about other enzymes; 15 structures is small; the endothiapepsin truth was corrected
after inspecting the file; actinidin is partly development (it prompted the
modified-residue handling); and a plausible catalytic cluster is not a proven
active site. Needs network access on first run to fetch the PDB files.

