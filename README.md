# ProteinToolkit

An integrated protein analysis pipeline: sequence statistics, motif and
catalytic-residue detection, rule-based + machine-learning EC classification,
structural pocket/active-site analysis, and ligand docking (AutoDock Vina),
fused into one confidence-scored report.

## Why

Most tools in this space specialize: a sequence stats calculator, a motif
scanner, a pocket detector, a docking engine. ProteinToolkit runs the
sequence-level triage (stats, motifs, catalytic residues, EC classification)
with **no structure required**, then optionally confirms candidates
structurally (pockets, spatial catalytic-site geometry, docking) — so you can
screen many sequences cheaply and only run the expensive structural stage on
the ones that pass triage.

The closest published prior art is CAPIM (Özsari et al. 2025, *Protein
Science* 34(11):e70347, [doi.org/10.1002/pro.70347](https://doi.org/10.1002/pro.70347)),
which combines pocket detection (P2Rank) + catalytic/EC annotation (GASS) +
docking (AutoDock Vina) into one report. CAPIM's structural methods are more
rigorous than this toolkit's; this toolkit's addition is the sequence-only
triage front end and the hedged-confidence reporting discipline (see
Limitations below).

## Install

```bash
pip install -r requirements.txt
```

Optional, for real docking (not required — falls back to a mock adapter
otherwise): download a precompiled AutoDock Vina binary from
[github.com/ccsb-scripps/AutoDock-Vina/releases](https://github.com/ccsb-scripps/AutoDock-Vina/releases)
and place it at `bin/vina.exe` (see [bin/README.md](bin/README.md)).

## Usage

**Sequence-only analysis** (stats, motifs, catalytic residues, EC classification):

```bash
echo "YOUR_SEQUENCE" | python main.py
```

**With structure** (adds pocket detection, spatial catalytic-site check,
structural active-site prediction) — accepts a local PDB file or a PDB ID,
which is fetched from RCSB automatically:

```bash
echo "YOUR_SEQUENCE" | python main.py 1LYZ
```

**Full pipeline with docking** (receptor + ligand + real Vina docking +
evidence fusion):

```bash
python pipeline.py --pdb 1LYZ --smiles "CC(=O)Oc1ccccc1C(=O)O" --engine vina
```

**Virtual screening** (which of *many* candidate ligands best fits this
receptor? — docks a whole library and ranks by score; receptor prep/pocket
detection run once, not once per candidate):

```bash
python screen.py --pdb 1LYZ --library candidates.smi --engine vina --top 10
```

`candidates.smi` is a text file, one candidate per line: `SMILES [name]`.

**Batch triage** (screen many sequences from a FASTA file, ranked by
combined confidence, no structure needed):

```bash
python triage.py sequences.fasta --out triage.csv --top 20
```

**Model calibration check** (is the EC classifier's confidence trustworthy?):

```bash
python -m ml.evaluation --model ml/models/enzyme_classifier_xgb.pkl
```

Retraining the EC classifier from fresh UniProt data:

```bash
python -m ml.training.fetch_enzyme_dataset --per-class 3000
python -m ml.training.build_feature_dataset
python -m ml.training.train_xgb_enzyme_classifier --keep-old
```

## Architecture

```
core/        sequence stats, motif/catalytic/enzyme/function/ligand rules,
             evidence scoring, reporting, export
structure/   PDB parsing + fetch, contacts, pocket scoring, spatial
             catalytic-site geometry, structural evidence fusion
docking/     receptor/ligand prep, pocket detection, AutoDock Vina/GNINA
             adapters (falls back to a mock adapter if neither is installed)
ligand/      SMILES/SDF/MOL2/PDB parsing and chemical validation (RDKit if
             available, a dependency-free fallback parser otherwise)
ml/          EC classifier training (RandomForest + XGBoost), calibration
             evaluation
pipeline.py  full sequence + structure + docking + evidence-fusion CLI
main.py      sequence (+ optional structure) analysis CLI
triage.py    batch sequence-only screening CLI
screen.py    virtual screening: rank a ligand library against one receptor
```

## Limitations

Read this before citing any number from this tool in a paper.

- **Catalytic-residue detection is two-tiered.** `core/catalytic.py` is
  sequence-composition-only (no 3D structure needed, but prone to false
  positives on proteins that merely contain the right residue *types*
  without a real catalytic arrangement). When a structure is available,
  `structure/catalytic_geometry.py` additionally checks whether specific
  residue *instances* are actually close together in 3D space — meaningfully
  stronger evidence, but still a centroid-distance proxy, not a validated
  geometric/orientation model like GASS.
- **Docking uses placeholder ligand/receptor PDBQT charges.** Real partial
  charges and rotatable-bond trees are not computed (see the docstrings in
  `docking/engines/vina_adapter.py` and `docking/preparation.py`). Vina
  scores from this pipeline are real engine output on real structures, but
  are not chemically calibrated — don't quote them as binding affinities.
- **The EC classifier's accuracy number depends on how you split the data.**
  Training data is deduplicated by exact sequence before splitting (near-
  duplicate, non-exact matches are not currently filtered — full identity-
  aware clustering, e.g. CD-HIT-style, would be needed to rule out all
  leakage). Report the number from a fresh `python -m ml.evaluation` run,
  not from memory — and treat it as a self-reported holdout result, not a
  benchmark against CLEAN (Yu et al. 2023, *Science* 379(6639):1358-1363,
  [doi.org/10.1126/science.adf2465](https://doi.org/10.1126/science.adf2465))
  or any other published EC-classification baseline, unless you've actually
  run that comparison.
- **The GNN scaffold (`structure/gnn.py`) is explicitly unimplemented** —
  it builds a real residue-contact graph but the model itself always
  reports unavailable.
- **The rule-based confidence scores are hand-tuned weights, not calibrated
  probabilities.** Only the ML classifier's `predict_proba` output has been
  checked for calibration (see `ml/evaluation.py`); the motif/catalytic/
  enzyme/function/ligand confidence fields elsewhere in the pipeline are
  heuristic, not statistical.
- **GNINA has no supported path on Windows** (no official Windows binary);
  the adapter exists but falls back to the mock engine there.

## License

Not yet chosen — `licence` is currently empty. Pick one (MIT/BSD/Apache-2.0
are common for research software) before publishing.
