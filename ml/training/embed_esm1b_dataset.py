"""
Mean ESM-1b embeddings for every sequence in ml/training/enzyme_sequences.csv.

ESM-1b (Rives et al. 2021, MIT licence, 650M parameters) is a protein language
model; its layer-33 representations averaged over residues give one 1280-d
vector per sequence, which a small classifier can use far better than
hand-built composition features, using this toolkit's own training data and
classifier.

Resumable: work is written in shards (ml/training/esm1b_shards/) and finished
sequences are skipped on restart. Sequences are length-sorted and batched by a
token budget; math runs in bfloat16 autocast. Sequences are truncated to the
model's 1022-residue limit.

Needs `torch` and `fair-esm` (optional dependencies) and the weights file
esm1b_t33_650M_UR50S.pt (about 7.8 GB, plus the tiny
esm1b_t33_650M_UR50S-contact-regression.pt next to it) from
https://dl.fbaipublicfiles.com/fair-esm/models/ . Checkpoint loading keeps
torch's safe weights_only default and allows only argparse.Namespace, where
ESM stores its hyper-parameters.

Usage:
    python -m ml.training.embed_esm1b_dataset --weights /path/to/esm1b_t33_650M_UR50S.pt
    python -m ml.training.embed_esm1b_dataset --weights ... --limit 800 --shards /tmp/probe   # speed probe
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAXLEN = 1022
DEFAULT_SHARDS = ROOT / "ml" / "training" / "esm1b_shards"


def load_model(weights: Path):
    import torch
    import esm
    torch.serialization.add_safe_globals([argparse.Namespace])
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.eval().to(device), alphabet, device


def embed_batch(model, alphabet, device, batch):
    """batch: list of (id, sequence <= MAXLEN). Returns an (n, 1280) float32 array."""
    import torch
    convert = alphabet.get_batch_converter()
    _, _, toks = convert(batch)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                         enabled=(device == "cuda")):
        rep = model(toks.to(device), repr_layers=[33], return_contacts=False)["representations"][33]
    return np.stack([rep[i, 1:len(seq) + 1].float().mean(0).cpu().numpy()
                     for i, (_, seq) in enumerate(batch)])


def make_batches(items, max_tokens):
    """items: list of (id, seq), already length-sorted. Yields lists within a token budget."""
    batch, longest = [], 0
    for item in items:
        longest = max(longest, len(item[1]) + 2)
        if batch and longest * (len(batch) + 1) > max_tokens:
            yield batch
            batch, longest = [], len(item[1]) + 2
        batch.append(item)
    if batch:
        yield batch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sequences", default=str(ROOT / "ml" / "training" / "enzyme_sequences.csv"))
    parser.add_argument("--shards", default=str(DEFAULT_SHARDS))
    # Measured on an 8 GB RTX 5050: ~17.7 seq/s at 768-1024 tokens, ~3.8 at 4096
    # (larger batches are slower here, not faster).
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None,
                        help="random subset, for a speed probe")
    args = parser.parse_args(argv)

    shards = Path(args.shards)
    shards.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sequences)
    if args.limit:
        df = df.sample(args.limit, random_state=0)
    done = set()
    for p in shards.glob("part_*.npz"):
        done.update(np.load(p, allow_pickle=False)["ids"].tolist())
    items = [(i, s[:MAXLEN]) for i, s in zip(df["ID"], df["Sequence"]) if i not in done]
    items.sort(key=lambda t: len(t[1]))
    print(f"{len(done)} already embedded, {len(items)} to do", flush=True)
    if not items:
        return 0

    model, alphabet, device = load_model(Path(args.weights))
    print(f"ESM-1b on {device}", flush=True)
    shard_no = len(list(shards.glob("part_*.npz")))
    ids, embs, t0, n = [], [], time.time(), 0
    for batch in make_batches(items, args.max_tokens):
        embs.append(embed_batch(model, alphabet, device, batch))
        ids += [b[0] for b in batch]
        n += len(batch)
        if len(ids) >= 2000:
            np.savez(shards / f"part_{shard_no:04d}.npz", ids=np.array(ids),
                     emb=np.concatenate(embs).astype(np.float16))
            shard_no += 1
            ids, embs = [], []
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(items)}  {rate:.1f} seq/s  eta {(len(items) - n) / rate / 60:.0f} min",
                  flush=True)
    if ids:
        np.savez(shards / f"part_{shard_no:04d}.npz", ids=np.array(ids),
                 emb=np.concatenate(embs).astype(np.float16))
    bad = sum(not np.isfinite(np.load(p)["emb"]).all() for p in shards.glob("part_*.npz"))
    print(f"done in {(time.time() - t0) / 60:.1f} min; shards with non-finite values: {bad}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
