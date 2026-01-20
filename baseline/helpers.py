#!/usr/bin/env python3
import argparse, json, gc, pickle, os, itertools
from pathlib import Path
from typing import Any, Dict, Optional, Union, Iterator
import numpy as np
from sklearn.metrics import roc_auc_score as _sk_auc

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1] if HERE.name.endswith(".py") else Path.cwd()
RES_PATH = PROJECT_ROOT / "results"


def _count_examples(hs_dump: Dict[str, Any]) -> int:
    return len(hs_dump)

def _make_run_id(
    perturb: str = "",
    fewshot: bool = False,
    incremental: bool = False,
    positive: bool = False,
) -> str:
    """
    Construct a consistent run identifier used for:
      - results.pkl (if present)
      - *_results.txt metrics
      - <run_id>_all/all/*.pkl shards

    Format examples:
      - Default: "-1"
      - Incremental: "-1_incremental"
      - With perturb "foo": "foo_-1"
      - With perturb "foo" + incremental: "foo_-1_incremental"
    """
    rev_ind = -1
    perturb_prefix = "" if perturb == "all" or perturb == "" else f"{perturb}_"
    run_id = f"{perturb_prefix}{rev_ind}"
    if fewshot:
        run_id += "_fewshot"
    if incremental:
        run_id += "_incremental"
    return run_id

def _dump_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)

def shard_all(
    model: str,
    flat_chunk_size: int = 5000,
    data: str = "",
    data_file=None,
    perturb: str = "",
    fewshot: bool = False,
    incremental: bool = False,
):
    """
    Shard a monolithic dict of results into
    results/inference/<data>/<model>/<run_id>_all/all/*.pkl, where
    <run_id> encodes perturb, fewshot, etc.
    """
    data = data.lower()
    base_dir = RES_PATH / "inference" / data / model

    run_id = _make_run_id(
        perturb=perturb,
        fewshot=fewshot,
        incremental=incremental,
    )

    hs_dump = data_file
    total_examples = _count_examples(hs_dump)

    root = base_dir / f"{run_id}_all"
    all_dir = root / "all"
    all_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    while hs_dump:
        chunk_keys = list(next(iter(hs_dump)) for _ in range(min(flat_chunk_size, len(hs_dump))))
        chunk_keys = list(hs_dump.keys())[:flat_chunk_size]

        shard: Dict[Any, Any] = {}
        for k in chunk_keys:
            shard[k] = hs_dump.pop(k)

        _dump_pickle(shard, all_dir / f"shard_{shard_idx:06d}.pkl")
        shard_idx += 1
        del shard  # release promptly

    gc.collect()

    print("Done. Wrote 'all/' shards")
    print("  ", root)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Shard a monolithic hidden-state pickle into a single all/ set of shards, "
            "or merge multiple existing all/ shard sets into a mixed dataset (and compute metrics)."
        )
    )
    # Sharding inputs
    ap.add_argument(
        "--data",
        help="Source dataset name for sharding (e.g., book). Required for sharding mode.",
    )
    ap.add_argument("--model", required=True, help="Model name (e.g., babygpt).")
    ap.add_argument(
        "--flat_chunk_size",
        type=int,
        default=5000,
        help="Shard size for flat outputs.",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.data:
        raise SystemExit("--data is required for sharding mode.")
    shard_all(
        data=args.data,
        model=args.model,
        flat_chunk_size=args.flat_chunk_size,
        data_file=None,
    )
