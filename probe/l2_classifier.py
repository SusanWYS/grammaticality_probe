#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from probe.helpers import (
    RES_PATH,
    _choose_layer_indices,
    format_data,
    train_logistic_l2,
    _make_out_dir,
)


def _train_layer_probe(
    *,
    layer_idx: Optional[int],
    model_name: str,
    data: str,
    start_exp_exp: int,
    end_exp_exp: int,
    add_prob: bool,
) -> Dict[str, Any]:
    # Training/validation split from in-domain data (80/20)
    dataset = format_data(
        data=data,
        model_name=model_name,
        split=True,
        train_ratio=0.8,
        layer_idx=layer_idx,
        add_prob=add_prob,
    )

    best_auc_score = -1.0
    best_acc_score = -1.0

    for exp in range(start_exp_exp, end_exp_exp + 1):
        exp_val = float(2**exp)
        res = train_logistic_l2(
            exp_val,
            model_name=model_name,
            dataset=dataset,
        )

        if res["test_auc"] > best_auc_score:
            best_auc_score = res["test_auc"]
            best_by_auc = {
                        "exp": exp_val,
                        "weights": res["weights"],
                        "intercept": res["intercept"],
                        "val_auc": res["test_auc"],
                        "val_acc": res["test_acc"],
                        "train_auc": res["train_auc"],
                        "train_acc": res["train_acc"],
                        }

    return {"overall_stats": {"best_auc": best_by_auc}}


def run_pipeline(
    *,
    data: str,
    model: str,
    start_exp: int,
    end_exp: int,
    add_prob: bool,
) -> None:
    out_dir = _make_out_dir(data=data, model=model, add_prob=add_prob)
    layer_indices = _choose_layer_indices(
        data=data,
        model=model,
        start_exp=start_exp,
        end_exp=end_exp,
        add_prob=add_prob,
    )

    for layer_idx in layer_indices:
        stats = _train_layer_probe(
            layer_idx=layer_idx,
            model_name=model,
            data=data,
            start_exp_exp=start_exp,
            end_exp_exp=end_exp,
            add_prob=add_prob,
        )

        layer_suffix = f"layer{layer_idx}"
        base_name = f"-1_{layer_suffix}_exp{start_exp}_to_{end_exp}"
        out_path = out_dir / f"{base_name}.pkl"

        with open(out_path, "wb") as fh:
            pickle.dump(stats, fh, protocol=4)

        print(stats["overall_stats"])
        print(f"Saved layer {layer_idx} results to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise L2 sweep for probes (train/val only).")
    parser.add_argument("--model", type=str, default="llama3-2-1B", help="Model name used for shards.")
    parser.add_argument("--data", type=str, default="synthetic", help="Dataset name for training/validation.")
    parser.add_argument("--start_exp", type=int, default=-2, help="Start exponent for 2^exp sweep.")
    parser.add_argument("--end_exp", type=int, default=5, help="End exponent for 2^exp sweep.")
    parser.add_argument(
        "--add_prob",
        action="store_true",
        help="Append length-normalized probability feature to hidden state vectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data=args.data,
        model=args.model,
        start_exp=args.start_exp,
        end_exp=args.end_exp,
        add_prob=args.add_prob,
    )


if __name__ == "__main__":
    main()
