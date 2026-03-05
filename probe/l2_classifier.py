#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from probe.helpers import (
    RES_PATH,
    _choose_layer_indices,
    _evaluate_classifier,
    _select_best_l2_model,
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
    base_data: Optional[str] = None,
    eval_datasets: Optional[List[str]] = None,
) -> None:
    out_dir = _make_out_dir(data=data, model=model, add_prob=add_prob)

    # Determine layer indices: use base_data's best layer when doing transfer
    if base_data is not None:
        best_model = _select_best_l2_model(
            base_data, model, start_exp, end_exp, add_prob=add_prob,
        )
        if best_model is None:
            raise RuntimeError(
                f"No trained L2 model found for base_data={base_data}, model={model}"
            )
        layer_indices: Sequence[int] = [best_model["layer_idx"]]
    else:
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
        if base_data is not None:
            base_name += f"_trained_on_{data}"
        out_path = out_dir / f"{base_name}.pkl"

        with open(out_path, "wb") as fh:
            pickle.dump(stats, fh, protocol=4)

        print(stats["overall_stats"])
        print(f"Saved layer {layer_idx} results to {out_path}")

        # Transfer evaluation: evaluate on each eval dataset
        if eval_datasets:
            best_auc = stats["overall_stats"]["best_auc"]
            weights = best_auc["weights"]
            intercept = best_auc["intercept"]

            for eval_ds in eval_datasets:
                eval_dataset = format_data(
                    data=eval_ds,
                    model_name=model,
                    split=False,
                    layer_idx=layer_idx,
                    add_prob=add_prob,
                )

                metrics, probabilities, logits = _evaluate_classifier(
                    features=eval_dataset["X_test"],
                    labels=eval_dataset["y_test"],
                    metadata=eval_dataset["meta_test"],
                    weights=weights,
                    intercept=intercept,
                )

                eval_stats = {"overall_stats": metrics}
                eval_name = (
                    f"-1_{layer_suffix}_exp{start_exp}_to_{end_exp}"
                    f"_trained_on_{data}_tested_on_{eval_ds}"
                )
                eval_out_dir = _make_out_dir(
                    data=eval_ds, model=model, add_prob=add_prob,
                )
                eval_out_path = eval_out_dir / f"{eval_name}.pkl"

                with open(eval_out_path, "wb") as fh:
                    pickle.dump(eval_stats, fh, protocol=4)

                print(f"Transfer eval on {eval_ds}: {metrics}")
                print(f"Saved transfer results to {eval_out_path}")


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
    parser.add_argument(
        "--base_data",
        type=str,
        default=None,
        help="Base dataset whose best layer is used for transfer (e.g., 'synthetic').",
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to evaluate the trained probe on (transfer evaluation).",
    )
    parser.add_argument(
        "--plausibility",
        action="store_true",
        help="Shortcut: data=plausibility, base_data=synthetic, eval_datasets=[blimp, cola, syngym].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = args.data
    base_data = args.base_data
    eval_datasets = args.eval_datasets

    if args.plausibility:
        data = "plausibility"
        base_data = "synthetic"
        eval_datasets = ["blimp", "cola", "syngym"]

    run_pipeline(
        data=data,
        model=args.model,
        start_exp=args.start_exp,
        end_exp=args.end_exp,
        add_prob=args.add_prob,
        base_data=base_data,
        eval_datasets=eval_datasets,
    )


if __name__ == "__main__":
    main()
