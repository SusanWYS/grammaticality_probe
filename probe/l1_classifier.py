#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
from snapml import LogisticRegression

from probe.helpers import FormatDataResult, _probe_dir, format_data, train_logistic_l2, train_l1

RANDOM_RUNS = 30
NONZERO_EPS = 1e-12


def l1_sweep_select_indices(
    *,
    dataset: FormatDataResult,
    target_count: int,
    exp_start: float,
    exp_step: float,
    max_steps: int,
) -> Tuple[float, np.ndarray]:
    """
    Adjust L1 exp until the number of non-zero weights is within 5% of target_count.
    Returns (final_exp, selected_indices).
    """
    def count_nonzero_weights(model: LogisticRegression) -> Tuple[np.ndarray, int]:
        """Return (nonzero_mask, nonzero_count) for a fitted LogisticRegression."""
        coef = model.coef_.reshape(-1)
        mask = np.abs(coef) > NONZERO_EPS
        return mask, int(mask.sum())

    exp = float(exp_start)
    step = float(exp_step)

    prev_delta: Optional[int] = None

    for sweep_step in range(max_steps):
        l1_model = train_l1(dataset, exp)
        nz_mask, nz_count = count_nonzero_weights(l1_model)

        print(f"L1 exp={exp:.6g} produced {nz_count} non-zero neurons.")

        # Stop if within 5% of target.
        tol = int(target_count * 0.05)
        if abs(nz_count - target_count) <= tol:
            selected_indices = np.where(nz_mask)[0]
            return exp, selected_indices

        cur_delta = nz_count - target_count

        crossed = prev_delta is not None and (
            (prev_delta > 0 and cur_delta < 0)
            or (prev_delta < 0 and cur_delta > 0)
        )
        if crossed:
            step = step / 3.0
            if step <= 0:
                step = max(exp / 3.0, 1e-6)

        prev_delta = cur_delta
        if nz_count < target_count:
            candidate = exp - step
            exp = candidate if candidate > 0 else max(exp / 3.0, 1e-6)
        else:
            exp += step

    raise Exception(f"l1 sweep failed to find {target_count} neurons")


def l2_sweep(
    *,
    dataset: FormatDataResult,
    selected_indices: np.ndarray,
    l2_start_exp: int,
    l2_end_exp: int,
    model_name: str,
) -> Dict[str, Any]:
    """
    Sweep L2 regularization over 2^[l2_start_exp, l2_end_exp] on selected features.
    Select the best model by validation AUC (fallback to ACC if AUC is NaN).
    """
    best_score = -math.inf
    for exp in range(l2_start_exp, l2_end_exp + 1):
        l2_exp = float(2**exp)
        l2_res = train_logistic_l2(
            l2_exp,
            model_name=model_name,
            dataset=dataset,
            selected_indices=selected_indices,
        )

        score = float(l2_res["test_auc"])
        if score > best_score:
            best_score = score
            best_res = {
                    "val_auc": float(l2_res["test_auc"]),
                    "val_acc": float(l2_res["test_acc"]),
                    "weights": np.asarray(l2_res["weights"]).copy(),
                    "intercept": np.asarray(l2_res["intercept"]).copy(),
                    "non_zero_neurons": selected_indices.tolist(),
                    }

    return best_res


def run_pipeline(
    *,
    data: str,
    model: str,
    target_ratios: list[float],
    l1_exp_start: float,
    l1_exp_step: float,
    l2_start_exp: int,
    l2_end_exp: int,
    add_prob: bool = False,
    random: bool = False,
    max_l1_steps: int = 100,
) -> None:
    """Run sparse selection (L1) + retraining (L2) pipeline and save outputs."""

    def run_l1_l2(random: bool, rnd_idx: int) -> None:
        rng = np.random.default_rng(rnd_idx)
        if random:
            selected_indices = rng.choice(D, size=target_count, replace=False)
        else:
            _, selected_indices = l1_sweep_select_indices(
                dataset=dataset,
                target_count=target_count,
                exp_start=l1_exp_start,
                exp_step=l1_exp_step,
                max_steps=max_l1_steps,
            )

        best_res = l2_sweep(
            dataset=dataset,
            selected_indices=selected_indices,
            l2_start_exp=l2_start_exp,
            l2_end_exp=l2_end_exp,
            model_name=model,
        )
        summary = {
            "l2_best": best_res,
            "layer_idx": "all",
        }
        random_suffix = f"_random{rnd_idx}" if random else ""
        if random:
            summary["random_idx"] = rnd_idx
        out_path = (
            probe_dir
            / f"-1_layerall{random_suffix}_l2_{l2_start_exp}_to_"
            f"{l2_end_exp}_num{target_ratio}.pkl"
        )
        with open(out_path, "wb") as fh:
            pickle.dump(summary, fh, protocol=4)
        print(f"Saved results to {out_path}")
        return

    dataset = format_data(
        data=data,
        model_name=model,
        split=True,
        train_ratio=0.8,
        layer_idx=None,
        add_prob=add_prob,
    )
    D = int(dataset["dim"])
    probe_dir = _probe_dir("l1", add_prob, random_select=random) / data / model
    probe_dir.mkdir(parents=True, exist_ok=True)

    for target_ratio in target_ratios:
        target_count = int(math.ceil(target_ratio * D * 0.01))
        print(f"Target non-zero neurons: {target_count} (of {D} total features)")
        run_count = RANDOM_RUNS if random else 1
        for rnd_idx in range(run_count):
            run_l1_l2(random=random, rnd_idx=rnd_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse L1+L2 probe pipeline (train/val only)"
    )
    parser.add_argument("--model", type=str, default="llama3-2-1B", help="Model name")
    parser.add_argument("--data", type=str, default="synthetic", help="Dataset name")
    parser.add_argument(
        "--l1_exp_start",
        type=float,
        default=1.0,
        help="Starting exp for L1 sweep",
    )
    parser.add_argument(
        "--l1_exp_step",
        type=float,
        default=5000.0,
        help="Additive step to adjust L1 exp toward target sparsity",
    )
    parser.add_argument(
        "--l2_start_exp",
        type=int,
        default=-2,
        help="Start exponent for 2^exp L2 sweep",
    )
    parser.add_argument(
        "--l2_end_exp",
        type=int,
        default=5,
        help="End exponent for 2^exp L2 sweep",
    )
    parser.add_argument(
        "--add_prob",
        action="store_true",
        help="Append length-normalized probability as an additional feature",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random neuron selections instead of L1 selection (30 runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data=args.data,
        model=args.model,
        target_ratios=[0.01, 0.05, 0.1, 0.5],
        l1_exp_start=args.l1_exp_start,
        l1_exp_step=args.l1_exp_step,
        l2_start_exp=args.l2_start_exp,
        l2_end_exp=args.l2_end_exp,
        add_prob=args.add_prob,
        random=args.random,
    )


if __name__ == "__main__":
    main()
