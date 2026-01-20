#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from probe.helpers import (
    RES_PATH,
    evaluate_surprisal_probe,
    train_surprisal_probe,
    _get_gpu_device_ids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(
    *,
    data: str,
    model: str,
    train_ratio: float,
    start_exp: int,
    end_exp: int,
    incremental: bool,
    eval_data: Optional[str] = None,
) -> Dict[str, Any]:
    train_data = data.lower()
    eval_data_name = eval_data.lower() if eval_data is not None else None
    device_ids = _get_gpu_device_ids()

    suffix = "__incremental" if incremental else ""

    if eval_data_name is not None:
        trained_probe_dir = RES_PATH / "probe" / "surprisal_probe" / train_data / model
        probe_path = trained_probe_dir / f"-1{suffix}.pkl"
        results = evaluate_surprisal_probe(
            probe_path=probe_path,
            eval_data=eval_data_name,
            model_name=model,
            incremental=incremental,
        )
        out_filename = f"-1{suffix}_eval_{eval_data_name}.pkl"
        out_dir = (
            RES_PATH
            / "probe"
            / "surprisal_probe"
            / "cross_eval"
            / f"{train_data}_to_{eval_data_name}"
            / model
        )
        out_path = out_dir / out_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"results": results}
        with open(out_path, "wb") as fh:
            pickle.dump(blob, fh)
        print(results)
        return results
    else:
        results, chosen_layer = train_surprisal_probe(
            data=train_data,
            model_name=model,
            train_ratio=train_ratio,
            start_exp=start_exp,
            end_exp=end_exp,
            device_ids=device_ids,
            incremental=incremental,
        )
        out_dir = RES_PATH / "probe" / "surprisal_probe" / train_data / model
        out_path = out_dir / f"-1{suffix}.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"results": results, "layer_idx": int(chosen_layer)}
        print(results)
        with open(out_path, "wb") as fh:
            pickle.dump(blob, fh)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a surprisal probe (SnapML Ridge, exp sweep)."
    )
    parser.add_argument("--data", type=str, default="synthetic", help="Dataset name (train dataset).")
    parser.add_argument("--model", type=str, default="olmo2-7B", help="Model name.")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data to use for training in the outer train/dev split.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Use incremental inference shards."
        ),
    )
    parser.add_argument(
        "--start_exp",
        type=int,
        default=-2,
        help="exp sweep start exponent k (exp = 2^k).",
    )
    parser.add_argument(
        "--end_exp",
        type=int,
        default=5,
        help="exp sweep end exponent k (exp = 2^k). Inclusive.",
    )

    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help=(
            "If set (e.g., blimp/CoLA/SyntaxGym/plausibility), run in EVAL mode:"
            "load the probe trained on --data from the default path and evaluate on --eval_data. "
            "If --incremental is also set, uses the incremental-positive probe and incremental shards."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data=args.data,
        model=args.model,
        train_ratio=args.train_ratio,
        start_exp=args.start_exp,
        end_exp=args.end_exp,
        incremental=args.incremental,
        eval_data=args.eval_data,
    )

if __name__ == "__main__":
    main()
