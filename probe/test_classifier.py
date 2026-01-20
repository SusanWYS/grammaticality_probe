#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np

from probe.helpers import (
    evaluate_probe_selection,
    _probe_dir,
    _assert_prob_run_exists,
    _load_example_lookup,
    _load_l1_model,
    _load_random_l1_models,
    _select_best_l2_model,
)

DataSplit = Tuple[np.ndarray, np.ndarray, list[Tuple[int, bool]]]
DataCache = Dict[Optional[int], DataSplit]

def run_pipeline(
    *,
    model: str,
    eval_data: str,
    start_exp: int,
    end_exp: int,
    target_ratio: Optional[float] = None,
    train_data: Optional[str] = None,
    add_prob: bool = False,
    random: bool = False,
) -> Dict[str, Any]:
    train_data_name = train_data or eval_data
    if add_prob:
        _assert_prob_run_exists("l2", train_data_name, model)
    example_lookup = _load_example_lookup(eval_data, model)
    data_cache: DataCache = {}

    if target_ratio is None:
        selections = [
            _select_best_l2_model(
                train_data_name,
                model,
                start_exp,
                end_exp,
                add_prob=add_prob,
            )
        ]
        print(f"add_prob is {add_prob}")
        out_base_dir = _probe_dir("l2", add_prob) / eval_data / model
        print(out_base_dir)
        suffixes = ["l2"]
    elif random:
        selections = _load_random_l1_models(
            train_data_name,
            model,
            target_ratio,
            start_exp,
            end_exp,
            add_prob=add_prob,
        )
        out_base_dir = _probe_dir("l1", add_prob, random_select=True) / eval_data / model
        suffixes = [
            f"l1_random{sel.get('random_idx')}" if sel.get("random_idx") is not None else "l1_random"
            for sel in selections
        ]
    else:
        selections = [
            _load_l1_model(
                train_data_name, model, target_ratio, start_exp, end_exp, add_prob=add_prob
            )
        ]
        out_base_dir = _probe_dir("l1", add_prob) / eval_data / model
        suffixes = ["l1"]

    outputs: Dict[str, Any] = {}
    for sel, suffix in zip(selections, suffixes):
        outputs[suffix] = evaluate_probe_selection(
            selection=sel,
            model=model,
            test=eval_data,
            start_exp=start_exp,
            end_exp=end_exp,
            target_ratio=target_ratio,
            add_prob=add_prob,
            suffix=suffix,
            out_dir=out_base_dir,
            data_cache=data_cache,
            example_lookup=example_lookup,
        )
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test probes on held-out data")
    parser.add_argument("--model", type=str, default="llama3-2-1B", help="Model name")
    parser.add_argument("--eval_data", type=str, required=True, help="Evaluation dataset name")
    parser.add_argument("--start_exp", type=int, default=-2, help="Start exponent for exp sweep")
    parser.add_argument("--end_exp", type=int, default=5, help="End exponent for exp sweep")
    parser.add_argument("--target_ratio", type=float, default=None, help="Ratio of neurons (%)")
    parser.add_argument("--train_data", type=str, default="synthetic", help="Training data used for probe selection")
    parser.add_argument(
        "--add_prob",
        action="store_true",
        help="Append length-normalized probability feature to hidden states",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Evaluate all random neuron selections saved by l1_classifier",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        model=args.model,
        eval_data=args.eval_data,
        start_exp=args.start_exp,
        end_exp=args.end_exp,
        target_ratio=args.target_ratio,
        train_data=args.train_data,
        add_prob=args.add_prob,
        random=args.random,
    )


if __name__ == "__main__":
    main()
