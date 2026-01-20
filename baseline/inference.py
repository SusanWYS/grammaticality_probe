import argparse
import pickle
from typing import Any, Dict, Optional, Union
from torcheval.metrics.functional import binary_auroc
import torch
import sys
import os
from itertools import islice
from baseline.helpers import shard_all, _make_run_id
from tqdm import tqdm
import numpy as np
from pathlib import Path
import random

from data.format_data import (
    BLiMP,
    SynthData,
    SyntaxGym,
    CoLA,
    Plausibility,
    Scala,
    ITACoLA,
    RuCoLA,
    JCoLA,
    SLING,
    BLiMP_nl
)
from models.all_models import hfModels

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]

def forward_pass(
    good_ex: str,
    bad_ex: str,
    model: hfModels,
    incremental: bool = False,
):
    """
    Returns a dictionary where 'hs' and 'norm_prob' are:
    - Single values if incremental=False
    - Lists of values if incremental=True
    """
    results = {
        "good_hs": None,
        "norm_good_prob": None,
        "bad_hs": None,
        "norm_bad_prob": None,
    }

    if good_ex:
        results["good_hs"], results["norm_good_prob"] = model.forward_and_return_hs(
            good_ex, incremental=incremental
        )

    if bad_ex:
        results["bad_hs"], results["norm_bad_prob"] = model.forward_and_return_hs(
            bad_ex, incremental=incremental
        )

    return results


def inference_general(
    data_name: str,
    model_name: str = "olmo2-7B",
    perturb: str = "all",
    incremental: bool = False
) -> None:
    """Run inference on general datasets and optionally merge shards for 'mixed'."""

    if data_name == "blimp":
        data = BLiMP()
    elif data_name == "syntaxgym":
        data = SyntaxGym()
    elif data_name.lower() == "cola":
        data = CoLA()
    elif data_name.lower() == "plausibility":
        data = Plausibility()
    elif data_name.lower() == "scala":
        data = Scala()
    elif data_name.lower() == "itacola":
        data = ITACoLA()
    elif data_name.lower() == "rucola":
        data = RuCoLA()
    elif data_name.lower() == "jcola":
        data = JCoLA()
    elif data_name.lower() == "sling":
        data = SLING()
    elif data_name.lower() == "blimp-nl":
        data = BLiMP_nl()
    else:
        data = SynthData(data_name, perturb=perturb)

    ds = data.load_data()  # {count: subdict}
    data_len = len(ds["good_txt"])


    model = hfModels(model_name=model_name)
    run_id = _make_run_id(
        perturb=perturb,
        incremental=incremental,
    )

    data_dir = PROJECT_ROOT / "results" / "inference" / data_name / model_name
    os.makedirs(data_dir, exist_ok=True)

    # Standard non-few-shot path
    all_norm_probs, all_labels = [], []
    correct_num, total_num = 0, 0
    output: Dict[int, Dict[str, Union[torch.Tensor, float]]] = {}

    for count in tqdm(range(data_len)):
        total_num += 1
        good_ex = ds["good_txt"][count]
        bad_ex = ds["bad_txt"][count]
        res = forward_pass(
            good_ex,
            bad_ex,
            model,
            incremental=incremental
        )

        good_prob = res["norm_good_prob"]
        bad_prob = res["norm_bad_prob"]

        if isinstance(good_prob, (int, float)):
            all_labels += [1]
            all_norm_probs += [float(good_prob)]
        if isinstance(bad_prob, (int, float)):
            all_labels += [0]
            all_norm_probs += [float(bad_prob)]
            if isinstance(good_prob, (int, float)) and good_prob > bad_prob:
                correct_num += 1
        output[count] = res

    shard_all(
        model_name,
        data=data_name,
        data_file=output,
        perturb=perturb,
        incremental=incremental,
    )
    metrics_path = data_dir / f"{run_id}_results.txt"

    with open(metrics_path, "w+") as f:
        if all_norm_probs:
            nonpair_auc_score = binary_auroc(
                torch.tensor(all_norm_probs),
                torch.tensor(all_labels),
            )
            f.write(f"acc: {correct_num / total_num}\n")
            f.write(f"nonpair_auc:{nonpair_auc_score}\n")
        else:
            f.write("acc: n/a\n")
            f.write("nonpair_auc:n/a\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        "--data",
        type=str,
        default="blimp",
        help="Dataset to run the inference on",
    )
    parser.add_argument("--model", type=str, default="olmo2-7B", help="Model name")
    parser.add_argument("--perturb", default="all")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Save per-token hidden states and log-probs (starting from the second token).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference_general(
        data_name=args.data,
        model_name=args.model,
        perturb=args.perturb,
        incremental=args.incremental,
    )
