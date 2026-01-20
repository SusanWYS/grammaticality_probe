import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from tqdm import tqdm

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
)
from baseline.helpers import shard_all
from models.all_models import hfModels

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]

def run_general_fewshot(
    ds: Dict[str, Dict[int, str]],
    data_len: int,
    model: hfModels,
    data_dir: Path,
    *,
    results_data_name: str,
) -> None:
    """Run few-shot evaluation for general datasets."""

    correct_num, total_num = 0, 0
    output: Dict[int, Dict[str, float]] = {}

    for count in tqdm(range(data_len)):
        good_ex = ds["good_txt"][count]
        bad_ex = ds["bad_txt"][count]
        good_probs = model.fewshot_in_english(
            good_ex
        )
        bad_probs = model.fewshot_in_english(
            bad_ex
        )

        # fewshot_in_english (default) returns "Yes"/"No".
        g_yes, g_no = good_probs["Yes"], good_probs["No"]
        b_yes, b_no = bad_probs["Yes"], bad_probs["No"]

        correct_num += int(g_yes > g_no)
        correct_num += int(b_no > b_yes)
        total_num += 2

        output[count] = {
            "good_grammatical_prob": g_yes,
            "good_ungrammatical_prob": g_no,
            "bad_grammatical_prob": b_yes,
            "bad_ungrammatical_prob": b_no,
        }

    shard_all(
        model.model_name,
        data=results_data_name,
        data_file=output,
        fewshot=True,
    )
    metrics_fname = f"all_-1_fewshot_results.txt"
    with open(data_dir / metrics_fname, "w") as f:
        f.write(f"acc: {correct_num / total_num}\n")

    print(f"\n[RESULTS FEW-SHOT] acc={correct_num/total_num:.6f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for running few-shot inference directly."""

    parser = argparse.ArgumentParser(description="Run few-shot inference only")
    parser.add_argument("--data", type=str, default="blimp", help="Dataset name")
    parser.add_argument("--model", type=str, default="olmo2-7B", help="Model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.lower()
    model = hfModels(model_name=args.model)
    if data == "blimp":
        dataset = BLiMP()
    elif data == "syntaxgym":
        dataset = SyntaxGym()
    elif data == "cola":
        dataset = CoLA()
    else:
        raise Exception(f"Fewshot not supported for {data}")

    ds = dataset.load_data()
    data_len = len(ds["good_txt"])

    results_data_name = args.data

    data_dir = PROJECT_ROOT / "results" / "inference" / results_data_name / args.model
    os.makedirs(data_dir, exist_ok=True)

    run_general_fewshot(
        ds,
        data_len,
        model,
        data_dir,
        results_data_name=results_data_name,
    )


if __name__ == "__main__":
    main()
