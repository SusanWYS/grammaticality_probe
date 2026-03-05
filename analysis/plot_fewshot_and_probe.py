#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_PATH = PROJECT_ROOT / "results"

MODEL_MAPS: Dict[str, str] = {
    "olmo2-7B": "OLMo-2-7B",
    "olmo3-7B": "OLMo-3-7B",
    "llama3-2-1B": "Llama-3.2-1B",
    "llama3-1-8B": "Llama-3.1-8B",
    "gemma2-2B": "Gemma-2-2B",
    "gemma2-9B": "Gemma-2-9B",
}
DEFAULT_MODELS: Sequence[str] = tuple(MODEL_MAPS.keys())
TARGET_DATASETS: Sequence[str] = ("blimp", "cola", "syngym")

DATA_LABELS: Dict[str, str] = {
    "blimp": "BLiMP",
    "cola": "CoLA",
    "syngym": "SyntaxGym",
    "eventknow": "EventKnow",
    "scala": "Scala",
    "itacola": "ItaCoLA",
    "rucola": "RuCoLA",
    "jcola": "JCoLA",
    "sling": "SLING",
}

FEWSHOT_COLOR = "#d95f02"
PROBE_COLOR = "cadetblue"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 9,
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def _is_valid(val: object) -> bool:
    try:
        return val is not None and not np.isnan(float(val))
    except Exception:
        return False


def _display_model(model: str) -> str:
    return MODEL_MAPS.get(model, model)


def _parse_acc_from_file(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with open(path, "r") as fh:
        for line in fh:
            if "acc" not in line:
                continue
            try:
                # Allow formats like "acc: 0.83" or "acc=0.83"
                _, after = line.split("acc", 1)
                for token in after.replace("=", " ").replace(":", " ").split():
                    if _is_valid(token):
                        return float(token)
            except ValueError:
                continue
    return None


def _choose_latest_fewshot_file(
    base_dir: Path, *, name_filter: Optional[str], perturb_prefix: Optional[str]
) -> Optional[Path]:
    candidates = []
    for path in base_dir.glob("*fewshot*_results.txt"):
        filename = path.name
        if name_filter and name_filter not in filename:
            continue
        if perturb_prefix and not filename.startswith(perturb_prefix):
            continue
        candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_fewshot_accs(
    data_name: str,
    models: Iterable[str],
    *,
    name_filter: Optional[str],
    perturb_prefix: Optional[str],
) -> Dict[str, Optional[float]]:
    accs: Dict[str, Optional[float]] = {}
    base = RES_PATH / "inference" / ("blimp" if data_name == "blimp" else data_name)

    for model in models:
        model_dir = base / model
        if not model_dir.exists():
            accs[model] = None
            continue
        target_file = _choose_latest_fewshot_file(
            model_dir, name_filter=name_filter, perturb_prefix=perturb_prefix
        )
        accs[model] = _parse_acc_from_file(target_file) if target_file else None
    return accs


def _select_best_probe_stats(model_dir: Path, *, concat: bool) -> Optional[dict]:
    best_score: Optional[float] = None
    best_stats: Optional[dict] = None

    for pkl_path in sorted(model_dir.glob("*.pkl")):
        if concat and "_concat" not in pkl_path.name:
            continue
        if not concat and "_concat" in pkl_path.name:
            continue

        with open(pkl_path, "rb") as fh:
            stats = pickle.load(fh)

        overall = stats.get("overall_stats", {}) if isinstance(stats, dict) else {}
        val_auc = overall.get("auc")
        val_acc = overall.get("test_paired_acc")
        score = val_auc if _is_valid(val_auc) else val_acc
        if not _is_valid(score):
            continue

        score = float(score)
        if best_score is None or score > best_score:
            best_score = score
            best_stats = stats

    return best_stats


def _extract_overall_acc(stats: Optional[dict]) -> Optional[float]:
    if not isinstance(stats, dict):
        return None
    overall = stats.get("overall_stats", {})
    for key in ("acc", "test_acc", "val_acc"):
        val = overall.get(key)
        if _is_valid(val):
            return float(val)
    return None


def load_probe_accs(
    data_name: str,
    models: Iterable[str],
    *,
    concat: bool,
    use_prob_variant: bool,
) -> Dict[str, Optional[float]]:
    accs: Dict[str, Optional[float]] = {}
    base_dir = RES_PATH / "probe"
    prob_dir = base_dir / "l2_prob" / data_name
    probe_root = prob_dir if use_prob_variant and prob_dir.exists() else base_dir / "l2" / data_name

    for model in models:
        model_dir = probe_root / model
        if not model_dir.exists():
            accs[model] = None
            continue
        best_stats = _select_best_probe_stats(model_dir, concat=concat)
        accs[model] = _extract_overall_acc(best_stats)
    return accs


def plot_bars(
    data_name: str,
    models: Sequence[str],
    fewshot: Dict[str, Optional[float]],
    probe: Dict[str, Optional[float]],
    ax: plt.Axes,
) -> Sequence[plt.Artist]:
    xs = np.arange(len(models))
    width = 0.35

    fewshot_vals = [val if _is_valid(val) else np.nan for val in (fewshot.get(m) for m in models)]
    probe_vals = [val if _is_valid(val) else np.nan for val in (probe.get(m) for m in models)]

    bars1 = ax.bar(
        xs - width / 2,
        fewshot_vals,
        width,
        color=FEWSHOT_COLOR,
        label="Few-shot ACC",
        alpha=0.9,
    )
    bars2 = ax.bar(
        xs + width / 2,
        probe_vals,
        width,
        color=PROBE_COLOR,
        label="$\ell_2$ Probe Score",
        alpha=0.8,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([_display_model(m) for m in models], rotation=25, ha="right")
    ax.set_ylabel("Non-Pairwise ACC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{DATA_LABELS.get(data_name, data_name)}")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, axis="y")
    return (bars1[0], bars2[0])


def plot_panels(
    datasets: Sequence[str],
    models: Sequence[str],
    fewshot_accs: Dict[str, Dict[str, Optional[float]]],
    probe_accs: Dict[str, Dict[str, Optional[float]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(13.5, 3.6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax in axes:
        ax.tick_params(axis="y", which="both", labelleft=True)
    legend_handles: Optional[Sequence[plt.Artist]] = None

    for ax, data_name in zip(axes, datasets):
        handles = plot_bars(
            data_name,
            models,
            fewshot_accs.get(data_name, {}),
            probe_accs.get(data_name, {}),
            ax,
        )
        if legend_handles is None:
            legend_handles = handles

    if legend_handles:
        fig.legend(legend_handles, ["Fewshot (Metalinguistic)", "$\ell_2$ Probe Score"], loc="upper center", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot few-shot accuracies alongside best-layer L2 probe accuracies "
            "for BLiMP, CoLA, and SyntaxGym on a shared panel figure."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=TARGET_DATASETS,
        help="Datasets to include (default: blimp cola syngym).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to include (default: known model set).",
    )
    parser.add_argument(
        "--fewshot-name-filter",
        default=None,
        help="Optional substring to require in few-shot result filenames (e.g., patch5%%).",
    )
    parser.add_argument(
        "--perturb-prefix",
        default=None,
        help="Optional filename prefix filter for few-shot results (e.g., 'all' or 'syntax').",
    )
    parser.add_argument(
        "--use-prob-probe",
        action="store_true",
        help="Use the l2_prob directory if present when loading probe results.",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Use concatenated probe checkpoints (filenames containing '_concat').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot. Defaults to analysis/blimp_cola_syngym_fewshot_probe.pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models: Sequence[str] = tuple(args.models)
    datasets: Sequence[str] = tuple(args.datasets)
    dataset_slug = "_".join(datasets)
    output_path = (
        args.output
        if args.output
        else PROJECT_ROOT / "analysis" / f"{dataset_slug}_fewshot_probe.pdf"
    )

    fewshot_accs = {
        data_name: load_fewshot_accs(
            data_name,
            models,
            name_filter=args.fewshot_name_filter,
            perturb_prefix=args.perturb_prefix,
        )
        for data_name in datasets
    }
    probe_accs = {
        data_name: load_probe_accs(
            data_name, models, concat=args.concat, use_prob_variant=args.use_prob_probe
        )
        for data_name in datasets
    }

    print("Few-shot accuracies:")
    for data_name in datasets:
        print(f"Dataset: {data_name}")
        for model in models:
            print(f"  {model}: {fewshot_accs.get(data_name, {}).get(model)}")
    print("Best L2 probe accuracies (overall acc):")
    for data_name in datasets:
        print(f"Dataset: {data_name}")
        for model in models:
            print(f"  {model}: {probe_accs.get(data_name, {}).get(model)}")

    plot_panels(datasets, models, fewshot_accs, probe_accs, output_path)


if __name__ == "__main__":
    main()
