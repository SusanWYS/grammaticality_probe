#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_PATH = PROJECT_ROOT / "results"

MODELS = [
    "olmo2-7B",
    "olmo3-7B",
    "llama3-2-1B",
    "llama3-1-8B",
    "gemma2-2B",
    "gemma2-9B",
]

MODEL_LABELS = {
    "olmo2-7B": "OLMo-2-7B",
    "olmo3-7B": "OLMo-3-7B",
    "llama3-2-1B": "Llama-3.2-1B",
    "llama3-1-8B": "Llama-3.1-8B",
    "gemma2-2B": "Gemma-2-2B",
    "gemma2-9B": "Gemma-2-9B",
}

PANEL_LABELS = {"cola": "CoLA", "syngym": "SyntaxGym", "syntaxgym": "SyntaxGym"}

# ---------------------------------------------------------------------------
# Style (MATCHES your 2nd file)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

TRANSFER_COLOR = "purple"
ORIGINAL_COLOR = "#d95f02"


def _is_valid(val: object) -> bool:
    try:
        return val is not None and not math.isnan(float(val))
    except Exception:
        return False


def _extract_auc(stats: Dict[str, object]) -> Optional[float]:
    overall = stats.get("overall_stats", stats) if isinstance(stats, dict) else {}
    candidates = [
        overall.get("auc"),
        overall.get("test_auc"),
        overall.get("val_auc"),
    ]

    best_auc = overall.get("best_auc")
    if isinstance(best_auc, dict):
        candidates.extend(
            [
                best_auc.get("auc"),
                best_auc.get("val_auc"),
                best_auc.get("test_auc"),
            ]
        )

    for cand in candidates:
        if _is_valid(cand):
            return float(cand)
    return None


def _collect_best_auc(
    data_name: str,
    *,
    transfer: bool,
    train_data: str,
    add_prob: bool,
) -> Dict[str, Optional[float]]:
    base_dir = RES_PATH / "probe" / ("l2_prob" if add_prob else "l2") / data_name
    aucs: Dict[str, Optional[float]] = {}

    for model in MODELS:
        model_dir = base_dir / model
        if not model_dir.exists():
            aucs[model] = None
            continue

        best_auc = None
        for pkl_path in sorted(model_dir.glob("*.pkl")):
            name = pkl_path.name
            has_transfer_suffix = f"trained_on_{train_data}_tested_on_{data_name}" in name

            if transfer and not has_transfer_suffix:
                continue
            if not transfer and has_transfer_suffix:
                continue

            with open(pkl_path, "rb") as fh:
                stats = pickle.load(fh)

            auc_val = _extract_auc(stats)
            if _is_valid(auc_val) and (best_auc is None or float(auc_val) > best_auc):
                best_auc = float(auc_val)

        aucs[model] = best_auc

    return aucs


def _global_ylim(*series: Dict[str, Optional[float]], pad_frac: float = 0.05) -> Tuple[float, float]:
    vals: List[float] = []
    for collection in series:
        for val in collection.values():
            if _is_valid(val):
                vals.append(float(val))

    if not vals:
        return (0.0, 1.0)

    min_v, max_v = min(vals), max(vals)
    span = max(max_v - min_v, 0.05)
    pad = pad_frac * span
    lo = max(0.0, min_v - pad)
    hi = min(1.0, max_v + pad)
    if hi <= lo:
        hi = min(1.0, lo + 0.05)
    return (lo, hi)


def _plot_panel(
    ax,
    *,
    dataset_key: str,
    originals: Dict[str, Optional[float]],
    transfers: Dict[str, Optional[float]],
) -> None:
    xs = list(range(len(MODELS)))

    # Clean look (matches 2nd file)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    orig_xs: List[int] = []
    orig_vals: List[float] = []
    trans_xs: List[int] = []
    trans_vals: List[float] = []

    for idx, model in enumerate(MODELS):
        orig_val = originals.get(model)
        trans_val = transfers.get(model)

        if _is_valid(orig_val):
            orig_xs.append(idx)
            orig_vals.append(float(orig_val))
        if _is_valid(trans_val):
            trans_xs.append(idx)
            trans_vals.append(float(trans_val))

    # Marker sizing/linewidth to match 2nd file
    if trans_xs:
        ax.scatter(
            trans_xs,
            trans_vals,
            s=36,
            marker="s",
            facecolors="white",
            edgecolors=TRANSFER_COLOR,
            linewidths=1.0,
            label="Supervised",
            zorder=3,
            clip_on=False,
        )

    if orig_xs:
        ax.scatter(
            orig_xs,
            orig_vals,
            s=36,
            marker="o",
            facecolors="white",
            edgecolors=ORIGINAL_COLOR,
            linewidths=1.0,
            label="Unsupervised",
            zorder=3,
            clip_on=False,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=60, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(PANEL_LABELS.get(dataset_key, dataset_key.title()))
    ax.set_xlim(-0.6, len(MODELS) - 0.4)


def plot_transfer_vs_original(
    output_path: Path,
    *,
    train_data: str = "blimp",
    add_prob: bool = False,
) -> None:
    cola_originals = _collect_best_auc("cola", transfer=False, train_data=train_data, add_prob=add_prob)
    cola_transfers = _collect_best_auc("cola", transfer=True, train_data=train_data, add_prob=add_prob)
    syngym_originals = _collect_best_auc("syngym", transfer=False, train_data=train_data, add_prob=add_prob)
    syngym_transfers = _collect_best_auc("syngym", transfer=True, train_data=train_data, add_prob=add_prob)

    ymin, ymax = _global_ylim(cola_originals, cola_transfers, syngym_originals, syngym_transfers)

    # Figure ratio/size matches 2nd file (ACL-ish one-column row)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.65), sharey=True)
    for ax in axes:
        ax.tick_params(axis="y", which="both", labelleft=True)
    left_ax, right_ax = axes

    _plot_panel(left_ax, dataset_key="cola", originals=cola_originals, transfers=cola_transfers)
    _plot_panel(right_ax, dataset_key="syngym", originals=syngym_originals, transfers=syngym_transfers)

    for ax in axes:
        ax.set_ylim(ymin, ymax)

    # Keep your combined legend, but sized/placed like your 2nd file
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    if handles:
        unique: Dict[str, object] = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle

        fig.legend(
            list(unique.values()),
            list(unique.keys()),
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.22),
            borderaxespad=0.0,
            columnspacing=0.6,
            handletextpad=0.4,
        )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.28, wspace=0.35)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AUC of original L2 probes vs. BLiMP-transfer probes on CoLA and SyntaxGym."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("l2_original_vs_blimp_transfer_auc.pdf"),
        help="Output file for the figure.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="blimp",
        help="Training dataset name used for transfer probes (defaults to 'blimp').",
    )
    parser.add_argument(
        "--add-prob",
        action="store_true",
        help="Use add_prob directories (l2_prob) instead of plain l2 results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_transfer_vs_original(args.output, train_data=args.train_data, add_prob=args.add_prob)


if __name__ == "__main__":
    main()
