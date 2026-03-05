#!/usr/bin/env python3
from __future__ import annotations

"""
Plot layer-wise neuron distributions for L1 sparse probe selections.

The script arranges 6 rows (models) by 4 columns (sparsity levels) and, for
each panel, plots a histogram of which layers the selected neurons come from.

Sparsity levels are interpreted as per-mille values (0.1, 0.5, 1, 5 permille). The
script searches for the corresponding ``num{level}`` pickle files produced by
``probe/l1_sparse_probe.py`` under ``results/probe/l1[/...]/<data>/<model>``.

Example:
    python analysis/plot_l1_histogram.py \
        --data synthetic --rev_ind -1 --output l1_sparse_layers.pdf
"""

import argparse
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from models.helpers import model_hdim, model_layers
from probe.helpers import RES_PATH

# Per-mille sparsity levels (0.1permille, 0.5permille, 1permille, 5permille)
SPARSITY_LEVELS: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)

# Models are ordered to match the rest of the analysis figures
DEFAULT_MODELS: Tuple[str, ...] = (
    "olmo2-7B",
    "olmo3-7B",
    "llama3-2-1B",
    "llama3-1-8B",
    "gemma2-2B",
    "gemma2-9B",
)

model_maps = {
    "olmo2-7B": "OLMo-2-7B",
    "olmo3-7B": "OLMo-3-7B",
    "llama3-2-1B": "Llama-3.2-1B",
    "llama3-1-8B": "Llama-3.1-8B",
    "gemma2-2B": "Gemma-2-2B",
    "gemma2-9B": "Gemma-2-9B",
}


def _candidate_patterns(level: float, rev_ind: int) -> List[str]:
    """Return filename glob patterns that may match a given sparsity level.

    Pickles are saved with ``num{target}`` where ``target`` is the raw
    ``target_num`` passed to ``run_pipeline``. To bridge the percent vs
    per-mille interpretation, we search for both ``level`` and ``level * 0.1``.
    """
    patterns: List[str] = []
    numeric_candidates = {level, level}
    for numeric_val in numeric_candidates:
        for fmt in (f"{numeric_val:g}", f"{numeric_val:.2f}", f"{numeric_val:.3f}"):
            patterns.append(f"{rev_ind}_layerall*num{fmt}.pkl")
    seen = set()
    deduped: List[str] = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            deduped.append(pattern)
    return deduped


def _find_result_file(base_dir: Path, level: float, rev_ind: int) -> Optional[Path]:
    """Locate the pickle produced by l1_sparse_probe for a sparsity level."""
    if not base_dir.exists():
        return None

    candidates: List[Path] = []
    for pattern in _candidate_patterns(level, rev_ind):
        candidates.extend(sorted(base_dir.glob(pattern)))

    if not candidates:
        return None

    def _sort_key(path: Path) -> Tuple[int, str]:
        return (1 if "random" in path.stem else 0, path.name)

    return sorted(candidates, key=_sort_key)[0]


def _load_non_zero_neurons(pkl_path: Path) -> Optional[List[int]]:
    """Extract the list of selected neurons from a probe result pickle."""
    try:
        with open(pkl_path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None

    if not isinstance(payload, Mapping):
        return None

    l2_best = payload.get("l2_best")
    if isinstance(l2_best, Mapping):
        neurons = l2_best.get("non_zero_neurons")
    else:
        neurons = None

    if neurons is None:
        return None

    try:
        return [int(idx) for idx in neurons]
    except Exception:
        return None


def _layer_percentages(neuron_indices: Sequence[int], model_name: str) -> np.ndarray:
    """Convert flattened neuron indices into layer-percentage positions."""
    if model_name not in model_hdim or model_name not in model_layers:
        raise KeyError(f"Model {model_name} missing hidden dim or layer metadata")

    hidden_size = int(model_hdim[model_name])
    num_layers = int(model_layers[model_name])

    layer_ids = np.asarray(neuron_indices, dtype=int) // hidden_size
    return (layer_ids + 1) / (num_layers + 1)


def _collect_ratios(
    data: str,
    models: Sequence[str],
    sparsity_levels: Sequence[float],
    *,
    rev_ind: int,
    add_prob: bool,
) -> Dict[str, Dict[float, Optional[np.ndarray]]]:
    """Load all layer-percentage vectors keyed by model and sparsity."""
    base_probe = "l1_prob" if add_prob else "l1"
    by_model: Dict[str, Dict[float, Optional[np.ndarray]]] = {}

    for model_name in models:
        per_level: Dict[float, Optional[np.ndarray]] = {}
        model_dir = RES_PATH / "probe" / base_probe / data / model_name
        for level in sparsity_levels:
            pkl_path = _find_result_file(model_dir, level, rev_ind)
            if pkl_path is None:
                per_level[level] = None
                continue

            neuron_indices = _load_non_zero_neurons(pkl_path)
            if not neuron_indices:
                per_level[level] = None
                continue

            try:
                per_level[level] = _layer_percentages(neuron_indices, model_name)
            except Exception:
                per_level[level] = None

        by_model[model_name] = per_level

    return by_model


def _match_alpha_range(filename: str, start_exp: int, end_exp: int) -> bool:
    return f"alpha{start_exp}_to_{end_exp}" in filename or f"l2_{start_exp}_to_{end_exp}" in filename


def _extract_layer_idx_from_name(filename: str) -> Optional[int]:
    match = re.search(r"layer(\d+)", filename)
    return int(match.group(1)) if match else None


def _load_best_l2_layers(
    data: str,
    models: Sequence[str],
    *,
    rev_ind: int,
    start_alpha: int = -2,
    end_alpha: int = 5,
    add_prob: bool = False,
) -> Dict[str, Optional[int]]:
    """Find the best-performing L2 probe layer for each model, if available.

    NOTE: We still compute this (cheap) in case you want to re-enable overlays
    later, but we do not draw any highlight bars/legends in the plot.
    """
    probe_dir = "l2_prob" if add_prob else "l2"
    best_layers: Dict[str, Optional[int]] = {}

    for model_name in models:
        base_dir = RES_PATH / "probe" / probe_dir / data / model_name
        if not base_dir.exists():
            best_layers[model_name] = None
            continue

        best_layer: Optional[int] = None
        best_score = -math.inf

        for pkl_path in sorted(base_dir.glob("*.pkl")):
            name = pkl_path.name
            if f"{rev_ind}_" not in name or not _match_alpha_range(name, start_alpha, end_alpha):
                continue

            layer_idx = _extract_layer_idx_from_name(name)
            if layer_idx is None:
                continue

            try:
                with open(pkl_path, "rb") as fh:
                    stats = pickle.load(fh)
            except Exception:
                continue

            overall = stats.get("overall_stats", {}) if isinstance(stats, Mapping) else {}
            best_auc = overall.get("best_auc", {}) if isinstance(overall, Mapping) else {}
            val_auc = best_auc.get("val_auc") if isinstance(best_auc, Mapping) else None
            val_acc = best_auc.get("val_acc") if isinstance(best_auc, Mapping) else None

            if isinstance(val_auc, (int, float)) and not math.isnan(val_auc):
                score = float(val_auc)
            elif isinstance(val_acc, (int, float)) and not math.isnan(val_acc):
                score = float(val_acc)
            else:
                continue

            if score > best_score:
                best_score = score
                best_layer = layer_idx

        best_layers[model_name] = best_layer

    return best_layers


def _style_panel_axes(ax: plt.Axes) -> None:
    """Keep only x/y axes (bottom/left spines); remove other panel borders."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(top=False, right=False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def _plot_histograms(
    ratios: Dict[str, Dict[float, Optional[np.ndarray]]],
    *,
    sparsity_levels: Sequence[float],
    output_path: Path,
) -> None:
    num_models = len(ratios)
    num_levels = len(sparsity_levels)

    # shorter panels
    panel_w = 3.5
    panel_h = 1.45
    fig, axes = plt.subplots(
        nrows=num_models,
        ncols=num_levels,
        figsize=(panel_w * num_levels, panel_h * num_models),
        sharex=True,
        squeeze=False,
    )

    for row_idx, (model_name, per_level) in enumerate(ratios.items()):
        num_layers = int(model_layers.get(model_name, 0))
        if num_layers <= 0:
            layer_bins = np.linspace(0.0, 1.0, 21)
        else:
            layer_bins = np.linspace(0.0, 1.0, num_layers + 1)

        for col_idx, level in enumerate(sparsity_levels):
            ax = axes[row_idx][col_idx]
            _style_panel_axes(ax)

            values = per_level.get(level)
            num_neurons = int(values.size) if isinstance(values, np.ndarray) else 0

            if values is None or values.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.set_yticks([])
            else:
                counts, edges = np.histogram(values, bins=layer_bins)
                widths = np.diff(edges)
                ax.bar(
                    edges[:-1],
                    counts,
                    width=widths,
                    align="edge",
                    color="lightslategray",
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.85,
                )

            ax.set_xlim(0, 1.0)
            ax.set_xticks(np.linspace(0.0, 1.0, 11))
            ax.set_xticklabels([f"{tick:.1f}" for tick in np.linspace(0.0, 1.0, 11)],rotation=25,fontsize=10)

            if row_idx == num_models - 1:
                ax.set_xlabel("Layer Depth (layer / total)", fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(f"{model_maps[model_name]}", fontsize=12)

            if row_idx == 0:
                ax.set_title(f"Sparsity: {level}%", fontsize=14)

            ax.text(
                0.98,
                0.92,
                f"n={num_neurons}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )


    fig.tight_layout(pad=0.6, w_pad=0.4, h_pad=0.3)
    fig.savefig(output_path, dpi=300)
    print(f"Saved histogram grid to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot L1 sparse neuron layer histograms")
    parser.add_argument("--data", type=str, default="synthetic", help="Dataset name")
    parser.add_argument("--rev_ind", type=int, default=-1, help="Revision index used in filenames")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to plot (default: six primary models)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("l1_sparse_layer_hist.pdf"),
        help="Where to save the figure",
    )
    parser.add_argument("--add_prob", action="store_true", help="Use l1_prob results instead of l1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = _collect_ratios(
        data=args.data,
        models=args.models,
        sparsity_levels=SPARSITY_LEVELS,
        rev_ind=args.rev_ind,
        add_prob=args.add_prob,
    )

    # Still computed (unused in plotting) in case you want to re-enable later.
    _ = _load_best_l2_layers(
        data=args.data,
        models=args.models,
        rev_ind=args.rev_ind,
        add_prob=args.add_prob,
    )

    _plot_histograms(
        ratios,
        sparsity_levels=SPARSITY_LEVELS,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
