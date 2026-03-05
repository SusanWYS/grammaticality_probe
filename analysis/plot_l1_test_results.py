#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from sklearn.metrics import roc_auc_score

from probe.helpers import RES_PATH


# Now interpreted as per-mille values: 0.1permille, 0.5permille, 1permille, 5permille
X_TICKS: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)
#
DEFAULT_MODELS: Tuple[str, ...] = (
    "llama3-2-1B",
    "llama3-1-8B",
    "gemma2-2B",
    "gemma2-9B",
)

# Canonical dataset display names
DATA = {
    "blimp": "BLiMP",
    "cola": "CoLA",
    "syngym": "SyntaxGym",
    "eventknow": "Plausibility Dataset",
    "scala": "ScaLA",
    "sling": "SLING",
    "nl-blimp": "BLiMP-NL",
    "itacola": "ItaCoLA",
    "rucola": "RuCoLA",
    "jcola": "JCoLA",
}

# Aliases users might pass (or older directory naming conventions).
DATA_ALIASES: Dict[str, str] = {
    "blimp-nl": "nl-blimp",
    "blimp_nl": "nl-blimp",
    "syntaxgym": "syngym",
}

# If data NOT in this set, we plot a single row (AUC only) in the single-dataset mode.
PAIRED_DATASETS = {"blimp", "sling", "nl-blimp"}

ResultPoint = Tuple[float, Optional[float], Optional[float], Optional[float]]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

# Figure sizes (your existing template sizes)
FIGSIZE_2x3: Tuple[float, float] = (4.30, 4.30 / 1.47)  # (w, h) for 2x3
FIGSIZE_1x3: Tuple[float, float] = (4.30, (4.30 / 1.47) * 0.62)  # shorter height for 1 row

# Row height used for stacked layouts (keep consistent with your 2-row template)
_STACK_ROW_H: float = (4.30 / 1.47) / 2.0  # ~height per metric row in the 2x3 plot

# Keep your exact model colors
COLOR_BY_MODEL: Dict[str, str] = {
    "olmo2-7B": "lightsteelblue",
    "olmo3-7B": "cadetblue",
    "llama3-2-1B": "lightcoral",
    "llama3-1-8B": "darkred",
    "gemma2-2B": "yellowgreen",
    "gemma2-9B": "darkolivegreen",
}

# Column grouping (your template)
FAMILIES: Tuple[Tuple[str, Tuple[str, str]], ...] = (
    ("OLMo", ("olmo2-7B", "olmo3-7B")),
    ("Llama", ("llama3-2-1B", "llama3-1-8B")),
    ("Gemma", ("gemma2-2B", "gemma2-9B")),
)

# Short labels for the per-column color legends
SHORT_LABEL: Dict[str, str] = {
    "olmo2-7B": "2-7B",
    "olmo3-7B": "3-7B",
    "llama3-2-1B": "3.2-1B",
    "llama3-1-8B": "3.1-8B",
    "gemma2-2B": "2-2B",
    "gemma2-9B": "2-9B",
}


def _canonical_data_key(data: str) -> str:
    d = (data or "").strip()
    d = DATA_ALIASES.get(d, d)
    return d


def _dataset_display(data: str) -> str:
    d = _canonical_data_key(data)
    return DATA.get(d, d)


def _is_valid(val: object) -> bool:
    try:
        return val is not None and not math.isnan(float(val))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _paired_accuracy(meta: List[Tuple[int, bool]], scores: Iterable[float]) -> Tuple[Optional[float], int]:
    by: Dict[int, List[Optional[float]]] = {}
    for (orig_idx, is_good), score in zip(meta, scores):
        slot = by.setdefault(int(orig_idx), [None, None])
        slot[1 if is_good else 0] = float(score)

    total = 0
    correct = 0
    for bad_good in by.values():
        bad, good = bad_good[0], bad_good[1]
        if bad is None or good is None:
            continue
        total += 1
        correct += 1 if good > bad else 0

    if total == 0:
        return None, 0
    return float(correct) / float(total), total


def _collect_per_example(stats: Dict[str, Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, bool]]]:
    labels: List[int] = []
    scores: List[float] = []
    meta: List[Tuple[int, bool]] = []

    for key, entry in stats.items():
        if key == "overall_stats":
            continue
        good_bad = ((True, entry.get("good")), (False, entry.get("bad")))
        for is_good, side in good_bad:
            if not side:
                continue
            score = side.get("probe_score")
            if not _is_valid(score):
                continue
            labels.append(1 if is_good else 0)
            scores.append(float(score))
            meta.append((int(key), is_good))

    return np.asarray(labels), np.asarray(scores), meta


def _compute_metrics(stats: Dict[str, Dict[str, object]]) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    labels, scores, meta = _collect_per_example(stats)
    if int(labels.size) == 0:
        return {"pairwise_acc": (None, None, None), "auc": (None, None, None)}

    paired_acc, _ = _paired_accuracy(meta, scores)
    auc_val = None
    if len(np.unique(labels)) > 1:
        auc_val = float(roc_auc_score(labels, scores))

    return {
        "pairwise_acc": (paired_acc, None, None),
        "auc": (auc_val, None, None),
    }


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------


def _locate_result_file(base_dir: Path, tick: float) -> Optional[Path]:
    candidates = [
        base_dir / f"-1_layerall_l1_test_-2_to_5_{tick:.1f}.pkl",
        base_dir / f"-1_layerall_l1_test_-2_to_5_{tick:.2f}.pkl",
        base_dir / f"-1_layerall_l1_test_-2_to_5_{tick:.3f}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pattern = f"-1_layerall_l1_test_-2_to_5_{tick}" + "*.pkl"
    for candidate in base_dir.glob(pattern):
        return candidate
    return None


def _load_metrics(pkl_path: Path) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)
    stats = payload if isinstance(payload, dict) else {}
    return _compute_metrics(stats)


def _gather_results(base_dir: Path) -> Dict[str, List[ResultPoint]]:
    acc_points: List[ResultPoint] = []
    auc_points: List[ResultPoint] = []

    for tick in X_TICKS:
        pkl_path = _locate_result_file(base_dir, tick)
        if pkl_path is None:
            acc_points.append((tick, None, None, None))
            auc_points.append((tick, None, None, None))
            continue
        metrics = _load_metrics(pkl_path)
        acc_val, acc_low, acc_high = metrics.get("pairwise_acc", (None, None, None))
        auc_val, auc_low, auc_high = metrics.get("auc", (None, None, None))
        acc_points.append((tick, acc_val, acc_low, acc_high))
        auc_points.append((tick, auc_val, auc_low, auc_high))

    return {"acc": acc_points, "auc": auc_points}


def _locate_random_result_files(base_dir: Path, tick: float) -> List[Path]:
    patterns = [
        f"*-2_to_5_{tick*10:.1f}.pkl",
        f"*-2_to_5_{tick*10:.2f}.pkl",
        f"*-2_to_5_{tick*10:.3f}.pkl",
        f"*-2_to_5.pkl",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(base_dir.glob(pattern)))
    if files:
        return files
    files.extend(sorted(base_dir.glob(f"*{tick}*.pkl")))
    return files


def _mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if _is_valid(v)]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _gather_random_results(base_dir: Path) -> Dict[str, List[ResultPoint]]:
    acc_points: List[ResultPoint] = []
    auc_points: List[ResultPoint] = []

    for tick in X_TICKS:
        pkl_paths = _locate_random_result_files(base_dir, tick * 0.1)
        if not pkl_paths:
            acc_points.append((tick, None, None, None))
            auc_points.append((tick, None, None, None))
            continue

        acc_vals: List[Optional[float]] = []
        auc_vals: List[Optional[float]] = []
        for pkl_path in pkl_paths:
            metrics = _load_metrics(pkl_path)
            acc_val, _, _ = metrics.get("pairwise_acc", (None, None, None))
            auc_val, _, _ = metrics.get("auc", (None, None, None))
            acc_vals.append(acc_val)
            auc_vals.append(auc_val)

        acc_points.append((tick, _mean(acc_vals), None, None))
        auc_points.append((tick, _mean(auc_vals), None, None))

    return {"acc": acc_points, "auc": auc_points}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_TICK_VALS = list(X_TICKS)


def _format_axes(
    ax,
    *,
    show_ylabel: bool,
    ylabel: str,
    show_xlabel: bool,
    show_xticklabels: bool,
) -> None:
    ax.set_xticks(_TICK_VALS)
    ax.xaxis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.xaxis.set_major_formatter(fmt)

    xmin = min(_TICK_VALS) / 1.1
    xmax = max(_TICK_VALS) * 1.03
    ax.set_xlim(xmin, xmax)

    if show_xticklabels:
        ax.set_xticklabels([f"{t:g}" for t in _TICK_VALS])
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(90)
            lbl.set_ha("center")
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis="x", which="major", pad=2, labelbottom=show_xticklabels)
    ax.tick_params(axis="y", which="major", pad=2)

    if show_xlabel:
        ax.set_xlabel("Sparsity (%)", labelpad=2)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(ylabel, labelpad=3)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_series(
    ax,
    points: Iterable[ResultPoint],
    *,
    color: str,
    linestyle: str,
    linewidth: float = 1.2,
) -> None:
    xs: List[float] = []
    ys: List[float] = []
    for x, y, _lo, _hi in points:
        if not _is_valid(y):
            continue
        xs.append(float(x))
        ys.append(float(y))

    ax.plot(xs, ys, linestyle=linestyle, linewidth=linewidth, color=color)


def _snap_ylim_step(ymin: float, ymax: float, step: float = 0.05) -> Tuple[float, float]:
    ymin = max(0.0, math.floor(ymin / step) * step)
    ymax = min(1.0, math.ceil(ymax / step) * step)
    if ymax <= ymin:
        ymax = min(1.0, ymin + step)
    return ymin, ymax


def _row_ylim(series: List[Iterable[ResultPoint]], *, pad: float = 0.015) -> Tuple[float, float]:
    vals: List[float] = []
    for pts in series:
        for _x, y, _lo, _hi in pts:
            if _is_valid(y):
                vals.append(float(y))
    if not vals:
        return (0.0, 1.0)
    ymin = max(0.0, min(vals) - pad)
    ymax = min(1.0, max(vals) + pad)
    return _snap_ylim_step(ymin, ymax, step=0.05)


def _add_top_style_legend(fig, *, y: float) -> None:
    imp = Line2D([0], [0], color="dimgrey", lw=1.4, linestyle="-", label="Important Neurons")
    rnd = Line2D([0], [0], color="dimgrey", lw=1.4, linestyle="--", label="Random Neurons")
    fig.legend(
        handles=[imp, rnd],
        labels=["LASSO-Selected Neurons", "Random Neurons"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.2,
    )


# ---------------------------------------------------------------------------
# Single-dataset plot (original behavior)
# ---------------------------------------------------------------------------


def plot_new_template(
    data: str,
    models: Sequence[str],
    output_path: Path,
    *,
    show_important: bool = True,
    show_random: bool = True,
) -> None:
    data = _canonical_data_key(data)

    base_root = RES_PATH / "probe" / "l1" / data
    random_root = RES_PATH / "probe" / "l1_random" / data

    single_row = data not in PAIRED_DATASETS

    imp_results: Dict[str, Dict[str, List[ResultPoint]]] = {}
    rnd_results: Dict[str, Dict[str, List[ResultPoint]]] = {}

    for m in models:
        if show_important:
            imp_results[m] = _gather_results(base_root / m)
        if show_random:
            rnd_results[m] = _gather_random_results(random_root / m)

    if single_row:
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_1x3, sharex=True)
        axes_2d = np.asarray(axes).reshape(1, 3)
    else:
        fig, axes_2d = plt.subplots(2, 3, figsize=FIGSIZE_2x3, sharex=True)

    # Titles
    for j, (family_name, _family_models) in enumerate(FAMILIES):
        axes_2d[0, j].set_title(family_name)

    # Compute y-limits
    if single_row:
        auc_series: List[Iterable[ResultPoint]] = []
        for _family_name, fam_models in FAMILIES:
            for m in fam_models:
                if m not in models:
                    continue
                if show_important and m in imp_results:
                    auc_series.append(imp_results[m]["auc"])
                if show_random and m in rnd_results:
                    auc_series.append(rnd_results[m]["auc"])
        auc_ylim = _row_ylim(auc_series)
    else:
        acc_series: List[Iterable[ResultPoint]] = []
        auc_series = []
        for _family_name, fam_models in FAMILIES:
            for m in fam_models:
                if m not in models:
                    continue
                if show_important and m in imp_results:
                    acc_series.append(imp_results[m]["acc"])
                    auc_series.append(imp_results[m]["auc"])
                if show_random and m in rnd_results:
                    acc_series.append(rnd_results[m]["acc"])
                    auc_series.append(rnd_results[m]["auc"])
        acc_ylim = _row_ylim(acc_series)
        auc_ylim = _row_ylim(auc_series)

    # Plot per column
    for col, (_family_name, fam_models) in enumerate(FAMILIES):
        col_handles: List[Line2D] = []
        col_labels: List[str] = []

        for m in fam_models:
            if m not in models:
                continue
            color = COLOR_BY_MODEL.get(m, "black")

            if single_row:
                ax_auc = axes_2d[0, col]
                if show_important and m in imp_results:
                    _plot_series(ax_auc, imp_results[m]["auc"], color=color, linestyle="-")
                if show_random and m in rnd_results:
                    _plot_series(ax_auc, rnd_results[m]["auc"], color=color, linestyle="--")
            else:
                ax_acc = axes_2d[0, col]
                ax_auc = axes_2d[1, col]
                if show_important and m in imp_results:
                    _plot_series(ax_acc, imp_results[m]["acc"], color=color, linestyle="-")
                    _plot_series(ax_auc, imp_results[m]["auc"], color=color, linestyle="-")
                if show_random and m in rnd_results:
                    _plot_series(ax_acc, rnd_results[m]["acc"], color=color, linestyle="--")
                    _plot_series(ax_auc, rnd_results[m]["auc"], color=color, linestyle="--")

            col_handles.append(Line2D([0], [0], color=color, lw=1.4, linestyle="-"))
            col_labels.append(SHORT_LABEL.get(m, m))

        if single_row:
            _format_axes(
                axes_2d[0, col],
                show_ylabel=(col == 0),
                ylabel="AUC",
                show_xlabel=True,
                show_xticklabels=True,
            )
            axes_2d[0, col].set_ylim(*auc_ylim)
            axes_2d[0, col].yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Per-column legends (single dataset mode)
            if col_handles:
                axes_2d[0, col].legend(
                    handles=col_handles,
                    labels=col_labels,
                    ncol=1,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.1),
                    frameon=False,
                    borderaxespad=0.0,
                    handlelength=1.8,
                    handletextpad=0.5,
                )
        else:
            _format_axes(
                axes_2d[0, col],
                show_ylabel=(col == 0),
                ylabel="ACC",
                show_xlabel=False,
                show_xticklabels=False,
            )
            _format_axes(
                axes_2d[1, col],
                show_ylabel=(col == 0),
                ylabel="AUC",
                show_xlabel=True,
                show_xticklabels=True,
            )

            axes_2d[0, col].set_ylim(*acc_ylim)
            axes_2d[1, col].set_ylim(*auc_ylim)

            axes_2d[0, col].yaxis.set_major_locator(MaxNLocator(nbins=4))
            axes_2d[1, col].yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Per-column legends (single dataset mode)
            if col_handles:
                axes_2d[1, col].legend(
                    handles=col_handles,
                    labels=col_labels,
                    ncol=1,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.1),
                    frameon=False,
                    borderaxespad=0.0,
                    handlelength=1.8,
                    handletextpad=0.5,
                )

    _add_top_style_legend(fig, y=(1.15 if single_row else 1))
    # fig.suptitle(_dataset_display(data), y=1.10)

    # Layout tweaks
    if single_row:
        fig.subplots_adjust(
            left=0.08,
            right=0.995,
            top=0.86,
            bottom=0.35,  # more room for per-column legends
            wspace=0.15,
            hspace=0.00,
        )
    else:
        fig.subplots_adjust(
            left=0.08,
            right=0.995,
            top=0.95,
            bottom=0.25,
            wspace=0.15,
            hspace=0.12,
        )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.10)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Stacked multi-dataset layouts (new)
# ---------------------------------------------------------------------------


def _stack_row_specs(datasets: Sequence[str]) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (dataset_key, metric_key, metric_label).
    Paired datasets expand to ACC + AUC rows; others are AUC-only.
    """
    specs: List[Tuple[str, str, str]] = []
    for d0 in datasets:
        d = _canonical_data_key(d0)
        if d in PAIRED_DATASETS:
            specs.append((d, "acc", "ACC"))
            specs.append((d, "auc", "AUC"))
        else:
            specs.append((d, "auc", "AUC"))
    return specs


def plot_stacked_datasets(
    *,
    datasets: Sequence[str],
    families: Sequence[Tuple[str, Tuple[str, str]]],
    models: Sequence[str],
    output_path: Path,
    show_important: bool = True,
    show_random: bool = True,
    show_dataset_name_in_ylabel: bool = True,
) -> None:
    # Canonicalize datasets once
    canon_datasets = [_canonical_data_key(d) for d in datasets]
    row_specs = _stack_row_specs(canon_datasets)

    ncols = len(families)
    nrows = len(row_specs)

    # Width: keep your default width; height scales with number of metric rows.
    fig_w = 4.30
    fig_h = max(1.0, _STACK_ROW_H * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)
    axes_2d = np.asarray(axes)
    if axes_2d.ndim == 1:
        axes_2d = axes_2d.reshape(nrows, ncols)

    # Column titles (only at the very top row)
    for j, (family_name, _family_models) in enumerate(families):
        axes_2d[0, j].set_title(family_name)

    # Preload results per dataset/model
    imp_by_data: Dict[str, Dict[str, Dict[str, List[ResultPoint]]]] = {}
    rnd_by_data: Dict[str, Dict[str, Dict[str, List[ResultPoint]]]] = {}
    for d in canon_datasets:
        base_root = RES_PATH / "probe" / "l1" / d
        random_root = RES_PATH / "probe" / "l1_random" / d
        imp_by_data[d] = {}
        rnd_by_data[d] = {}
        for m in models:
            if show_important:
                imp_by_data[d][m] = _gather_results(base_root / m)
            if show_random:
                rnd_by_data[d][m] = _gather_random_results(random_root / m)

    # Prepare per-column legends (one time)
    col_leg_handles: List[List[Line2D]] = [[] for _ in range(ncols)]
    col_leg_labels: List[List[str]] = [[] for _ in range(ncols)]
    for col, (_fam_name, fam_models) in enumerate(families):
        for m in fam_models:
            if m not in models:
                continue
            color = COLOR_BY_MODEL.get(m, "black")
            col_leg_handles[col].append(Line2D([0], [0], color=color, lw=1.4, linestyle="-"))
            col_leg_labels[col].append(SHORT_LABEL.get(m, m))

    # Plot each row spec (dataset + metric) across columns
    for r, (d, metric_key, metric_label) in enumerate(row_specs):
        # Compute y-limits per row across all series that appear in this row
        series: List[Iterable[ResultPoint]] = []
        for _fam_name, fam_models in families:
            for m in fam_models:
                if m not in models:
                    continue
                if show_important and m in imp_by_data[d]:
                    series.append(imp_by_data[d][m][metric_key])
                if show_random and m in rnd_by_data[d]:
                    series.append(rnd_by_data[d][m][metric_key])
        ylim = _row_ylim(series)

        for c, (_fam_name, fam_models) in enumerate(families):
            ax = axes_2d[r, c]

            for m in fam_models:
                if m not in models:
                    continue
                color = COLOR_BY_MODEL.get(m, "black")
                if show_important and m in imp_by_data[d]:
                    _plot_series(ax, imp_by_data[d][m][metric_key], color=color, linestyle="-")
                if show_random and m in rnd_by_data[d]:
                    _plot_series(ax, rnd_by_data[d][m][metric_key], color=color, linestyle="--")

            is_bottom = (r == (nrows - 1))

            # Put dataset name in the ylabel (only left column) so it's always visible
            if c == 0 and show_dataset_name_in_ylabel:
                ylabel = f"{_dataset_display(d)}\n{metric_label}"
            else:
                ylabel = metric_label if c == 0 else metric_label  # unused if show_ylabel=False

            _format_axes(
                ax,
                show_ylabel=(c == 0),
                ylabel=ylabel if c == 0 else "",
                show_xlabel=is_bottom,
                show_xticklabels=is_bottom,
            )
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Add the per-column color legends once (attach to the bottom row axes)
    for col in range(ncols):
        if not col_leg_handles[col]:
            continue
        axes_2d[nrows - 1, col].legend(
            handles=col_leg_handles[col],
            labels=col_leg_labels[col],
            ncol=1,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.55),
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.8,
            handletextpad=0.5,
        )

    # Single (global) style legend at the top
    _add_top_style_legend(fig, y=1.02)

    # Layout: give a bit more room on the left for multi-line ylabels, and at bottom for legends
    fig.subplots_adjust(
        left=0.12,
        right=0.995,
        top=0.95,
        bottom=0.12,
        wspace=0.18,
        hspace=0.30,
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.10)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Important vs Random (small-multiples template)")
    parser.add_argument(
        "--data",
        default="blimp",
        help=(
            "Dataset name (matches results directory). "
            "Special modes: other_english (plots cola+syngym stacked), multi (plots scala,nl-blimp,itacola,rucola,jcola,sling stacked on Llama+Gemma)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names to include (matches subdirectories under results/probe/l1/<data>/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. Defaults to l1_{data}_imp_vs_random_grid.pdf (or a mode-specific name).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only_random", action="store_true", help="Plot only Random (dashed) curves.")
    mode.add_argument("--only_important", action="store_true", help="Plot only Important (solid) curves.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _canonical_data_key(args.data)

    show_imp = not args.only_random
    show_rnd = not args.only_important

    # Mode 1: other_english -> stack CoLA + SyntaxGym vertically, legend once
    if data == "other_english":
        output = args.output if args.output is not None else Path("l1_other_english_imp_vs_random_grid.pdf")
        plot_stacked_datasets(
            datasets=("cola", "syngym"),
            families=FAMILIES,
            models=args.models,
            output_path=output,
            show_important=show_imp,
            show_random=show_rnd,
            show_dataset_name_in_ylabel=True,
        )
        return

    # Mode 2: multi -> stack 6 datasets vertically, only Llama + Gemma families, legend once
    if data == "multi":
        output = args.output if args.output is not None else Path("l1_multi_imp_vs_random_grid.pdf")
        families_multi = (FAMILIES[1], FAMILIES[2])  # Llama, Gemma
        plot_stacked_datasets(
            datasets=("scala", "nl-blimp", "itacola", "rucola", "jcola", "sling"),
            families=families_multi,
            models=args.models,
            output_path=output,
            show_important=show_imp,
            show_random=show_rnd,
            show_dataset_name_in_ylabel=True,
        )
        return

    # Default: original single-dataset plot
    output = args.output if args.output is not None else Path(f"l1_{data}_imp_vs_random_grid.pdf")
    plot_new_template(
        data,
        args.models,
        output,
        show_important=show_imp,
        show_random=show_rnd,
    )


if __name__ == "__main__":
    main()
