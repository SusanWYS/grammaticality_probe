#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
DATA = {
    "blimp": "BLiMP",
    "cola": "CoLA",
    "syngym": "SyntaxGym",
}

model_maps = {
    "olmo2-7B": "OLMo-2-7B",
    "olmo3-7B": "OLMo-3-7B",
    "llama3-2-1B": "Llama-3.2-1B",
    "llama3-1-8B": "Llama-3.1-8B",
    "gemma2-2B": "Gemma-2-2B",
    "gemma2-9B": "Gemma-2-9B",
}


def _cap_first(s: str) -> str:
    """Capitalize the first letter only (preserve rest)."""
    return model_maps[s]


# ---------------------------------------------------------------------------
# ACL-ish matplotlib style (mirrors your reference script)
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

# Two series colors
PROBE_COLOR = "cadetblue"
LOGPROB_COLOR = "#d95f02"


def _is_valid(val: object) -> bool:
    try:
        return val is not None and not math.isnan(float(val))
    except Exception:
        return False


def _binomial_ci(p: float, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if not _is_valid(p) or n <= 0:
        return None
    se = math.sqrt(p * (1.0 - p) / float(n))
    low = max(0.0, p - z * se)
    high = min(1.0, p + z * se)
    return (low, high)


# ---------------------------------------------------------------------------
# DeLong CI for AUC
# ---------------------------------------------------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Midranks for ties (1-indexed ranks).
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]

    n = x_sorted.size
    ranks_sorted = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1.0  # 1-indexed
        ranks_sorted[i:j] = mid
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _fast_delong_auc_var(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float, int, int]:
    """
    DeLong variance estimate for ROC AUC (single set of scores).

    Returns:
        auc, var(auc), m (#pos), n (#neg)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.size != y_score.size:
        raise ValueError("y_true and y_score must be 1D arrays of the same length.")

    # Filter non-finite scores
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    uniq = np.unique(y_true)
    if uniq.size < 2 or not np.all(np.isin(uniq, [0, 1])):
        raise ValueError("y_true must be binary {0,1} and contain both classes.")

    pos = (y_true == 1)
    neg = (y_true == 0)
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("Need at least one positive and one negative example.")

    # positives first then negatives
    preds = np.concatenate([y_score[pos], y_score[neg]])

    # Midranks
    tx = _compute_midrank(preds[:m])
    ty = _compute_midrank(preds[m:])
    tz = _compute_midrank(preds)

    # AUC from ranks
    auc = (tz[:m].sum() / m - (m + 1.0) / 2.0) / n

    # DeLong components
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m

    # Variance pieces (guard ddof when size==1)
    sx = float(np.var(v01, ddof=1)) if m > 1 else 0.0
    sy = float(np.var(v10, ddof=1)) if n > 1 else 0.0
    var = sx / m + sy / n

    # Numerical guard
    if var < 0 and var > -1e-15:
        var = 0.0

    return float(auc), float(var), m, n


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _delong_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    z: float = 1.96,
    *,
    use_logit: bool = True,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    DeLong CI for ROC AUC.

    Returns:
        (auc, (low, high)) or (None, None) if invalid.
    """
    try:
        auc, var, _, _ = _fast_delong_auc_var(y_true, y_score)
    except Exception:
        return None, None

    if not (0.0 <= auc <= 1.0) or not math.isfinite(var):
        return None, None

    se = math.sqrt(max(0.0, var))

    if use_logit:
        # Work on logit scale to keep bounds in [0,1]
        eps = 1e-12
        a = min(1.0 - eps, max(eps, auc))
        logit = math.log(a / (1.0 - a))
        se_logit = se / (a * (1.0 - a))

        low = _sigmoid(logit - z * se_logit)
        high = _sigmoid(logit + z * se_logit)
        return auc, (max(0.0, low), min(1.0, high))

    low = max(0.0, auc - z * se)
    high = min(1.0, auc + z * se)
    return auc, (low, high)


def _paired_accuracy(
    meta: List[Tuple[int, bool]], scores: Iterable[float]
) -> Tuple[Optional[float], int]:
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
        if good > bad:
            correct += 1

    if total == 0:
        return None, 0
    return correct / total, total


def _collect_per_example(
    stats: Dict[str, Dict[str, object]],
    *,
    use_probe_scores: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, bool]]]:
    labels: List[int] = []
    scores: List[float] = []
    meta: List[Tuple[int, bool]] = []
    for key, entry in stats.items():
        if key == "overall_stats":
            continue
        good_bad = (
            (True, entry.get("good")),
            (False, entry.get("bad")),
        )
        for is_good, side in good_bad:
            if not side:
                continue
            score = side.get("probe_score") if use_probe_scores else side.get(
                "example_norm_prob"
            )
            if not _is_valid(score):
                continue
            labels.append(1 if is_good else 0)
            scores.append(float(score))
            meta.append((int(key), is_good))

    return np.asarray(labels), np.asarray(scores), meta


def _compute_metrics(
    stats: Dict[str, Dict[str, object]], *, use_probe_scores: bool
) -> Dict[str, object]:
    labels, scores, meta = _collect_per_example(stats, use_probe_scores=use_probe_scores)
    if int(labels.size) == 0:
        return {"acc": None, "acc_ci": None, "auc": None, "auc_ci": None}

    paired_acc, paired_total = _paired_accuracy(meta, scores)
    acc_ci = _binomial_ci(paired_acc, paired_total) if paired_acc is not None else None

    auc_val: Optional[float] = None
    auc_ci: Optional[Tuple[float, float]] = None
    if len(np.unique(labels)) > 1:
        auc_val, auc_ci = _delong_auc_ci(labels, scores)

    return {
        "acc": paired_acc,
        "acc_ci": acc_ci,
        "auc": auc_val,
        "auc_ci": auc_ci,
    }


def load_probe_and_inference_metrics(
    data_name: str,
    with_prob: bool = False,
    *,
    concat: bool = False,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """Return best probe metrics + norm-prob inference metrics with confidence intervals."""
    base_dir = RES_PATH / "probe"
    if (base_dir / "l2_prob" / data_name).exists() and with_prob:
        base_dir = base_dir / "l2_prob" / data_name
    else:
        base_dir = base_dir / "l2" / data_name

    probe_metrics: Dict[str, Dict[str, object]] = {}
    inference_metrics: Dict[str, Dict[str, object]] = {}

    for model in MODELS:
        model_dir = base_dir / model
        best_score = None
        best_stats: Optional[Dict[str, Dict[str, object]]] = None

        if model_dir.exists():
            for pkl_path in sorted(model_dir.glob("*.pkl")):
                if concat and "_concat" not in pkl_path.name:
                    continue
                if not concat and "_concat" in pkl_path.name:
                    continue

                with open(pkl_path, "rb") as fh:
                    stats = pickle.load(fh)

                overall = stats.get("overall_stats", {})
                val_auc = overall.get("auc")
                val_acc = overall.get("test_paired_acc")
                score = val_auc if _is_valid(val_auc) else val_acc
                if not _is_valid(score):
                    continue

                score = float(score)
                if best_score is None or score > best_score:
                    best_score = score
                    best_stats = stats

        if best_stats is None:
            probe_metrics[model] = {"acc": None, "auc": None, "acc_ci": None, "auc_ci": None}
            inference_metrics[model] = {"acc": None, "auc": None, "acc_ci": None, "auc_ci": None}
        else:
            probe_metrics[model] = _compute_metrics(best_stats, use_probe_scores=True)
            inference_metrics[model] = _compute_metrics(best_stats, use_probe_scores=False)

    return probe_metrics, inference_metrics


def _pretty_metric_label(metric: str) -> str:
    if metric == "acc":
        return "ACC"
    if metric == "auc":
        return "AUC"
    return metric


def _plot_metric(
    ax,
    metric: str,
    probe: Dict[str, Dict[str, object]],
    inference: Dict[str, Dict[str, object]],
) -> None:
    xs = list(range(len(MODELS)))
    probe_xs, probe_ys = [], []
    logp_xs, logp_ys = [], []
    probe_errs: List[Tuple[float, float]] = []
    logp_errs: List[Tuple[float, float]] = []

    for idx, model in enumerate(MODELS):
        probe_info = probe.get(model, {})
        logp_info = inference.get(model, {})

        probe_val = probe_info.get(metric)
        logp_val = logp_info.get(metric)

        probe_ci = probe_info.get(f"{metric}_ci")
        logp_ci = logp_info.get(f"{metric}_ci")

        if _is_valid(probe_val):
            v = float(probe_val)
            probe_xs.append(idx)
            probe_ys.append(v)
            if probe_ci and _is_valid(probe_ci[0]) and _is_valid(probe_ci[1]):
                lo, hi = float(probe_ci[0]), float(probe_ci[1])
                probe_errs.append((max(0.0, v - lo), max(0.0, hi - v)))
            else:
                probe_errs.append((0.0, 0.0))

        if _is_valid(logp_val):
            v = float(logp_val)
            logp_xs.append(idx)
            logp_ys.append(v)
            if logp_ci and _is_valid(logp_ci[0]) and _is_valid(logp_ci[1]):
                lo, hi = float(logp_ci[0]), float(logp_ci[1])
                logp_errs.append((max(0.0, v - lo), max(0.0, hi - v)))
            else:
                logp_errs.append((0.0, 0.0))

    # Hollow markers for AUC; solid for ACC.
    hollow = (metric == "auc")

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    # ----- Probe points -----
    if probe_xs:
        ax.scatter(
            probe_xs,
            probe_ys,
            s=36,
            marker="D",
            facecolors="white" if hollow else PROBE_COLOR,
            edgecolors=PROBE_COLOR,
            linewidths=1,
            label="Probe Score",
            zorder=3,
            clip_on=False,
        )
        probe_yerr = np.array(probe_errs).T if probe_errs else None
        if probe_yerr is not None and probe_yerr.any():
            ax.errorbar(
                probe_xs,
                probe_ys,
                yerr=probe_yerr,
                fmt="none",
                ecolor=PROBE_COLOR,
                elinewidth=0.8,
                capsize=2.5,
                alpha=0.7,
                zorder=2,
                clip_on=False,
            )

    # ----- Logprob points -----
    if logp_xs:
        ax.scatter(
            logp_xs,
            logp_ys,
            s=36,
            marker="o",
            facecolors="white" if hollow else LOGPROB_COLOR,
            edgecolors=LOGPROB_COLOR,
            linewidths=1,
            label="LM Logprob",
            zorder=3,
            clip_on=False,
        )

        logp_yerr = np.array(logp_errs).T if logp_errs else None
        if logp_yerr is not None and logp_yerr.any():
            ax.errorbar(
                logp_xs,
                logp_ys,
                yerr=logp_yerr,
                fmt="none",
                ecolor=LOGPROB_COLOR,
                elinewidth=0.8,
                capsize=2.5,
                alpha=0.7,
                zorder=2,
                clip_on=False,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([_cap_first(m) for m in MODELS], rotation=60, ha="right")
    ax.set_ylabel(_pretty_metric_label(metric))

    # Extra padding so points don't sit on the frame
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.margins(x=0.06, y=0.10)


def _global_ylim_from_panels(
    panels: List[Tuple[str, Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]],
    *,
    pad_frac: float = 0.03,
) -> Tuple[float, float]:
    """
    Compute a shared (ymin, ymax) across all panels, including CI bounds.
    Clamps to [0, 1] since ACC/AUC live there.
    panels: list of (metric, probe_metrics, logprob_metrics)
    """
    ymin = math.inf
    ymax = -math.inf

    for metric, probe, logp in panels:
        for model in MODELS:
            for info in (probe.get(model, {}), logp.get(model, {})):
                val = info.get(metric)
                if not _is_valid(val):
                    continue
                v = float(val)

                lo = v
                hi = v
                ci = info.get(f"{metric}_ci")
                if ci:
                    if _is_valid(ci[0]):
                        lo = min(lo, float(ci[0]))
                    if _is_valid(ci[1]):
                        hi = max(hi, float(ci[1]))

                ymin = min(ymin, lo)
                ymax = max(ymax, hi)

    if ymin == math.inf or ymax == -math.inf:
        return (0.0, 1.0)

    span = ymax - ymin
    if span <= 1e-12:
        span = 0.05
    pad = pad_frac * span
    ymin -= pad
    ymax += pad

    ymin = max(0.0, ymin)
    ymax = min(1.0, ymax)

    if ymax <= ymin:
        ymax = min(1.0, ymin + 0.05)

    return (ymin, ymax)


def plot_probe_vs_logprob_acl_column(output_path: Path, *, concat: bool = False) -> None:
    # Only these datasets
    blimp_probe, blimp_logp = load_probe_and_inference_metrics("blimp", concat=concat)
    cola_probe, cola_logp = load_probe_and_inference_metrics("cola", concat=concat)
    syngym_probe, syngym_logp = load_probe_and_inference_metrics("syngym", concat=concat)

    # Shared y-limits across all four panels (include CI bounds)
    panels = [
        ("acc", blimp_probe, blimp_logp),
        ("auc", blimp_probe, blimp_logp),
        ("auc", cola_probe, cola_logp),
        ("auc", syngym_probe, syngym_logp),
    ]
    shared_ymin, shared_ymax = _global_ylim_from_panels(panels)

    # One-column ACL figure: ~3.2–3.4 inches wide; compact height
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.15), sharex="col")

    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    # Row 1: BLiMP (ACC, AUC)
    _plot_metric(ax00, "acc", blimp_probe, blimp_logp)
    ax00.set_title("")

    _plot_metric(ax01, "auc", blimp_probe, blimp_logp)
    ax01.set_title("")

    # Row 2: CoLA AUC, SyntaxGym AUC
    _plot_metric(ax10, "auc", cola_probe, cola_logp)
    ax10.set_title("CoLA")

    _plot_metric(ax11, "auc", syngym_probe, syngym_logp)
    ax11.set_title("SyntaxGym")

    # Combined title "BLiMP" centered above the top row at the same height as titles
    pos0 = ax00.get_position()
    pos1 = ax01.get_position()
    x_center = 0.5 * (pos0.x0 + pos1.x1)

    # Convert axes.titlepad (points) to figure coordinates so the vertical placement matches title height
    title_pad_pts = float(plt.rcParams.get("axes.titlepad", 6.0))
    y_offset = (title_pad_pts / 72.0) / float(fig.get_figheight())
    y_title = pos0.y1 + y_offset

    fig.text(
        x_center,
        y_title,
        "BLiMP",
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
    )

    # Apply shared y-range to all panels
    for ax in (ax00, ax01, ax10, ax11):
        ax.set_ylim(shared_ymin, shared_ymax)

    # Hide top-row x tick labels to reduce clutter
    for ax in (ax00, ax01):
        ax.tick_params(axis="x", labelbottom=False)

    # -----------------------------------------------------------------------
    # Legend encodes BOTH:
    #  - Series: Probe vs Logprob (shape/color)
    #  - Metric style: ACC solid vs AUC hollow
    # In ONE ROW
    # -----------------------------------------------------------------------
    series_probe = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="None",
        markerfacecolor=PROBE_COLOR,
        markeredgecolor=PROBE_COLOR,
        markeredgewidth=1,
        markersize=6,
        label="Probe Score",
    )
    series_logp = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=LOGPROB_COLOR,
        markeredgecolor=LOGPROB_COLOR,
        markeredgewidth=1,
        markersize=6,
        label="LM Logprob",
    )
    style_acc = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="dimgrey",
        markeredgecolor="dimgrey",
        markeredgewidth=1,
        markersize=6,
        label="ACC",
    )
    style_auc = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="dimgrey",
        markeredgewidth=1,
        markersize=6,
        label="AUC",
    )

    fig.legend(
        [series_probe, series_logp, style_acc, style_auc],
        ["Probe Score", "LM Logprob", "ACC", "AUC"],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.45, 1.05),
        borderaxespad=0.0,
        columnspacing=0.45,
        handletextpad=0.35,
        labelspacing=0.2,
        borderpad=0.1,
    )

    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.14, wspace=0.35, hspace=0.35)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot L2 probe vs. logprob metrics (ACL one-column 2x2)"
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Plot results for concatenated L2 probes saved with --concat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to a name derived from --concat flag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_name = (
        "l2_probe_vs_logprob_acl_2x2_concat.pdf"
        if args.concat
        else "l2_probe_vs_logprob_acl_2x2.pdf"
    )
    output_path = args.output or Path(default_name)
    plot_probe_vs_logprob_acl_column(output_path, concat=args.concat)


if __name__ == "__main__":
    main()
