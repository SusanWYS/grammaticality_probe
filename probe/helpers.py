from __future__ import annotations

import gc
import math
import os
import pickle
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch
from numpy.lib.format import open_memmap
from sklearn.metrics import roc_auc_score
from snapml import LogisticRegression

# ---------------------------------------------------------------------------
# Module Configuration
# ---------------------------------------------------------------------------

torch.manual_seed(0)
random.seed(0)

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
RES_PATH = PROJECT_ROOT / "results"

LAST_TOK_HS = "last_tok_hs"

_SHARD_CACHE: Dict[Path, Any] = {}

_EXAMPLE_FIELD_NAMES: Dict[str, str] = {
    "good": "good_hs",
    "bad": "bad_hs",
}

_PROB_KEYS: Dict[str, List[str]] = {
    "good": ["norm_good_prob"],
    "bad": ["norm_bad_prob"],
}


def _get_gpu_device_ids() -> List[int]:
    """Return available CUDA device ordinals for the current process.
    """
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0
    return list(range(device_count))


GPU_DEVICE_IDS: List[int] = _get_gpu_device_ids()


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------


class FormatDataResult(TypedDict):
    """Result dictionary from format_data function."""

    X_train: Optional[np.ndarray]
    X_test: np.ndarray
    y_train: Optional[np.ndarray]
    y_test: np.ndarray
    meta_train: Optional[List[Tuple[int, bool]]]
    meta_test: Optional[List[Tuple[int, bool]]]
    train_stats: Optional[List[Optional[np.ndarray]]]
    test_stats: Optional[List[Optional[np.ndarray]]]
    dim: int
    split: bool


# ---------------------------------------------------------------------------
# BLiMP Category Mapping
# ---------------------------------------------------------------------------

blimp_map: Dict[str, List[str]] = {
    "ANAPHOR AGREEMENT": [
        "anaphor_gender_agreement",
        "anaphor_number_agreement",
    ],
    "ARGUMENT STRUCTURE": [
        "animate_subject_passive",
        "animate_subject_trans",
        "causative",
        "drop_argument",
        "inchoative",
        "intransitive",
        "passive_1",
        "passive_2",
        "transitive",
    ],
    "BINDING": [
        "principle_A_c_command",
        "principle_A_case_1",
        "principle_A_case_2",
        "principle_A_domain_1",
        "principle_A_domain_2",
        "principle_A_domain_3",
        "principle_A_reconstruction",
    ],
    "CONTROL/ RAISING": [
        "existential_there_object_raising",
        "existential_there_subject_raising",
        "expletive_it_object_raising",
        "tough_vs_raising_1",
        "tough_vs_raising_2",
    ],
    "DETER-MINER-NOUN AGR.": [
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2",
    ],
    "ELLIPSIS": [
        "ellipsis_n_bar_1",
        "ellipsis_n_bar_2",
    ],
    "FILLER GAP": [
        "wh_questions_object_gap",
        "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance",
        "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ],
    "IRREGULAR FORMS": [
        "irregular_past_participle_adjectives",
        "irregular_past_participle_verbs",
    ],
    "ISLAND EFFECTS": [
        "adjunct_island",
        "complex_NP_island",
        "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "left_branch_island_echo_question",
        "left_branch_island_simple_question",
        "sentential_subject_island",
        "wh_island",
    ],
    "NPI LICENSING": [
        "matrix_question_npi_licensor_present",
        "npi_present_1",
        "npi_present_2",
        "only_npi_licensor_present",
        "only_npi_scope",
        "sentential_negation_npi_licensor_present",
        "sentential_negation_npi_scope",
    ],
    "QUANTIFIERS": [
        "existential_there_quantifiers_1",
        "existential_there_quantifiers_2",
        "superlative_quantifiers_1",
        "superlative_quantifiers_2",
    ],
    "SUBJECT-VERB AGR.": [
        "distractor_agreement_relational_noun",
        "distractor_agreement_relative_clause",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
    ],
}


# ---------------------------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------------------------


def _make_out_dir(*, data: str, model: str, add_prob: bool) -> Path:
    """Create and return the output directory for probe results.
    """
    probe_subdir = "l2_prob" if add_prob else "l2"
    out_dir = RES_PATH / "probe" / probe_subdir / data / model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _probe_dir(base: str, add_prob: bool, random_select: bool = False) -> Path:
    """Construct the probe results directory path.
    """
    prob_suffix = "_prob" if add_prob else ""
    random_suffix = "_random" if random_select else ""
    return RES_PATH / "probe" / f"{base}{prob_suffix}{random_suffix}"


def _inference_shard_dir(
    data: str,
    model_name: str,
    perturb: str = "",
    incremental: bool = False,
) -> Path:
    """Build the path to inference shards.
    """
    perturb_prefix = "" if perturb in ("all", "") else f"{perturb}_"
    run_id = f"{perturb_prefix}-1"
    if incremental:
        run_id += "_incremental"
    return RES_PATH / "inference" / data / model_name / f"{run_id}_all" / "all"


def _incremental_shard_dir(*, data: str, model_name: str) -> Path:
    """Build the path to incremental inference shards.
    """
    return _inference_shard_dir(data.lower(), model_name, perturb="", incremental=True)


def _memmap_dir(
    dataset_name: str,
    model_name: str,
    *,
    add_prob: bool = False,
    layer_idx: Optional[int] = None,
) -> Path:
    """Construct the directory path for memory-mapped arrays.
    """
    base = RES_PATH / "probe" / "_memmaps" / dataset_name / model_name / "-1"
    if add_prob:
        base = base / "prob"
    if layer_idx is not None:
        base = base / f"layer{layer_idx}"
    return base


# ---------------------------------------------------------------------------
# Shard Loading Utilities
# ---------------------------------------------------------------------------


def _list_all_shard_pickles(path: Path) -> List[Path]:
    """Return a sorted list of pickle files directly under a path.
    """
    try:
        if not path.exists():
            return []
    except OSError:
        return []
    pickle_files = [f for f in path.glob("*.pkl") if f.is_file()]
    pickle_files.sort(key=lambda p: p.name)
    return pickle_files


def _load_shard_block(pkl_path: Path, use_cache: bool) -> Any:
    """Load a shard pickle from disk with optional caching.
    """
    if use_cache and pkl_path in _SHARD_CACHE:
        return _SHARD_CACHE[pkl_path]

    with open(pkl_path, "rb") as file_handle:
        block = pickle.load(file_handle)

    if use_cache:
        _SHARD_CACHE[pkl_path] = block

    return block


def _yield_examples_in_order(obj: Any) -> Iterator[Dict[str, Any]]:
    """Yield examples from a nested shard structure in sorted order.
    """
    if isinstance(obj, dict) and "good_hs" in obj:
        yield obj
        return

    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=str):
            yield from _yield_examples_in_order(obj[key])


def _stream_all_examples(
    shard_dir: Path,
    category_name: Optional[str] = None,
    use_cache: bool = True,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Stream all examples from shard files with their indices.
    """
    example_idx = 0

    if category_name is not None:
        category_path = shard_dir / f"{category_name}.pkl"
        block = _load_shard_block(category_path, use_cache=use_cache)
        for example in _yield_examples_in_order(block):
            yield example_idx, example
            example_idx += 1
        del block
        gc.collect()
        return

    for pkl_path in _list_all_shard_pickles(shard_dir):
        block = _load_shard_block(pkl_path, use_cache=use_cache)
        for example in _yield_examples_in_order(block):
            yield example_idx, example
            example_idx += 1
        del block
        gc.collect()


def _count_examples_in_shards(
    shard_dir: Path,
    category_name: Optional[str] = None,
    use_cache: bool = True,
) -> int:
    """Count total examples across all shards in a directory.
    """
    if category_name is not None:
        category_path = shard_dir / f"{category_name}.pkl"
        block = _load_shard_block(category_path, use_cache=use_cache)
        total = sum(1 for _ in _yield_examples_in_order(block))
        del block
        gc.collect()
        return total

    total = 0
    for pkl_path in _list_all_shard_pickles(shard_dir):
        block = _load_shard_block(pkl_path, use_cache=use_cache)
        total += sum(1 for _ in _yield_examples_in_order(block))
        del block
        gc.collect()
    return total


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def _extract_prob(example: Dict[str, Any], is_good: bool) -> Optional[float]:
    """Extract normalized probability from an example.
    """
    key = "norm_good_prob" if is_good else "norm_bad_prob"
    if key not in example:
        return None
    val = example[key]
    if val is None:
        return None
    return float(val)


def _to_vec(
    example: Dict[str, Any],
    dataset_name: str,
    neuron_indices: Optional[np.ndarray],
    which: str,
    add_prob: bool = False,
    layer_idx: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Extract a feature vector from an example's hidden states.
    """
    if which not in ("good", "bad"):
        return None

    hidden_state_key = _EXAMPLE_FIELD_NAMES[which]
    raw_hidden_states = example.get(hidden_state_key)
    if raw_hidden_states is None:
        return None

    # Convert to numpy array
    if isinstance(raw_hidden_states, torch.Tensor):
        tensor = raw_hidden_states
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        feature_vector = tensor.detach().contiguous().numpy()
    elif isinstance(raw_hidden_states, np.ndarray):
        feature_vector = raw_hidden_states.astype(np.float32, copy=False)
    else:
        return None

    # Select specific layer if requested
    if layer_idx is not None and feature_vector.ndim >= 2:
        if not (0 <= layer_idx < feature_vector.shape[0]):
            raise ValueError(
                f"Requested layer_idx {layer_idx} out of range for "
                f"vector with shape {feature_vector.shape}"
            )
        feature_vector = feature_vector[layer_idx]

    # Flatten and ensure contiguous
    feature_vector = np.ascontiguousarray(feature_vector.reshape(-1))

    # Handle non-finite values
    if not np.all(np.isfinite(feature_vector)):
        feature_vector = np.nan_to_num(feature_vector, copy=False)

    # Select specific neurons if requested
    if neuron_indices is not None:
        feature_vector = feature_vector[neuron_indices]

    # Append probability features if requested
    if add_prob:
        prob_keys = _PROB_KEYS["good"] if which in ("good", "last_tok") else _PROB_KEYS["bad"]
        prob_values: List[float] = []
        for prob_key in prob_keys:
            raw_val = example.get(prob_key, 0.0)
            if isinstance(raw_val, torch.Tensor):
                scalar = float(raw_val.detach().cpu().reshape(-1)[0])
            elif isinstance(raw_val, np.ndarray):
                scalar = float(raw_val.reshape(-1)[0])
            else:
                scalar = float(raw_val)
            prob_values.append(scalar)
        if prob_values:
            prob_array = np.asarray(prob_values, dtype=np.float32)
            feature_vector = np.concatenate([feature_vector, prob_array], axis=0)

    return feature_vector


def _feature_dim_from_example(
    example: Dict[str, Any],
    dataset_name: str,
    neuron_indices: Optional[np.ndarray],
    add_prob_features: bool,
    layer_idx: Optional[int] = None,
) -> int:
    """Determine feature dimensionality from a sample example.
    """
    vec = _to_vec(
        example, dataset_name, neuron_indices,
        which="good", add_prob=add_prob_features, layer_idx=layer_idx
    )
    if vec is None:
        vec = _to_vec(
            example, dataset_name, neuron_indices,
            which="bad", add_prob=add_prob_features, layer_idx=layer_idx
        )
    if vec is None:
        raise RuntimeError("Could not extract any vector from example.")
    return int(vec.shape[0])


# ---------------------------------------------------------------------------
# Normalization Utilities
# ---------------------------------------------------------------------------


def _compute_norm_stats(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalization statistics from feature matrix.
    """
    mean_vector = np.mean(features, axis=0)
    std_vector = np.std(features, axis=0)
    safe_std_vector = std_vector.copy()
    safe_std_vector[safe_std_vector == 0] = 1.0
    return mean_vector, std_vector, safe_std_vector


def _apply_norm_from_stats(
    features: np.ndarray,
    mean_vector: Optional[np.ndarray],
    safe_std_vector: Optional[np.ndarray],
) -> np.ndarray:
    """Apply z-score normalization using precomputed statistics.
    """
    if mean_vector is None or safe_std_vector is None:
        return features
    return (features - mean_vector) / safe_std_vector


def _copy_or_normalize_row(
    destination_row: np.ndarray,
    source_vector: np.ndarray,
    mean_vector: Optional[np.ndarray],
    safe_std_vector: Optional[np.ndarray],
) -> None:
    """Copy source vector to destination, optionally normalizing in-place.
    """
    destination_row[...] = source_vector
    if mean_vector is not None and safe_std_vector is not None:
        destination_row -= mean_vector
        destination_row /= safe_std_vector


def _gather_train_stats(
    shard_dir: Path,
    train_example_ids: Optional[set[int]],
    dataset_name: str,
    neuron_indices: Optional[np.ndarray],
    add_prob_features: bool,
    category_name: Optional[str],
    layer_idx: Optional[int],
    use_cache: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute normalization statistics from training examples.
    """
    sum_vector: Optional[np.ndarray] = None
    sum_sq_vector: Optional[np.ndarray] = None
    row_count = 0

    for example_idx, example in _stream_all_examples(
        shard_dir, category_name, use_cache=use_cache
    ):
        if train_example_ids is not None and example_idx not in train_example_ids:
            continue

        for which in ("bad", "good"):
            vec = _to_vec(
                example, dataset_name, neuron_indices,
                which=which, add_prob=add_prob_features, layer_idx=layer_idx
            )
            if vec is None:
                continue
            if sum_vector is None:
                sum_vector = np.zeros_like(vec)
                sum_sq_vector = np.zeros_like(vec)
            sum_vector += vec
            sum_sq_vector += vec * vec
            row_count += 1

    if row_count == 0 or sum_vector is None or sum_sq_vector is None:
        return None, None, None

    mean_vector = sum_vector / float(row_count)
    variance_vector = sum_sq_vector / float(row_count) - mean_vector * mean_vector
    std_vector = np.sqrt(np.maximum(variance_vector, 1e-12))

    safe_std_vector = std_vector.copy()
    safe_std_vector[safe_std_vector == 0] = 1.0

    return mean_vector, std_vector, safe_std_vector


# ---------------------------------------------------------------------------
# Data Splitting Utilities
# ---------------------------------------------------------------------------


def generate_inds(
    data_len: int,
    seed: int = 0,
    train_ratio: float = 0.8,
) -> Dict[str, np.ndarray]:
    """Generate train/test split indices.
    """
    random.seed(seed)
    indices = list(range(data_len))
    random.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    return {
        "train": np.array(indices[:split_point]),
        "test": np.array(indices[split_point:]),
    }


def _route_destination(
    example_idx: int,
    split: bool,
    kept_train_ids: set[int],
    final_test_ids: set[int],
    wanted: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Determine which split an example belongs to.
    """
    if not split:
        return "test", None

    if example_idx in kept_train_ids:
        return "train", None
    if example_idx in final_test_ids:
        return "test", None
    return "skip", None


# ---------------------------------------------------------------------------
# Paired Accuracy Computation
# ---------------------------------------------------------------------------


def _compute_paired_acc_from_scores(
    metadata: List[Tuple[int, bool]],
    positive_scores: np.ndarray,
) -> float:
    """Compute paired accuracy from prediction scores.
    """
    pairs: Dict[int, List[Optional[float]]] = {}

    for (example_idx, is_good), score in zip(metadata, positive_scores):
        slot = pairs.setdefault(example_idx, [None, None])  # [bad, good]
        slot[1 if is_good else 0] = float(score)

    total_pairs = 0
    correct_pairs = 0
    for bad_good in pairs.values():
        bad_score, good_score = bad_good
        if bad_score is None or good_score is None:
            continue
        total_pairs += 1
        if good_score > bad_score:
            correct_pairs += 1

    return (correct_pairs / total_pairs) if total_pairs > 0 else float("nan")


# ---------------------------------------------------------------------------
# Classifier Probe Evaluation
# ---------------------------------------------------------------------------


def _evaluate_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: List[Tuple[int, bool]],
    weights: np.ndarray,
    intercept: np.ndarray,
    neuron_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a trained classifier on test data.
    """
    if neuron_indices is not None:
        features = features[:, neuron_indices]

    logits = features @ weights.T + intercept
    probabilities = (1.0 / (1.0 + np.exp(-logits))).squeeze(1)
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float((predictions == labels).mean())
    auc = roc_auc_score(labels, probabilities) if labels.size > 0 else float("nan")
    paired_accuracy = _compute_paired_acc_from_scores(metadata, probabilities)

    metrics = {"acc": accuracy, "auc": auc, "test_paired_acc": paired_accuracy}
    return metrics, probabilities, logits.squeeze(1)


def evaluate_probe_selection(
    *,
    selection: Dict[str, Any],
    model: str,
    test: str,
    start_exp: int,
    end_exp: int,
    target_ratio: Optional[float],
    add_prob: bool,
    suffix: str,
    out_dir: Path,
    data_cache: Dict[
        Optional[int], Tuple[np.ndarray, np.ndarray, List[Tuple[int, bool]]]
    ],
    example_lookup: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a trained probe on a test dataset.
    """
    layer_idx = selection["layer_idx"]
    weights = np.asarray(selection["weights"])
    intercept = np.asarray(selection["intercept"])
    neuron_indices = selection.get("non_zero_neurons")

    print(f"Testing layer {layer_idx} for model {model} on test dataset '{test}'.")

    # Load test data if not cached
    if layer_idx not in data_cache:
        effective_layer = None if layer_idx is None or layer_idx == "all" else layer_idx
        formatted = format_data(
            data=test,
            model_name=model,
            split=False,
            layer_idx=effective_layer,
            add_prob=add_prob,
        )
        data_cache[layer_idx] = (
            formatted["X_test"],
            formatted["y_test"],
            formatted["meta_test"] or [],
        )

    test_features, test_labels, test_metadata = data_cache[layer_idx]

    metrics, probs, logits = _evaluate_classifier(
        test_features, test_labels, test_metadata,
        weights, intercept, neuron_indices=neuron_indices
    )

    print(
        f"Test results -> acc: {metrics['acc']:.4f}, auc: {metrics['auc']:.4f}, "
        f"paired_acc: {metrics['test_paired_acc']:.4f}"
    )

    # Build per-example results
    results: Dict[str, Any] = {"overall_stats": metrics}
    for (example_idx, is_good), prob, logit in zip(test_metadata, probs, logits):
        key = str(example_idx)
        entry = results.setdefault(key, {"good": None, "bad": None})
        example = example_lookup.get(int(example_idx), {})
        norm_prob_val = _extract_prob(example, is_good)
        entry_data = {
            "example_norm_prob": norm_prob_val if norm_prob_val is not None else float(prob),
            "probe_score": float(logit),
        }
        entry["good" if is_good else "bad"] = entry_data

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_suffix = f"layer{layer_idx}" if layer_idx is not None else "layerN"
    target_suffix = "" if target_ratio is None else f"_{target_ratio}"
    out_path = out_dir / f"-1_{layer_suffix}_{suffix}_test_{start_exp}_to_{end_exp}{target_suffix}.pkl"

    with open(out_path, "wb+") as file_handle:
        pickle.dump(results, file_handle)
    print(f"Saved test results to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Regression Probe Training and Evaluation
# ---------------------------------------------------------------------------


def _fit_ridge_regression_snapml(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    exp: float,
    device_ids: List[int],
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Fit Ridge regression using Snap ML on GPU.
    """
    if exp <= 0:
        raise ValueError(f"exp must be positive for ridge regression, got {exp}.")

    from snapml import LinearRegression as SnapRidge

    features = np.ascontiguousarray(features, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)

    model = SnapRidge(
        regularizer=float(exp),
        fit_intercept=True,
        penalty="l2",
        use_gpu=True,
        device_ids=list(device_ids) if device_ids else [],
        verbose=0,
    )
    model.fit(features, targets)

    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", 0.0)

    if coef is None:
        raise RuntimeError("SnapML Ridge did not expose coef_ after fitting.")

    weights = np.asarray(coef, dtype=np.float32).reshape(-1)
    if isinstance(intercept, (list, tuple, np.ndarray)):
        bias = float(np.asarray(intercept).reshape(-1)[0])
    else:
        bias = float(intercept)

    extra_info = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device_count": int(torch.cuda.device_count()),
    }

    return weights, bias, extra_info


def _evaluate_regression_from_weights(
    data: Any,
    weights: np.ndarray,
    bias: float,
    *,
    y: Optional[np.ndarray] = None,
    split: str = "test",
) -> Dict[str, float]:
    """Evaluate regression model using provided weights.
    """
    if y is None:
        if split == "train":
            features = data.get("X_train")
            targets = data.get("y_train")
        elif split == "test":
            features = data.get("X_test")
            targets = data.get("y_test")
        else:
            raise ValueError(f"Unknown split '{split}'; expected 'train' or 'test'.")
    else:
        features = np.asarray(data)
        targets = np.asarray(y)

    if features is None or targets is None or features.size == 0 or targets.size == 0:
        return {"mse": float("nan"), "mae": float("nan"), "corr": float("nan")}

    predictions = features @ weights + bias
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))

    if targets.size > 1:
        try:
            corr = float(np.corrcoef(predictions, targets)[0, 1])
        except Exception:
            corr = float("nan")
    else:
        corr = float("nan")

    return {"mse": mse, "mae": mae, "corr": corr, "r2": corr ** 2}


def _tune_exp_ridge(
    *,
    dataset: Any,
    start_exp: int,
    end_exp: int,
    device_ids: List[int],
) -> Dict[str, Any]:
    """Tune ridge regression exp via grid search on dev set.
    """
    exps = [float(2.0 ** k) for k in range(int(start_exp), int(end_exp) + 1)]

    # Unpack dataset
    if isinstance(dataset, dict):
        X_train = dataset.get("X_train")
        X_dev = dataset.get("X_test")
        y_train = dataset.get("y_train")
        y_dev = dataset.get("y_test")
    elif isinstance(dataset, tuple) and len(dataset) >= 4:
        X_train, y_train, X_dev, y_dev = dataset[:4]
    else:
        raise ValueError("Unsupported dataset type for ridge tuning.")

    if X_train is None or y_train is None or X_dev is None or y_dev is None:
        raise ValueError("Ridge tuning requires train/dev splits with features and labels.")

    sweep_results: List[Dict[str, Any]] = []
    best_exp: Optional[float] = None
    best_dev_mse = float("inf")
    best_weights: Optional[np.ndarray] = None
    best_bias = 0.0
    best_extra: Dict[str, Any] = {}

    for exp in exps:
        print(f"started training {exp}")
        weights, bias, extra = _fit_ridge_regression_snapml(
            X_train, y_train, exp=exp, device_ids=device_ids
        )
        dev_metrics = _evaluate_regression_from_weights(X_dev, weights, bias, y=y_dev)
        sweep_results.append({"exp": float(exp), "dev_metrics": dev_metrics})

        dev_mse = float(dev_metrics.get("mse", float("inf")))
        if np.isfinite(dev_mse) and dev_mse < best_dev_mse:
            best_dev_mse = dev_mse
            best_exp = float(exp)
            best_weights = weights
            best_bias = bias
            best_extra = extra

    if best_exp is None or best_weights is None:
        raise RuntimeError("exp sweep failed to produce a valid dev MSE.")

    train_metrics = _evaluate_regression_from_weights(X_train, best_weights, best_bias, y=y_train)
    dev_metrics_final = _evaluate_regression_from_weights(X_dev, best_weights, best_bias, y=y_dev)

    return {
        "weights": best_weights,
        "bias": float(best_bias),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics_final,
        "n_train": int(y_train.size),
        "n_dev": int(y_dev.size),
        "dim": int(best_weights.size),
        "backend": "snapml_ridge",
        "device_ids": list(device_ids),
        "exp_sweep": {
            "start_exp": int(start_exp),
            "end_exp": int(end_exp),
            "exps": [float(a) for a in exps],
            "criterion": "dev_mse",
            "best_exp": float(best_exp),
            "best_dev_mse": float(best_dev_mse),
            "sweep_results": sweep_results,
        },
        "extra": best_extra,
    }


# ---------------------------------------------------------------------------
# Surprisal Probe Training and Evaluation
# ---------------------------------------------------------------------------


def train_surprisal_probe(
    *,
    data: str,
    model_name: str,
    train_ratio: float,
    start_exp: int,
    end_exp: int,
    device_ids: List[int],
    incremental: bool,
) -> Tuple[Dict[str, Any], int]:
    """Train a surprisal prediction probe.
    Selects the best layer using saved L2 probe results, then trains
    a ridge regression model to predict token surprisal.
    """
    chosen_layer = _select_best_l2_model(
        data=data,
        model=model_name,
        start_exp=start_exp,
        end_exp=end_exp,
        add_prob=False,
    )["layer_idx"]

    X_train, y_train, X_dev, y_dev, norm_stats = prepare_surprisal_data(
        data=data,
        model_name=model_name,
        layer_idx=int(chosen_layer),
        incremental=incremental,
        test=False,
        train_ratio=train_ratio,
    )

    results = _tune_exp_ridge(
        dataset=(X_train, y_train, X_dev, y_dev, norm_stats),
        start_exp=start_exp,
        end_exp=end_exp,
        device_ids=device_ids,
    )
    results["norm_stats"] = norm_stats
    results["layer_idx"] = int(chosen_layer)

    return results, int(chosen_layer)


def evaluate_surprisal_probe(
    *,
    probe_path: Path,
    eval_data: str,
    model_name: str,
    incremental: bool,
) -> Dict[str, Any]:
    """Evaluate a trained surprisal probe on a dataset.
    """
    with open(probe_path, "rb") as file_handle:
        blob = pickle.load(file_handle)

    trained_layer = blob.get("layer_idx")
    if trained_layer is None:
        raise ValueError(f"Probe at {probe_path} is missing layer_idx.")
    layer_idx = int(trained_layer)

    results_blob = blob.get("results", {}) or {}
    weights = np.asarray(results_blob.get("weights"), dtype=np.float32).reshape(-1)
    bias = float(results_blob.get("bias"))
    norm_stats = results_blob.get("norm_stats")

    # Handle pooled evaluation
    datasets = ["blimp", "cola", "syntaxgym"] if eval_data.lower() == "all" else [eval_data]

    feature_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []

    for dataset in datasets:
        X_ds, y_ds = prepare_surprisal_data(
            data=dataset,
            model_name=model_name,
            stats=norm_stats,
            layer_idx=layer_idx,
            incremental=incremental,
            test=True,
        )
        feature_parts.append(X_ds)
        target_parts.append(y_ds)

    X_eval = np.concatenate(feature_parts, axis=0) if len(feature_parts) > 1 else feature_parts[0]
    y_eval = np.concatenate(target_parts, axis=0) if len(target_parts) > 1 else target_parts[0]

    eval_metrics = _evaluate_regression_from_weights(X_eval, weights, bias, y=y_eval)

    return {
        "weights": weights,
        "bias": bias,
        "eval_metrics": eval_metrics,
        "n_eval": int(y_eval.size),
        "dim": int(weights.size),
        "probe_path": str(probe_path),
        "eval_data": eval_data,
        "eval_datasets": datasets,
        "layer_idx": layer_idx,
        "backend": results_blob.get("backend"),
        "exp": (results_blob.get("exp_sweep", {}) or {}).get("best_exp"),
    }


def prepare_surprisal_data(
    *,
    data: str,
    model_name: str,
    layer_idx: int,
    incremental: bool,
    test: bool,
    seed: int = 0,
    train_ratio: float = 0.8,
    stats: Optional[Dict[str, Optional[np.ndarray]]] = None,
    apply_norm: bool = True,
    perturb: str = "",
    cat: Optional[str] = None,
    use_shard_cache: bool = True,
) -> Any:
    """Prepare data for surprisal probe training or evaluation.
    Loads hidden states and probability values from inference shards,
    optionally splitting into train/dev sets.
    """
    if incremental:
        return _prepare_incremental_surprisal_data(
            data=data,
            model_name=model_name,
            layer_idx=layer_idx,
            test=test,
            seed=seed,
            train_ratio=train_ratio,
        )

    return _prepare_standard_surprisal_data(
        data=data,
        model_name=model_name,
        layer_idx=layer_idx,
        test=test,
        seed=seed,
        train_ratio=train_ratio,
        stats=stats,
        apply_norm=apply_norm,
        perturb=perturb,
        cat=cat,
        use_shard_cache=use_shard_cache,
    )


def _prepare_incremental_surprisal_data(
    *,
    data: str,
    model_name: str,
    layer_idx: int,
    test: bool,
    seed: int,
    train_ratio: float,
) -> Any:
    """Prepare surprisal data from incremental inference shards.
    """
    shard_dir = _incremental_shard_dir(data=data, model_name=model_name)
    if not shard_dir.exists():
        raise FileNotFoundError(
            f"Incremental shards not found at {shard_dir}. "
            "Please run baseline/inference.py with --incremental first."
        )

    feature_list: List[np.ndarray] = []
    target_list: List[float] = []

    # Conditions to process: always "good", plus "bad" if in test mode
    conditions = ["good", "bad"] if test else ["good"]

    for _, example in _stream_all_examples(shard_dir, use_cache=False):
        for condition in conditions:
            hs_key = f"{condition}_hs"
            prob_key = f"norm_{condition}_prob"

            hs_list = example.get(hs_key)
            prob_list = example.get(prob_key)

            if not isinstance(hs_list, list) or not isinstance(prob_list, list):
                continue

            for hidden_states, prob in zip(hs_list, prob_list):
                if hidden_states is None or prob is None:
                    continue

                if not isinstance(hidden_states, torch.Tensor):
                    try:
                        hidden_states = torch.as_tensor(hidden_states)
                    except (ValueError, TypeError, RuntimeError):
                        continue

                if layer_idx is None:
                    vec = hidden_states.reshape(-1).detach().cpu().numpy()
                else:
                    vec = hidden_states[layer_idx].reshape(-1).detach().cpu().numpy()

                feature_list.append(vec.astype(np.float32))
                target_list.append(float(prob))

    features = np.asarray(feature_list, dtype=np.float32)
    targets = np.asarray(target_list, dtype=np.float32)

    if features.ndim == 1:
        features = features.reshape(-1, 1)

    if test:
        return features, targets

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(features))
    split_point = int(len(indices) * float(train_ratio))

    train_idx = indices[:split_point]
    dev_idx = indices[split_point:]

    return features[train_idx], targets[train_idx], features[dev_idx], targets[dev_idx], None


def _prepare_standard_surprisal_data(
    *,
    data: str,
    model_name: str,
    layer_idx: int,
    test: bool,
    seed: int,
    train_ratio: float,
    stats: Optional[Dict[str, Optional[np.ndarray]]],
    apply_norm: bool,
    perturb: str,
    cat: Optional[str],
    use_shard_cache: bool,
) -> Any:
    """Prepare surprisal data from standard inference shards.
    """
    dataset_name = data.lower()
    shard_dir = _inference_shard_dir(dataset_name, model_name, perturb=perturb)

    shard_files = _list_all_shard_pickles(shard_dir)
    if not shard_files:
        mode = "test" if test else "train"
        raise RuntimeError(
            f"No shard files found at {shard_dir} "
            f"(data={dataset_name}, model={model_name}, mode={mode}, perturb={perturb})."
        )

    if test:
        return _load_surprisal_test_data(
            shard_dir, dataset_name, layer_idx, cat, stats, apply_norm, use_shard_cache
        )

    return _load_surprisal_train_data(
        shard_dir, dataset_name, layer_idx, cat, seed, train_ratio, apply_norm, use_shard_cache
    )


def _load_surprisal_test_data(
    shard_dir: Path,
    dataset_name: str,
    layer_idx: int,
    cat: Optional[str],
    stats: Optional[Dict[str, Optional[np.ndarray]]],
    apply_norm: bool,
    use_shard_cache: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data for surprisal evaluation.
    """
    feature_list: List[np.ndarray] = []
    target_list: List[float] = []

    for _, example in _stream_all_examples(shard_dir, cat, use_cache=use_shard_cache):
        for which in ("bad", "good"):
            vec = _to_vec(
                example, dataset_name, neuron_indices=None,
                which=which, add_prob=False, layer_idx=layer_idx
            )
            target = _extract_prob(example, is_good=(which == "good"))
            if vec is None or target is None:
                continue
            feature_list.append(vec)
            target_list.append(target)

    features = np.asarray(feature_list, dtype=np.float32)
    targets = np.asarray(target_list, dtype=np.float32)

    if apply_norm and stats is not None:
        mean_vec = stats.get("mean")
        safe_std_vec = stats.get("safe_std")
        features = _apply_norm_from_stats(features, mean_vec, safe_std_vec)

    return features, targets


def _load_surprisal_train_data(
    shard_dir: Path,
    dataset_name: str,
    layer_idx: int,
    cat: Optional[str],
    seed: int,
    train_ratio: float,
    apply_norm: bool,
    use_shard_cache: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Optional[np.ndarray]]]:
    """Load train/dev data for surprisal training.
    """
    total_examples = _count_examples_in_shards(shard_dir, cat, use_cache=use_shard_cache)
    split_dict = generate_inds(data_len=total_examples, seed=seed, train_ratio=train_ratio)

    train_ids: set[int] = set(map(int, split_dict["train"]))
    dev_ids: set[int] = set(map(int, split_dict["test"]))

    train_features: List[np.ndarray] = []
    train_targets: List[float] = []
    dev_features: List[np.ndarray] = []
    dev_targets: List[float] = []

    for example_idx, example in _stream_all_examples(shard_dir, cat, use_cache=use_shard_cache):
        dest, _ = _route_destination(
            example_idx, split=True,
            kept_train_ids=train_ids, final_test_ids=dev_ids, wanted=None
        )
        if dest == "skip":
            continue

        vec = _to_vec(
            example, dataset_name, neuron_indices=None,
            which="good", add_prob=False, layer_idx=layer_idx
        )
        target = _extract_prob(example, is_good=True)
        if vec is None or target is None:
            continue

        if dest == "train":
            train_features.append(vec)
            train_targets.append(target)
        else:
            dev_features.append(vec)
            dev_targets.append(target)

    X_train = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(train_targets, dtype=np.float32)
    X_dev = np.asarray(dev_features, dtype=np.float32)
    y_dev = np.asarray(dev_targets, dtype=np.float32)

    out_stats: Dict[str, Optional[np.ndarray]] = {"mean": None, "std": None, "safe_std": None}

    if apply_norm and X_train.size > 0:
        mean_vec, std_vec, safe_std_vec = _compute_norm_stats(X_train)
        X_train = _apply_norm_from_stats(X_train, mean_vec, safe_std_vec)
        X_dev = _apply_norm_from_stats(X_dev, mean_vec, safe_std_vec)
        out_stats = {"mean": mean_vec, "std": std_vec, "safe_std": safe_std_vec}

    return X_train, y_train, X_dev, y_dev, out_stats


# ---------------------------------------------------------------------------
# Logistic Regression Training (Classification Probes)
# ---------------------------------------------------------------------------


def train_l1(
    dataset: FormatDataResult,
    exp: float,
    *,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train an L1-regularized logistic regression classifier.
    """
    X_train = dataset.get("X_train")
    y_train = dataset.get("y_train")

    if X_train is None or y_train is None:
        raise ValueError("train_l1 requires a split dataset with training data.")

    model = LogisticRegression(
        device_ids=GPU_DEVICE_IDS,
        regularizer=float(exp),
        penalty="l1",
        fit_intercept=True,
        dual=False,
        max_iter=int(max_iter),
        random_state=0,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_l2(
    exp: float,
    *,
    model_name: str,
    dataset: FormatDataResult,
    selected_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train an L2-regularized logistic regression classifier.
    """
    X_train = dataset.get("X_train")
    X_test = dataset.get("X_test")
    y_train = dataset.get("y_train")
    y_test = dataset.get("y_test")
    meta_train = dataset.get("meta_train")
    meta_test = dataset.get("meta_test")

    if X_train is None or y_train is None:
        raise ValueError("train_logistic_l2 requires a split dataset with training data.")

    # Apply feature selection if specified
    if selected_indices is not None:
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

    # Ensure proper array format
    X_train_arr = np.ascontiguousarray(np.asarray(X_train, dtype=np.float32))
    X_test_arr = np.ascontiguousarray(np.asarray(X_test, dtype=np.float32))

    if X_train_arr.ndim == 1:
        X_train_arr = X_train_arr.reshape(-1, 1)
    if X_test_arr.ndim == 1:
        X_test_arr = X_test_arr.reshape(-1, 1)

    # Train model
    start_time = time.time()
    classifier = LogisticRegression(
        use_gpu=True,
        device_ids=GPU_DEVICE_IDS,
        regularizer=float(exp),
        penalty="l2",
        tol=1e-6,
        fit_intercept=True,
        dual=False,
        max_iter=1000,
        random_state=0,
    )
    classifier.fit(X_train_arr, y_train)
    print(f"Training time (s): {time.time() - start_time:.2f}")

    # Find positive class column
    classes = list(classifier.classes_)
    positive_col = classes.index(1)

    # Evaluate on training set
    train_probs = classifier.predict_proba(X_train_arr)
    train_positive_probs = train_probs[:, positive_col]
    train_predictions = (train_positive_probs >= 0.5).astype(int)
    train_accuracy = float((train_predictions == y_train).mean())
    train_auc = float(roc_auc_score(y_train, train_positive_probs))

    # Evaluate on test set
    test_probs = classifier.predict_proba(X_test_arr)
    test_positive_probs = test_probs[:, positive_col]
    test_predictions = (test_positive_probs >= 0.5).astype(int)
    test_accuracy = float((test_predictions == y_test).mean())
    test_auc = float(roc_auc_score(y_test, test_positive_probs))

    # Compute paired accuracies if metadata available
    train_paired_acc: Optional[float] = None
    test_paired_acc: Optional[float] = None
    if meta_train is not None:
        train_paired_acc = _compute_paired_acc_from_scores(meta_train, train_positive_probs)
    if meta_test is not None:
        test_paired_acc = _compute_paired_acc_from_scores(meta_test, test_positive_probs)

    return {
        "weights": getattr(classifier, "coef_"),
        "intercept": getattr(classifier, "intercept_"),
        "alph": exp,
        "train_pred": train_predictions,
        "test_pred": test_predictions,
        "train_acc": train_accuracy,
        "test_acc": test_accuracy,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "paired_acc": test_paired_acc,  # Legacy alias
        "test_paired_acc": test_paired_acc,
        "train_paired_acc": train_paired_acc,
        "train_probs": train_probs,
        "test_probs": test_probs,
        "train_labels": y_train,
        "test_labels": y_test,
        "ind_and_goodness": {"train": meta_train, "test": meta_test},
    }


# ---------------------------------------------------------------------------
# Model Selection and Loading
# ---------------------------------------------------------------------------


def _select_best_l2_model(
    data: str,
    model: str,
    start_exp: int,
    end_exp: int,
    *,
    add_prob: bool,
) -> Dict[str, Any]:
    """Select the best L2 probe model from saved results.
    """
    l2_dir = _probe_dir("l2", add_prob) / data / model

    best_score = -math.inf
    best_result: Optional[Dict[str, Any]] = None

    for pkl_path in sorted(l2_dir.glob(f"-1_*exp{start_exp}_to_{end_exp}.pkl")):
        if "-1_" not in pkl_path.name:
            continue

        match = re.search(r"layer(\d+)", pkl_path.name)
        if match is None:
            continue
        layer_idx = int(match.group(1))

        with open(pkl_path, "rb") as file_handle:
            stats = pickle.load(file_handle)

        overall = stats.get("overall_stats", {})
        best_auc = overall.get("best_auc", {})
        score = best_auc.get("val_auc")

        if score is None or math.isnan(score):
            score = best_auc.get("val_acc")
        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_result = {
                "layer_idx": layer_idx,
                "weights": best_auc.get("weights"),
                "intercept": best_auc.get("intercept"),
            }

    return best_result


def _load_l1_model(
    data: str,
    model: str,
    target_ratio: float,
    start_exp: int,
    end_exp: int,
    *,
    add_prob: bool,
) -> Dict[str, Any]:
    """Load a trained L1 probe model.
    """
    l1_dir = _probe_dir("l1", add_prob) / "synthetic" / model

    candidates = [
        pkl_path
        for pkl_path in sorted(l1_dir.glob("*.pkl"))
        if "-1_" in pkl_path.name and str(target_ratio) in str(pkl_path)
    ]

    if not candidates:
        raise RuntimeError("No L1 probe results found for provided parameters")

    with open(candidates[-1], "rb") as file_handle:
        stats = pickle.load(file_handle)

    l2_best = stats.get("l2_best", {})
    return {
        "layer_idx": stats.get("layer_idx"),
        "weights": l2_best.get("weights"),
        "intercept": l2_best.get("intercept"),
        "non_zero_neurons": l2_best.get("non_zero_neurons", []),
    }


def _load_random_l1_models(
    data: str,
    model: str,
    target_ratio: Optional[float],
    start_exp: int,
    end_exp: int,
    *,
    add_prob: bool,
) -> List[Dict[str, Any]]:
    """Load multiple random L1 probe models for comparison.
    """
    l1_dir = _probe_dir("l1", add_prob, random_select=True) / "synthetic" / model
    models: List[Dict[str, Any]] = []

    for pkl_path in sorted(l1_dir.glob("*.pkl")):
        if "-1_" not in pkl_path.name or "random" not in pkl_path.name:
            continue
        if str(target_ratio) not in str(pkl_path):
            continue

        with open(pkl_path, "rb") as file_handle:
            stats = pickle.load(file_handle)

        l2_best = stats.get("l2_best", {})
        weights = l2_best.get("weights")
        intercept = l2_best.get("intercept")

        if weights is None or intercept is None:
            continue

        models.append({
            "layer_idx": stats.get("layer_idx"),
            "weights": weights,
            "intercept": intercept,
            "non_zero_neurons": l2_best.get("non_zero_neurons", []),
            "random_idx": stats.get("random_idx"),
        })

    if not models:
        raise RuntimeError("No random L1 probe results found for provided parameters")

    return models


def _load_example_lookup(data_name: str, model_name: str) -> Dict[int, Dict[str, Any]]:
    """Load all examples into a lookup dictionary by index.
    """
    shard_dir = _inference_shard_dir(data_name, model_name)
    lookup: Dict[int, Dict[str, Any]] = {}
    for example_idx, example in _stream_all_examples(shard_dir):
        lookup[int(example_idx)] = example
    return lookup


def _assert_prob_run_exists(base: str, data: str, model: str) -> None:
    """Assert that probability probe results exist.
    """
    prob_dir = _probe_dir(base, add_prob=True) / "synthetic" / model
    if not prob_dir.exists() or not any(prob_dir.glob("*.pkl")):
        raise FileNotFoundError(
            f"Expected prior --add_prob results under {prob_dir}. "
            "Run l2_classifier with --add_prob first."
        )


def _choose_layer_indices(
    *,
    data: str,
    model: str,
    start_exp: int,
    end_exp: int,
    add_prob: bool,
) -> Sequence[int]:
    """Determine which layer indices to probe.
    If adding probability features or evaluating non-synthetic data,
    returns only the best layer from prior L2 sweep. Otherwise,
    returns all layers for a full sweep.
    """
    if data != "synthetic" or add_prob:
        best_layer_res = _select_best_l2_model(
            data="synthetic",
            model=model,
            start_exp=start_exp,
            end_exp=end_exp,
            add_prob=False,
        )
        return [best_layer_res["layer_idx"]]

    return range(determine_num_layers(data, model))


def determine_num_layers(data_name: str, model_name: str) -> int:
    """Determine the number of layers in model hidden states.
    """
    shard_dir = _inference_shard_dir(data_name, model_name)
    pickle_files = _list_all_shard_pickles(shard_dir)

    if not pickle_files:
        raise FileNotFoundError(f"No inference shards found at {shard_dir}")

    stream = _stream_all_examples(shard_dir)
    try:
        _, first_example = next(stream)
    except StopIteration:
        raise ValueError("No examples found in shard files")

    if not isinstance(first_example, dict):
        raise ValueError("Unexpected shard structure when determining number of layers")

    first_vec = first_example.get("good_hs")
    if first_vec is None:
        first_vec = first_example.get("bad_hs")
    if first_vec is None:
        raise ValueError("Hidden states missing in first example")

    arr = np.asarray(first_vec)
    if arr.ndim < 2:
        raise ValueError("Hidden state does not appear to have layer dimension")

    return int(arr.shape[0])


# ---------------------------------------------------------------------------
# Dataset Formatting (Main Entry Point)
# ---------------------------------------------------------------------------


def format_data(
    data: str,
    neuron_loc: Optional[np.ndarray] = None,
    model_name: str = "olmo2-7B",
    seed: int = 0,
    split: bool = True,
    split_test_idx: Optional[List[Tuple[int, bool]]] = None,
    add_prob: bool = False,
    cat: Optional[str] = None,
    train_ratio: float = 0.8,
    perturb: str = "",
    layer_idx: Optional[int] = None,
    use_shard_cache: bool = True,
) -> FormatDataResult:
    """Format inference shard data for probe training.
    Loads hidden states from inference shards, applies normalization,
    and returns train/test splits in memory-mapped arrays.
    """
    dataset_name = data.lower()

    # Determine rows per example (some datasets only have one variant)
    rows_per_example = 1 if data in {"cola", "syntaxgym"} else 2

    # Build shard directory path
    if layer_idx is not None and dataset_name == "synthetic":
        shard_dir = _inference_shard_dir(
            dataset_name + "2", model_name, perturb=perturb
        )
    else:
        shard_dir = _inference_shard_dir(
            dataset_name, model_name, perturb=perturb
        )

    shard_files = _list_all_shard_pickles(shard_dir)
    if not shard_files:
        try:
            run_dir = f"{perturb}-1_all"
            shard_dir = RES_PATH / "inference" / data / model_name / run_dir / "all"
            shard_files = _list_all_shard_pickles(shard_dir)
        except Exception as exc:
            raise RuntimeError(
                f"No shard files found at {shard_dir} "
                f"(data={dataset_name}, model={model_name}, perturb={perturb})"
            ) from exc
    if not shard_files:
        raise RuntimeError(
            f"No shard files found at {shard_dir} "
            f"(data={dataset_name}, model={model_name}, perturb={perturb})"
        )

    # Determine feature dimension from first example
    first_block = _load_shard_block(shard_files[0], use_cache=use_shard_cache)
    first_example = next(_yield_examples_in_order(first_block))
    feature_dim = _feature_dim_from_example(
        first_example, dataset_name, neuron_loc, False, layer_idx=layer_idx
    )
    if add_prob:
        feature_dim += 1
    if not use_shard_cache:
        del first_block
        gc.collect()

    # Generate train/test splits
    total_examples = _count_examples_in_shards(shard_dir, cat, use_cache=use_shard_cache)
    split_dict = generate_inds(data_len=total_examples, seed=seed, train_ratio=train_ratio)
    train_ids = set(map(int, split_dict["train"]))
    test_ids = set(map(int, split_dict["test"]))

    memmap_dir = _memmap_dir(dataset_name, model_name, add_prob=add_prob, layer_idx=layer_idx)

    if split:
        return _format_data_split(
            shard_dir=shard_dir,
            dataset_name=dataset_name,
            neuron_loc=neuron_loc,
            add_prob=add_prob,
            cat=cat,
            layer_idx=layer_idx,
            use_shard_cache=use_shard_cache,
            feature_dim=feature_dim,
            rows_per_example=rows_per_example,
            train_ids=train_ids,
            test_ids=test_ids,
            memmap_dir=memmap_dir,
        )

    return _format_data_testonly(
        shard_dir=shard_dir,
        dataset_name=dataset_name,
        neuron_loc=neuron_loc,
        add_prob=add_prob,
        cat=cat,
        layer_idx=layer_idx,
        use_shard_cache=use_shard_cache,
        feature_dim=feature_dim,
        rows_per_example=rows_per_example,
        total_examples=total_examples,
        memmap_dir=memmap_dir,
    )


def _format_data_split(
    *,
    shard_dir: Path,
    dataset_name: str,
    neuron_loc: Optional[np.ndarray],
    add_prob: bool,
    cat: Optional[str],
    layer_idx: Optional[int],
    use_shard_cache: bool,
    feature_dim: int,
    rows_per_example: int,
    train_ids: set[int],
    test_ids: set[int],
    memmap_dir: Path,
) -> FormatDataResult:
    """Format data with train/test split into memory-mapped arrays.
    """
    print(
        f"[format_data] TRAIN kept: {len(train_ids)}/{len(train_ids)} "
        f"TEST kept: {len(test_ids)} (100% of TEST)."
    )

    # Compute normalization statistics from training data
    mean_vec, std_vec, safe_std_vec = _gather_train_stats(
        shard_dir=shard_dir,
        train_example_ids=train_ids,
        dataset_name=dataset_name,
        neuron_indices=neuron_loc,
        add_prob_features=False,
        category_name=cat,
        layer_idx=layer_idx,
        use_cache=use_shard_cache,
    )

    # Extend stats for probability feature if needed
    if add_prob and mean_vec is not None:
        mean_vec = np.concatenate([mean_vec, np.zeros(1, dtype=mean_vec.dtype)], axis=0)
        if std_vec is not None:
            std_vec = np.concatenate([std_vec, np.ones(1, dtype=std_vec.dtype)], axis=0)
        safe_std_vec = np.concatenate(
            [safe_std_vec, np.ones(1, dtype=safe_std_vec.dtype)], axis=0
        )

    # Create memory-mapped arrays
    num_train_rows = len(train_ids) * rows_per_example
    num_test_rows = len(test_ids) * rows_per_example
    memmap_dir.mkdir(parents=True, exist_ok=True)

    X_train = open_memmap(
        str(memmap_dir / f"X_train_{LAST_TOK_HS}_{feature_dim}x{num_train_rows}.npy"),
        mode="w+", dtype="float32", shape=(num_train_rows, feature_dim)
    )
    y_train = open_memmap(
        str(memmap_dir / f"y_train_{LAST_TOK_HS}_{num_train_rows}.npy"),
        mode="w+", dtype="float32", shape=(num_train_rows,)
    )
    X_test = open_memmap(
        str(memmap_dir / f"X_test_{LAST_TOK_HS}_{feature_dim}x{num_test_rows}.npy"),
        mode="w+", dtype="float32", shape=(num_test_rows, feature_dim)
    )
    y_test = open_memmap(
        str(memmap_dir / f"y_test_{LAST_TOK_HS}_{num_test_rows}.npy"),
        mode="w+", dtype="float32", shape=(num_test_rows,)
    )

    meta_train: List[Tuple[int, bool]] = []
    meta_test: List[Tuple[int, bool]] = []
    train_row_idx = 0
    test_row_idx = 0

    # Populate arrays
    for example_idx, example in _stream_all_examples(shard_dir, cat, use_cache=use_shard_cache):
        dest, _ = _route_destination(
            example_idx, split=True, kept_train_ids=train_ids,
            final_test_ids=test_ids, wanted=None
        )
        if dest == "skip":
            continue

        bad_vec = _to_vec(
            example, dataset_name, neuron_loc,
            which="bad", add_prob=add_prob, layer_idx=layer_idx
        )
        good_vec = _to_vec(
            example, dataset_name, neuron_loc,
            which="good", add_prob=add_prob, layer_idx=layer_idx
        )

        if dest == "train":
            if bad_vec is not None:
                _copy_or_normalize_row(X_train[train_row_idx], bad_vec, mean_vec, safe_std_vec)
                y_train[train_row_idx] = 0
                meta_train.append((example_idx, False))
                train_row_idx += 1
            if good_vec is not None:
                _copy_or_normalize_row(X_train[train_row_idx], good_vec, mean_vec, safe_std_vec)
                y_train[train_row_idx] = 1
                meta_train.append((example_idx, True))
                train_row_idx += 1
        else:  # test
            if bad_vec is not None:
                _copy_or_normalize_row(X_test[test_row_idx], bad_vec, mean_vec, safe_std_vec)
                y_test[test_row_idx] = 0
                meta_test.append((example_idx, False))
                test_row_idx += 1
            if good_vec is not None:
                _copy_or_normalize_row(X_test[test_row_idx], good_vec, mean_vec, safe_std_vec)
                y_test[test_row_idx] = 1
                meta_test.append((example_idx, True))
                test_row_idx += 1

    norm_stats = [mean_vec, std_vec]
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "train_stats": norm_stats,
        "test_stats": norm_stats,
        "dim": int(feature_dim),
        "split": True,
    }


def _format_data_testonly(
    *,
    shard_dir: Path,
    dataset_name: str,
    neuron_loc: Optional[np.ndarray],
    add_prob: bool,
    cat: Optional[str],
    layer_idx: Optional[int],
    use_shard_cache: bool,
    feature_dim: int,
    rows_per_example: int,
    total_examples: int,
    memmap_dir: Path,
) -> FormatDataResult:
    """Format data as test-only into memory-mapped arrays.
    """
    print(
        f"[format_data:testonly] Using {total_examples} / {total_examples} "
        "examples as TESTONLY."
    )

    # Compute normalization statistics from all data
    mean_vec, std_vec, safe_std_vec = _gather_train_stats(
        shard_dir=shard_dir,
        train_example_ids=None,
        dataset_name=dataset_name,
        neuron_indices=neuron_loc,
        add_prob_features=False,
        category_name=cat,
        layer_idx=layer_idx,
        use_cache=use_shard_cache,
    )

    # Extend stats for probability feature if needed
    if add_prob and mean_vec is not None:
        mean_vec = np.concatenate([mean_vec, np.zeros(1, dtype=mean_vec.dtype)], axis=0)
        if std_vec is not None:
            std_vec = np.concatenate([std_vec, np.ones(1, dtype=std_vec.dtype)], axis=0)
        safe_std_vec = np.concatenate(
            [safe_std_vec, np.ones(1, dtype=safe_std_vec.dtype)], axis=0
        )

    # Create memory-mapped arrays
    num_rows = total_examples * rows_per_example
    memmap_dir.mkdir(parents=True, exist_ok=True)

    X_out = open_memmap(
        str(memmap_dir / f"X_testonly_{LAST_TOK_HS}_{feature_dim}x{num_rows}.npy"),
        mode="w+", dtype="float32", shape=(num_rows, feature_dim)
    )
    y_out = open_memmap(
        str(memmap_dir / f"y_testonly_{LAST_TOK_HS}_{num_rows}.npy"),
        mode="w+", dtype="float32", shape=(num_rows,)
    )

    meta_out: List[Tuple[int, bool]] = []
    row_idx = 0

    all_ids: set[int] = set(range(total_examples))

    # Populate arrays
    for example_idx, example in _stream_all_examples(shard_dir, cat, use_cache=use_shard_cache):
        dest, _ = _route_destination(
            example_idx, split=False, kept_train_ids=set(),
            final_test_ids=all_ids, wanted=None
        )
        if dest == "skip":
            continue

        bad_vec = _to_vec(
            example, dataset_name, neuron_loc,
            which="bad", add_prob=add_prob, layer_idx=layer_idx
        )
        good_vec = _to_vec(
            example, dataset_name, neuron_loc,
            which="good", add_prob=add_prob, layer_idx=layer_idx
        )

        if bad_vec is not None:
            _copy_or_normalize_row(X_out[row_idx], bad_vec, mean_vec, safe_std_vec)
            y_out[row_idx] = 0
            meta_out.append((example_idx, False))
            row_idx += 1
        if good_vec is not None:
            _copy_or_normalize_row(X_out[row_idx], good_vec, mean_vec, safe_std_vec)
            y_out[row_idx] = 1
            meta_out.append((example_idx, True))
            row_idx += 1

    norm_stats = [mean_vec, std_vec]
    return {
        "X_train": None,
        "X_test": X_out,
        "y_train": None,
        "y_test": y_out,
        "meta_train": None,
        "meta_test": meta_out,
        "train_stats": norm_stats,
        "test_stats": norm_stats,
        "dim": int(feature_dim),
        "split": False,
    }
