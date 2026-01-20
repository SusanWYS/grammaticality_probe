# baseline Module

Runs LLM inference on acceptability judgment datasets to extract hidden states and compute normalized log-probabilities.

## `inference.py`

Extracts per-layer hidden states (at the last token position) and computes length-normalized log-probabilities for each sentence.

**CLI Usage:**

```bash
python -m baseline.inference \
    --data <dataset_name> \
    --model <model_name> \
    --perturb <perturbation_type> \
    [--incremental]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `blimp` | Dataset name (blimp, syntaxgym, cola, plausibility, scala, itacola, rucola, jcola, sling, blimp-nl, synthetic) |
| `--model` | str | `olmo2-7B` | Model name (see [Supported Models](../README.md#supported-models)) |
| `--perturb` | str | `all` | Perturbation type for synthetic data (delete, insertion, local_shuf, all) |
| `--incremental` | flag | - | Save per-token hidden states and log-probs |

**Output:**
Results are saved to `results/inference/<data>/<model>/<run_id>_all/all/*.pkl`

---

## `fewshot.py`

Runs few-shot grammaticality judgment using in-context learning.

**CLI Usage:**

```bash
python -m baseline.fewshot \
    --data <dataset_name> \
    --model <model_name>
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `blimp` | Dataset name (blimp, syntaxgym, cola) |
| `--model` | str | `olmo2-7B` | Model name |

**Output:**
Results are saved to `results/inference/<data>/<model>/-1_fewshot_all/all/*.pkl`

---

## `helpers.py`

Utility functions for sharding inference results.

**CLI Usage:**

```bash
python -m baseline.helpers \
    --data <dataset_name> \
    --model <model_name> \
    [--flat_chunk_size <int>]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | required | Dataset name |
| `--model` | str | required | Model name |
| `--flat_chunk_size` | int | `5000` | Number of examples per shard file |
