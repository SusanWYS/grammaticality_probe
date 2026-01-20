# probe Module

Implements probing classifiers for analyzing linguistic competence.

## `l2_classifier.py`

Trains L2-regularized logistic regression probes for grammaticality classification.

**CLI Usage:**

```bash
python -m probe.l2_classifier \
    --data <dataset_name> \
    --model <model_name> \
    --start_exp <int> \
    --end_exp <int> \
    [--add_prob]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `synthetic` | Training dataset |
| `--model` | str | `llama3-2-1B` | Model name |
| `--start_exp` | int | `-2` | Start exponent for exp sweep (exp = 2^exp) |
| `--end_exp` | int | `5` | End exponent for exp sweep |
| `--add_prob` | flag | - | Append normalized log-prob as feature |

**Output:**
Results saved to `results/probe/l2[_prob]/<data>/<model>/-1_layer<N>_exp<start>_to_<end>.pkl`

---

## `l1_classifier.py`

Sparse feature selection using L1 regularization followed by L2 retraining.

**CLI Usage:**

```bash
python -m probe.l1_classifier \
    --data <dataset_name> \
    --model <model_name> \
    --l1_exp_start <float> \
    --l1_exp_step <float> \
    --l2_start_exp <int> \
    --l2_end_exp <int> \
    [--add_prob] \
    [--random]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `synthetic` | Training dataset |
| `--model` | str | `llama3-2-1B` | Model name |
| `--l1_exp_start` | float | `1.0` | Starting exp for L1 sweep |
| `--l1_exp_step` | float | `5000.0` | Step size for L1 exp adjustment |
| `--l2_start_exp` | int | `-2` | Start exponent for L2 sweep |
| `--l2_end_exp` | int | `5` | End exponent for L2 sweep |
| `--add_prob` | flag | - | Append normalized log-prob as feature |
| `--random` | flag | - | Use random feature selection (30 runs) |

**Target Sparsity Levels:**
By default, trains probes at 0.01%, 0.05%, 0.1%, and 0.5% of total features.

**Output:**
Results saved to `results/probe/l1[_prob][_random]/<data>/<model>/-1_layerall[_random<N>]_l2_<start>_to_<end>_num<pct>.pkl`

---

## `surprisal_probe.py`

Trains ridge regression probes to predict surprisal (negative log-probability) from hidden states.

**CLI Usage:**

```bash
# Training mode
python -m probe.surprisal_probe \
    --data <dataset_name> \
    --model <model_name> \
    --train_ratio <float> \
    --start_exp <int> \
    --end_exp <int> \
    [--incremental]

# Evaluation mode
python -m probe.surprisal_probe \
    --data <train_dataset> \
    --model <model_name> \
    --eval_data <eval_dataset> \
    [--incremental]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `synthetic` | Training dataset |
| `--model` | str | `olmo2-7B` | Model name |
| `--train_ratio` | float | `0.8` | Train/dev split ratio |
| `--start_exp` | int | `-2` | Start exponent for exp sweep |
| `--end_exp` | int | `5` | End exponent for exp sweep |
| `--incremental` | flag | - | Use per-token incremental data |
| `--eval_data` | str | - | Evaluation dataset (enables eval mode) |

**Output:**
- Training: `results/probe/surprisal_probe/<data>/<model>/-1[__incremental].pkl`
- Evaluation: `results/probe/surprisal_probe/cross_eval/<train>_to_<eval>/<model>/-1[_incremental]_eval_<eval>.pkl`
