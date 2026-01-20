# Grammaticality Probing

This repository contains code for the paper "Implicit Representations of Grammaticality in Language Models".

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Replication Pipeline](#replication-pipeline)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
ling_comp/
├── baseline/           # Inference and few-shot evaluation
│   ├── inference.py    # Extract hidden states and log-probs
│   ├── fewshot.py      # Few-shot prompting evaluation
│   └── helpers.py      # Sharding utilities
├── data/               # Dataset loading and preprocessing
│   ├── format_data.py  # Dataset class definitions
│   └── helpers.py      # Tokenization and perturbation functions
├── models/             # Model wrappers
│   ├── all_models.py   # HuggingFace model interface
│   └── helpers.py      # Model configs and layer extraction
├── probe/              # Probing classifiers
│   ├── l2_classifier.py     # L2-regularized logistic regression
│   ├── l1_classifier.py     # Sparse L1+L2 probing pipeline
│   ├── surprisal_probe.py   # Surprisal prediction probes
│   └── helpers.py           # Data formatting and evaluation
└── results/            # Output directory for inference and probe results
```

---

## Modules

| Module | Description | Details |
|--------|-------------|---------|
| [baseline](baseline/README.md) | Inference and few-shot evaluation | Extract hidden states and log-probs from LLMs |
| [data](data/README.md) | Dataset loading and preprocessing | Load acceptability judgment benchmarks and generate perturbations |
| [models](models/README.md) | Model wrappers | HuggingFace model interface for causal LMs |
| [probe](probe/README.md) | Probing classifiers | L1/L2 logistic regression and surprisal probes |

---

## Replication Pipeline

### Step 1: Run Inference

Extract hidden states and compute log-probabilities:

```bash
# Run inference on BLiMP
python -m baseline.inference --data blimp --model olmo2-7B

# Run inference on synthetic data with all perturbations
python -m baseline.inference --data synthetic --model olmo2-7B --perturb all

# Run incremental inference (per-token)
python -m baseline.inference --data blimp --model olmo2-7B --incremental
```

### Step 2: Train and Test L2 Probes

Train and evaluate layer-wise L2-regularized probes:

```bash
# Train
python -m probe.l2_classifier --data synthetic --model olmo2-7B --start_exp -2 --end_exp 5

# Test
python -m probe.test_classifier --model olmo2-7B --eval_data BLiMP --train_data synthetic --start_exp -2 --end_exp 5

# Train / Eval with logprob as an extra feature
python -m ... --add_prob
```

### Step 3: Train and Test Sparse Probes

Train and evaluate sparse probes with feature selection:

```bash
# Train
python -m probe.l1_classifier --data synthetic --model olmo2-7B

# Test: choose target_ratio from [0.01, 0.05, 0.1, 0.5]
python -m probe.test_classifier --model olmo2-7B --eval_data BLiMP --train_data synthetic --target_ratio 0.01

# Train and Test with 30 randomly selected subsets of neurons
python -m ... --random
```

### Step 4: Train Surprisal Probes

Train and evaluate surprisal prediction:

```bash
# Train on last token only
python -m probe.surprisal_probe --data synthetic --model olmo2-7B

# Evaluate on BLiMP on last token only
python -m probe.surprisal_probe --data synthetic --model olmo2-7B --eval_data blimp

# Train / evaluate on all token positions
python -m ... --incremental
```
---

## Supported Models

| Models |
|--------------|
| **[olmo2-7B](https://huggingface.co/allenai/OLMo-2-1124-7B)** |
| **[olmo3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)** |
| **[gemma2-2B](https://huggingface.co/google/gemma-2-2b)** |
| **[gemma2-9B](https://huggingface.co/google/gemma-2-9b)** |
| **[llama3-1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)** |
| **[llama-3-2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)** |

---

## Supported Datasets

| Dataset | Language | Type | Paired |
|---------|----------|------|--------|
| [blimp](https://huggingface.co/datasets/nyu-mll/blimp) | English | Grammaticality | Yes |
| [cola](https://huggingface.co/datasets/nyu-mll/glue/viewer/cola) | English | Grammaticality | No |
| [syntaxgym](https://huggingface.co/datasets/cpllab/syntaxgym) | English | Grammaticality | No |
| [plausibility](https://github.com/carina-kauf/lm-event-knowledge/tree/ac6fca04224b47498895cc7ee4dc4d8d771666af/analyses/clean_data) | English | Plausibility | Yes |
| [blimp-nl](https://huggingface.co/datasets/juletxara/blimp-nl) | Dutch | Grammaticality | Yes |
| [scala](https://huggingface.co/datasets/blastwind/github-code-scala) | Swedish | Grammaticality | No |
| [itacola](https://huggingface.co/datasets/gsarti/itacola) | Italian | Grammaticality | No |
| [rucola](https://huggingface.co/datasets/RussianNLP/rucola) | Russian | Grammaticality | No |
| [jcola](https://huggingface.co/datasets/shunk031/JGLUE) | Japanese | Grammaticality | No |
| [sling](https://huggingface.co/datasets/suchirsalhan/SLING) | Chinese | Grammaticality | Yes |
| synthetic ([ptb](https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset), [gutenberg-dpo](https://huggingface.co/datasets/GenRM/gutenberg-dpo-v0.1-jondurbin)) | English | Perturbation-based | Yes |

---

## License

See [LICENSE](LICENSE) for details.
