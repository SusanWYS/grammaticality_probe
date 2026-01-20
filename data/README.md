# data Module

Handles dataset loading, preprocessing, and synthetic perturbation generation.

## `format_data.py`

Defines dataset classes for loading acceptability judgment benchmarks.

**Supported Dataset Classes:**

| Class | Dataset | Language | Source |
|-------|---------|----------|--------|
| `BLiMP` | BLiMP | English | `nyu-mll/blimp` |
| `BLiMP_nl` | BLiMP-NL | Dutch | `juletxara/blimp-nl` |
| `CoLA` | CoLA | English | `nyu-mll/glue` (cola) |
| `SyntaxGym` | SyntaxGym | English | `cpllab/SyntaxGym` |
| `Plausibility` | Plausibility | English | [Local CSV files](https://github.com/carina-kauf/lm-event-knowledge/tree/ac6fca04224b47498895cc7ee4dc4d8d771666af/analyses/clean_data) |
| `Scala` | ScaLA | Swedish | `alexandrainst/scala` |
| `ITACoLA` | ItaCoLA | Italian | `gsarti/itaCoLA` |
| `RuCoLA` | RuCoLA | Russian | `RussianNLP/ruCoLA` |
| `JCoLA` | JCoLA | Japanese | `shunk031/JGLUE` |
| `SLING` | SLING | English | `suchirsalhan/SLING` |
| `SynthData` | Synthetic | English | [PTB](https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset) + [Gutenberg](https://huggingface.co/datasets/GenRM/gutenberg-dpo-v0.1-jondurbin) |

**SynthData Perturbations:**

| Perturbation | Description |
|--------------|-------------|
| `delete` | Randomly delete 1-5 tokens |
| `insertion` | Insert 1-5 random tokens |
| `local_shuf` | Shuffle a 5-token window |
| `all` | Apply all perturbation types (1/3 each) |

