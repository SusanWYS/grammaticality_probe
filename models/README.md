# models Module

Wraps HuggingFace causal language models for inference.

## `all_models.py`

Provides the `hfModels` class for loading and running inference with LLMs.

**Class: `hfModels`**

```python
from models.all_models import hfModels

model = hfModels(model_name="olmo2-7B")

# Extract hidden states and normalized log-prob
hidden_states, norm_prob = model.forward_and_return_hs(text, incremental=False)

# Few-shot grammaticality scoring
scores = model.fewshot_in_english(sentence)
# Returns: {"Yes": [log_prob], "No": [log_prob]}
```

**Methods:**

| Method | Description |
|--------|-------------|
| `forward_and_return_hs(text, incremental)` | Returns (hidden_states, norm_prob). Hidden states are [L+1, H] for standard mode, list of [L+1, H] for incremental. |
| `fewshot_in_english(sentence)` | Returns log-probs for "Yes"/"No" grammaticality labels |

---

Model configuration and utility functions.

**Model Registry (`model_2_hfname`):**

| Short Name | HuggingFace Model ID |
|------------|---------------------|
| `olmo2-7B` | `allenai/OLMo-2-1124-7B` |
| `olmo3-7B` | `allenai/Olmo-3-1025-7B` |
| `gemma2-2B` | `google/gemma-2-2b` |
| `gemma2-9B` | `google/gemma-2-9b` |
| `llama3-1-8B` | `meta-llama/Llama-3.1-8B` |
| `llama3-2-1B` | `meta-llama/Llama-3.2-1B` |
