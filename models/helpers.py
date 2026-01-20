from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs
import torch
import torch.nn as nn
import re
from typing import Dict


model_2_hfname = {
  "olmo2-7B": "allenai/OLMo-2-1124-7B",
  "olmo2-13B": "allenai/OLMo-2-1124-13B",
  "olmo2-32B": "allenai/OLMo-2-0325-32B",
  "olmo3-7B": "allenai/Olmo-3-1025-7B",
  "gemma2-2B": "google/gemma-2-2b",
  "gemma2-9B": "google/gemma-2-9b",
  "gemma3-1B": "google/gemma-3-1b-pt",
  "llama3-8B": "meta-llama/Meta-Llama-3-8B",
  "llama3-1-8B": "meta-llama/Llama-3.1-8B",
  "llama3-2-1B": "meta-llama/Llama-3.2-1B",
  "llama3-2-3B": "meta-llama/Llama-3.2-3B",
  "olmo2-1B": "allenai/OLMo-2-0425-1B",
  "babygpt": "BabyLM-community/babylm-baseline-100m-gpt2",
  "babygpt10" : "BabyLM-community/babylm-baseline-10m-gpt2",
  "gpt-bert100": "ltg/gpt-bert-babylm-base",
  "gpt-bert10": "ltg/gpt-bert-babylm-small",
  "gpt2-small": "openai-community/gpt2"
}


model_layers = {
    "gemma3-1B": 26,
    "llama3-1-8B": 32,
    "olmo2-7B": 32,
    "olmo3-7B": 32,
    "gemma2-2B": 26,
    "gemma2-9B": 42,
    "llama3-2-1B": 16,
    "llama3-2-3B": 28,
    "llama3-8B": 32,
    "llama3-1-8B": 32,
    "olmo2-13B": 40,
    "olmo2-1B": 16
}

model_hdim= {
    "gemma3-1B": 1152,
    "llama3-1-8B": 4096,
    "olmo2-7B": 4096,
    "olmo3-7B": 4096,
    "gemma2-2B": 2304,
    "gemma2-9B": 3584,
    "llama3-2-1B": 2048,
    "llama3-2-3B": 3072,
    "llama3-8B": 4096,
    "llama3-1-8B": 4096,
    "olmo2-13B": 5120,
    "olmo2-1B": 2048
}

def _lm_head_module(model):
    for name in ("lm_head", "embed_out", "output", "language_model_head"):
        m = getattr(model, name, None)
        if isinstance(m, torch.nn.Module):
            return m

def _get_fewshot_template() -> Dict[str, object]:
    lang_config= {
            "prefix": (
                "Determine if the following sentences are grammatical.\n"
                "Sentence: The boy kick the ball. Grammatical: No.\n"
                "Sentence: That you are back surprised me. Grammatical: Yes.\n"
                "Sentence: The story goes on and on. Grammatical: Yes.\n"
                "Sentence: Last night I was ever drunk. Grammatical: No.\n"
            ),
            "template": "Sentence: {} Grammatical: {}.\n",
            "label_map": {True: "Yes", False: "No"},
        }
    return lang_config


def find_decoder_layers(model):
    """
    Return a list of decoder block modules for many HF causal LMs.
    Never returns None (returns [] if nothing found).
    """
    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module

    base = getattr(model, "model", model)

    def get_path(obj, path):
        cur = obj
        for p in path.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    # Try common attribute paths (ordered by likelihood)
    candidate_paths = (
        "transformer.h",        # GPT-2 / GPT-J / BLOOM / Falcon
        "model.decoder.layers", # OPT
        "decoder.layers",       # some decoders
        "gpt_neox.layers",      # GPT-NeoX / Pythia
        "layers",               # LLaMA / Mistral / Phi (after base = model.model)
    )
    for path in candidate_paths:
        obj = get_path(base, path)
        if isinstance(obj, (nn.ModuleList, list, tuple)):
            return list(obj)

    # Fallback: detect by class name
    block_type_names = {
        "GPT2Block", "GPTJBlock", "BloomBlock", "FalconDecoderLayer",
        "OPTDecoderLayer", "LlamaDecoderLayer", "MistralDecoderLayer",
        "NeoXLayer", "RWBlock", "PhiDecoderLayer"
    }
    blocks = [m for m in model.modules() if m.__class__.__name__ in block_type_names]
    return blocks
