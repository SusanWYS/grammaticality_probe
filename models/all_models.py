import os
import sys
import re
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import list_repo_refs
from tqdm import tqdm

from models.helpers import (
    model_2_hfname,
    find_decoder_layers,
    _get_fewshot_template,
    _lm_head_module
)
from probe.helpers import _stream_all_examples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True)

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]

class hfModels:
    def __init__(self, model_name):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_2_hfname[model_name])
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype="float16",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        causal_params = {
            "pretrained_model_name_or_path": model_2_hfname[model_name],
            "device_map": "auto",
        }
        causal_params["quantization_config"] = quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(**causal_params)

        self.model.eval()
        self.eos = self.tokenizer.eos_token_id

    @torch.no_grad()
    def fewshot_in_english(
        self,
        sentence: str
    ) -> Dict[str, float]:
        """
        Few-shot grammaticality scoring.

        We construct two prompts:
          {fewshots} Sentence: {sentence} Grammatical: Yes.
          {fewshots} Sentence: {sentence} Grammatical: No.

        Returns:
            {"Yes": [score_for_yes], "No": [score_for_no]}
        """
        config = _get_fewshot_template()
        prefix: str = config["prefix"]
        template: str = config["template"]
        label_map: Dict[bool, str] = config["label_map"]

        prefix_ids_no_spec = self.tokenizer(
            prefix,
            add_special_tokens=False,
        )["input_ids"]

        scores: Dict[str, float] = {}

        for grammatical_yes, key in ((True, "Yes"), (False, "No")):
            label_str = label_map[grammatical_yes]
            line = template.format(sentence, label_str)
            prompt = prefix + line

            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            inputs = {k: v.to(self.model.device) for k, v in encoded.items()}
            input_ids = inputs["input_ids"]
            attn = inputs.get("attention_mask", torch.ones_like(input_ids))

            with torch.no_grad():
                outputs = self.model(**inputs)

            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            tok_logps = (
                torch.log_softmax(shift_logits, dim=-1)
                .gather(2, shift_labels.unsqueeze(-1))
                .squeeze(-1)
            )

            cond_logp = tok_logps[0].sum()
            scores[key] = [cond_logp.item()]

        return scores


    @torch.no_grad()
    def forward_and_return_hs(self, text: str, incremental: bool = False) -> Tuple[any, any]:
        """
        Returns:
            hs: (Standard) Tensor [L+1, H] for last token
                (Incremental) List of Tensors [L+1, H] for each token.
            norm_prob: (Standard) float sequence-level log-prob
                    (Incremental) List of floats (normalized prefix log-probs).
        """
        inputs = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", torch.ones_like(input_ids))

        valid_len = int(attn[0].sum().item())
        last_idx = valid_len - 1
        layers = list(find_decoder_layers(self.model))
        L = len(layers)

        handles = []
        captures = [None] * (L + 1)
        if not incremental:
            def _mk_pre_hook(slot: int):
                def _pre(_, args):
                    # Capture the hidden state for the last valid token only
                    captures[slot] = args[0][0, last_idx, :].detach().to("cpu")
                return _pre

            all_target_layers = layers + [_lm_head_module(self.model)]
            for i, module in enumerate(all_target_layers):
                handles.append(module.register_forward_pre_hook(_mk_pre_hook(i)))

        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(**inputs, use_cache=False, output_hidden_states=incremental)

        for h in handles: h.remove()

        # Logits: [B, T, V] -> exclude last prediction; Labels: [B, T] -> exclude first token (BOS)
        tok_logps = (
            torch.log_softmax(out.logits[:, :-1, :], dim=-1)
            .gather(2, input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )[0]

        if incremental:
            if out.hidden_states is None:
                raise RuntimeError("output_hidden_states=True failed to return states.")

            hs_list = []
            norm_prob_list = []

            for pos in range(1, valid_len):
                token_hs = torch.stack([layer[0, pos, :].detach().to("cpu") for layer in out.hidden_states])
                hs_list.append(token_hs)
                prefix_lp = tok_logps[:pos].sum().item()
                norm_prob_list.append(prefix_lp / float(pos))

            return hs_list, norm_prob_list

        else:
            if any(c is None for c in captures):
                raise RuntimeError(f"Hook capture failed. Missing: {[i for i, c in enumerate(captures) if c is None]}")

            hs_tensor = torch.stack(captures)
            total_prob = tok_logps[:last_idx].sum().item()
            try:
                norm_prob = total_prob / float(last_idx)
                return hs_tensor, norm_prob
            except:
                return None, None


if __name__ == "__main__":
    pass
