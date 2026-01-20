from datasets import load_dataset, concatenate_datasets
import spacy
from spacy.tokens import Doc
from typing import Callable, Dict, List, Optional
from pathlib import Path
import pysbd
import pandas as pd
import sys
import os
import pickle
import regex as re
import random
import csv
import numpy as np
from data.helpers import (
    good_cond,
    bad_cond,
    tokenize,
    _reconstruct_sentence,
    build_or_load_vocab,
    delete_words_from_sentence,
    insert_words_from_vocab,
    local_shuffle,
    dialogue_sentences,
    spacy_sent_seg,
    read_wsj,
    blimp_configs,
    blimp_nl_configs
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]


class SynthData():
    def __init__(self, data_name, perturb = "local_shuf"):
        self.data_name = data_name
        self.perturb = perturb
        ptb_data = list(read_wsj())
        book_data = list(load_dataset("GenRM/gutenberg-dpo-v0.1-jondurbin")["train"]["chosen"])
        combined = ptb_data + book_data
        rng = random.Random(0)
        rng.shuffle(combined)
        self.data = combined

    def load_data(self):
        data_label = self.data_name

        if self.perturb == "delete":
            suffix = "delete"
        elif self.perturb == "insertion":
            suffix = "insertion"
        elif self.perturb == "local_shuf":
            suffix = "local_shuf"
        elif self.perturb == "all":
            suffix = "all"
        else:
            raise ValueError(f"Unsupported perturbation: {self.perturb}")
        data_path = PROJECT_ROOT / "data" / "cached" / f"{data_label}_{suffix}.pkl"
        if data_path.exists():
            with open(data_path, "rb+") as f:
                res = pickle.load(f)
                return res

        data = self.data

        seg_sent = spacy_sent_seg(data, data_label)
        seg_sent = seg_sent[:50000]

        vocab = None
        if self.perturb in {"insertion", "all"}:
            vocab = build_or_load_vocab(seg_sent, data_label)

        ret_data = {
            "good_txt": [],
            "bad_txt": [],
            "idx_pair": []
        }
        perturb_map = {
                    "delete": lambda tok: delete_words_from_sentence(tok),
                    "insertion": lambda tok: insert_words_from_vocab(tok, vocab),
                    "local_shuf": lambda tok: local_shuffle(tok),
                }

        def apply_perturbation(p_name, segments):
            perturb_func = perturb_map.get(p_name)
            if not perturb_func:
                return

            for seg in segments:
                tok_seg = tokenize(seg, pos=False)
                try:
                    res = perturb_func(tok_seg)
                    if res:
                        good_txt, bad_txt, meta = res
                        ret_data["good_txt"].append(good_txt)
                        ret_data["bad_txt"].append(bad_txt)
                        ret_data["idx_pair"].append(meta)
                except Exception:
                    continue
        if self.perturb == "all":
            n_total = len(seg_sent)
            third = n_total // 3
            apply_perturbation("insertion", seg_sent[:third])
            apply_perturbation("delete", seg_sent[third:third*2])
            apply_perturbation("local_shuf", seg_sent[third*2:third*3])
        else:
            apply_perturbation(self.perturb, seg_sent)
        print(len(ret_data["idx_pair"]))
        with open(data_path, "wb+") as f:
            pickle.dump(ret_data, f)

        return ret_data


class AcceptabilityDatasetBase:
    """Base helper to normalize acceptability datasets.
    """

    label_keys = (
        "acceptability", "label"
    )
    sentence_keys = (
        "sentence", "text", "utterance", "context",
    )

    def _label_key(self, example: Dict) -> str:
        for key in self.label_keys:
            if key in example: return key
        raise KeyError("No label-like key found")

    def _sentence_key(self, example: Dict) -> str:
        for key in self.sentence_keys:
            if key in example: return key
        raise KeyError("No sentence/text key found")

    def _is_good(self, label_val) -> bool:
        if isinstance(label_val, bool): return label_val
        if isinstance(label_val, (int, float)): return label_val >= 1
        if isinstance(label_val, str):
            return label_val.strip().lower() in {"acceptable", "1", "yes", "true", "grammatical", "good", "ok"}
        return False

    def _build_acceptability_dict(
        self,
        dataset,
        metadata_fn: Optional[Callable[[Dict], Dict]] = None
    ):
        ret_data = {
            "good_txt": [],
            "bad_txt": [],
            "idx_pair": [],
        }

        first_ex = dataset[0] if len(dataset) > 0 else {}
        has_aligned_keys = "sentence_good" in first_ex and "sentence_bad" in first_ex

        if has_aligned_keys:
            for ex in dataset:
                ret_data["good_txt"].append(ex["sentence_good"])
                ret_data["bad_txt"].append(ex["sentence_bad"])

                meta = metadata_fn(ex) if metadata_fn else None
                ret_data["idx_pair"].append(meta)

        else:
            for ex in dataset:
                lbl = ex[self._label_key(ex)]
                sent = ex[self._sentence_key(ex)]

                if self._is_good(lbl):
                    ret_data["good_txt"].append(sent)
                    ret_data["bad_txt"].append(None)
                else:
                    ret_data["good_txt"].append(None)
                    ret_data["bad_txt"].append(sent)

                meta = metadata_fn(ex) if metadata_fn else None
                ret_data["idx_pair"].append(meta)

        return ret_data


class CoLA(AcceptabilityDatasetBase):

    def __init__(self):
        pass

    def load_data(self, split: str = "train"):
        ds_dict = load_dataset("nyu-mll/glue", "cola")

        combined_dataset = concatenate_datasets([
            ds_dict["train"],
            ds_dict["validation"],
            ds_dict["test"]
        ])
        combined_dataset = combined_dataset.filter(lambda x: x["label"] in {0, 1})
        return self._build_acceptability_dict(combined_dataset)


class Scala(AcceptabilityDatasetBase):

    def load_data(self):
        # Languages supported by alexandrainst/scala
        ds_dict = load_dataset("alexandrainst/scala", "sv")
        combined = concatenate_datasets(list(ds_dict.values()))
        return self._build_acceptability_dict(combined)


class ITACoLA(AcceptabilityDatasetBase):

    def load_data(self):
        ds_dict = load_dataset("gsarti/itaCoLA", "scores")
        combined = concatenate_datasets(list(ds_dict.values()))
        return self._build_acceptability_dict(combined)


class RuCoLA(AcceptabilityDatasetBase):

    def load_data(self):
        ds_dict = load_dataset("RussianNLP/ruCoLA")
        combined = concatenate_datasets(list(ds_dict.values()))
        return self._build_acceptability_dict(combined)


class JCoLA(AcceptabilityDatasetBase):

    def load_data(self):
        ds_dict = load_dataset(
            "shunk031/JGLUE",
            "jcola",
            trust_remote_code=True,  # avoids future warnings / breakage
        )
        combined = concatenate_datasets(list(ds_dict.values()))
        return self._build_acceptability_dict(combined)


class SLING(AcceptabilityDatasetBase):
    pairwise: bool = True

    def load_data(self):
        ds = load_dataset("suchirsalhan/SLING", split="train")

        def _get_meta(ex):
            return {
                "pair_id": ex.get("pair_ID"),
                "phenomenon": ex.get("phenomenon"),
                "paradigm": ex.get("paradigm"),
                "field": ex.get("field"),
            }
        return self._build_acceptability_dict(ds, metadata_fn=_get_meta)


class BLiMP_nl(AcceptabilityDatasetBase):

    def __init__(self):
        self.configs = blimp_nl_configs

    def load_data(self):
        ds_ls = [load_dataset("juletxara/blimp-nl", c)["train"] for c in self.configs]
        ds = concatenate_datasets(ds_ls)

        def _get_meta(ex: Dict) -> Dict:
            return {
                "linguistic_phenomenon": ex.get("linguistic_phenomenon"),
                "paradigm": ex.get("paradigm"),
                "item_id": ex.get("item_id"),
            }

        return self._build_acceptability_dict(ds, metadata_fn=_get_meta)


class BLiMP(AcceptabilityDatasetBase):

    def __init__(self):
        self.configs = blimp_configs

    def load_data(self):
        ds_ls = [load_dataset("nyu-mll/blimp", c)["train"] for c in self.configs]
        ds = concatenate_datasets(ds_ls)

        def _get_meta(ex: Dict) -> Dict:
            return {
                "linguistic_term": ex.get("linguistic_term"),
                "uid": ex.get("UID"),
                "pair_id": ex.get("pair_id")
            }

        return self._build_acceptability_dict(ds, metadata_fn=_get_meta)

class SyntaxGym(AcceptabilityDatasetBase):
    def __init__(self):
        pass

    def load_data(self):
        ds = load_dataset("cpllab/syntaxgym", split="test")
        ret_data = {
            "good_txt": [],
            "bad_txt": [],
            "idx_pair": []
        }
        for item in ds['conditions']:
            for i, cond in enumerate(item['condition_name']):
                if cond in good_cond:
                    ret_data["good_txt"].append(item['content'][i])
                    ret_data["bad_txt"].append(None)
                    ret_data["idx_pair"].append(None)
                if cond in bad_cond:
                    ret_data["good_txt"].append(None)
                    ret_data["bad_txt"].append(item['content'][i])
                    ret_data["idx_pair"].append(None)
        return ret_data

class Plausibility():
    def __init__(self):
        pass

    def load_csv(self, fp, ret_dict):
        item_idx = 0
        with open(fp, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                item_idx += 1
                label = "good_txt" if row['Plausibility'] == "Plausible" else "bad_txt"
                ret_dict[label].append(row["Sentence"])
        if item_idx % 2 == 0:
            ret_data["idx_pair"].append(None)
        return ret_dict

    def load_data(self):
        ret_dict = {
            "good_txt": [],
            "bad_txt": [],
        }
        dir_path = PROJECT_ROOT / "data" / "EventKnowledge"
        ret_dict = self.load_csv(dir_path / "clean_DTFit_SentenceSet.csv", ret_dict)
        ret_dict = self.load_csv(dir_path / "clean_EventsAdapt_SentenceSet.csv", ret_dict)
        ret_dict = self.load_csv(dir_path / "clean_EventsRev_SentenceSet.csv", ret_dict)
        return ret_dict

if __name__ == "__main__":
    # babylm = BabyLM()
    # babylm = babylm.load_babylm()
    data = JCoLA()
    # data = swap.load_data()
    # data = SynthData("synthetic", perturb = "all")
    # data = swap.load_data()
    # data = Plausibility()
    # data = SynthData("synthetic", perturb="swap_pos")
    data_ = data.load_data()
    pass
