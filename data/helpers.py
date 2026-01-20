
from datasets import load_dataset, concatenate_datasets
import spacy
from spacy.tokens import Doc
from typing import Dict, List, Optional
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]

# Conditions labeled as "good" (grammatical / preferred) vs "bad" (ungrammatical or garden-path/dispreferred)
good_cond = {
    "np_match",
    "vp_match",
    "that_nogap",
    "what_subjgap",
    "what_gap",
    "neg_pos",
    "neg_neg",
    "match_sing",
    "match_plural",
    "no-sub_no-matrix",
    "sub_matrix",
}

bad_cond = {
    "np_mismatch",
    "vp_mismatch",
    "what_nogap",
    "that_subjgap",
    "what_matrixgap",
    "that_matrixgap",
    "that_gap",
    "pos_pos",
    "pos_neg",
    "mismatch_sing",
    "mismatch_plural",
    "sub_no-matrix",
    "no-sub_matrix",
}

blimp_nl_configs = [
    "adpositional_phrases", "adverbial_modification", "anaphor_agreement",
    "argument_structure", "auxiliaries", "binding_principle_a", "complementive",
    "crossing_dependencies", "determiners", "extraposition", "finite_argument_clause",
    "infinitival_argument_clause", "nominalization", "parasitic_gaps", "passive",
    "quantifiers", "r_words", "relativization", "topicalization", "verb_second",
    "wh_movement", "wh_movement_restrictions"
]

blimp_configs = [
    'adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement',
    'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island',
    'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction',
    'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1',
    'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
    'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2',
    'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun',
    'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2',
    'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2',
    'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive',
    'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
    'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2',
    'left_branch_island_echo_question', 'left_branch_island_simple_question',
    'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present',
    'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1',
    'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3',
    'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2',
    'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island',
    'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2',
    'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap',
    'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance',
    'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance'
]


def tokenize(inputs, pos = False):
    if not pos:
        nlp = spacy.blank("en")
    else:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(inputs)
    return doc


def _reconstruct_sentence(doc, replace_idx: Optional[int] = None, replacement_token: Optional[str] = None):
    """Return the sentence text with an optional token replacement."""
    rebuilt: List[str] = []
    for i, token in enumerate(doc):
        if i == replace_idx and replacement_token is not None:
            rebuilt.append(replacement_token + token.whitespace_)
        else:
            rebuilt.append(token.text + token.whitespace_)

    text = "".join(rebuilt)
    return text


def build_or_load_vocab(sentences: List[str], data_name: str):
    """Return a cached vocabulary list for ``data_name``.

    The vocabulary is built by extracting expbetic tokens from ``sentences``
    using a regex. The resulting list is cached under
    ``data/cached/{data_name}_vocab.pkl`` and re-used on subsequent calls.
    """

    vocab_path = PROJECT_ROOT / "data" / "cached" / f"{data_name}_vocab.pkl"
    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    vocab_set = set()
    for sent in sentences:
        for token in re.findall(r"[A-Za-z]+", sent):
            vocab_set.add(token)

    vocab_list = list(vocab_set)

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab_list, f)
    return vocab_list


def delete_words_from_sentence(doc):
    """Randomly delete between 1 and 5 expbetic tokens from ``doc``.

    A random delete count ``n`` is sampled uniformly from ``[1, 5]``. If the
    sentence does not contain strictly more expbetic tokens than ``n`` we
    skip the sentence by returning ``None``.
    """

    num_to_delete = random.randint(1, 5)
    word_indices = [i for i, token in enumerate(doc) if token.is_exp]

    if len(word_indices) <= num_to_delete:
        return None

    delete_indices = sorted(random.sample(word_indices, num_to_delete))
    delete_set = set(delete_indices)

    rebuilt = [
        token.text + token.whitespace_
    for i, token in enumerate(doc)
    if i not in delete_set
    ]

    bad_txt = "".join(rebuilt)

    good_txt = _reconstruct_sentence(doc)
    meta = {
        "deleted_indices": delete_indices,
        "deleted_tokens": [doc[idx].text for idx in delete_indices],
    }

    return good_txt, bad_txt, meta


def insert_words_from_vocab(doc: Doc, vocab: List[str]):
    """Insert 1-5 random vocabulary words into ``doc`` at random positions.

    Sentences without tokens or runs with an empty vocabulary are skipped.
    """

    if len(doc) == 0 or not vocab:
        return None

    num_insert = random.randint(1, 5)
    insert_positions = sorted(random.sample(range(len(doc) + 1), num_insert))
    insert_tokens = [random.choice(vocab) for _ in range(num_insert)]

    rebuilt: List[str] = []
    insert_idx = 0
    for i in range(len(doc) + 1):
        while insert_idx < num_insert and insert_positions[insert_idx] == i:
            rebuilt.append(insert_tokens[insert_idx] + " ")
            insert_idx += 1
        if i < len(doc):
            rebuilt.append(doc[i].text + doc[i].whitespace_)

    bad_txt = "".join(rebuilt)
    if bad_txt:
        bad_txt = bad_txt[0].upper() + bad_txt[1:]

    good_txt = _reconstruct_sentence(doc)
    meta = {
        "insert_positions": insert_positions,
        "insert_tokens": insert_tokens,
    }

    return good_txt, bad_txt, meta


def local_shuffle(doc: Doc):
    """Randomly shuffle five adjacent tokens within ``doc``.

    We choose a contiguous five-token window (tokens may include punctuation).
    If the shuffled order matches the original, we reshuffle until it differs.
    If no window can be shuffled into a new order, the sentence is skipped.
    """

    if len(doc) < 5:
        return None

    candidate_starts = [
        i for i in range(len(doc) - 4)
    ]
    if not candidate_starts:
        return None

    random.shuffle(candidate_starts)
    for start_idx in candidate_starts:
        segment_indices = list(range(start_idx, start_idx + 5))
        tokens_to_shuffle = [doc[i].text for i in segment_indices]

        if len(set(tokens_to_shuffle)) == 1:
            continue

        shuffled_tokens = tokens_to_shuffle
        while shuffled_tokens == tokens_to_shuffle:
            shuffled_tokens = random.sample(tokens_to_shuffle, len(tokens_to_shuffle))

        rebuilt = []
        for idx, token in enumerate(doc):
            if idx in segment_indices:
                rebuilt.append(shuffled_tokens[idx - segment_indices[0]] + token.whitespace_)
            else:
                rebuilt.append(token.text + token.whitespace_)

        bad_txt = "".join(rebuilt)
        if bad_txt:
            bad_txt = bad_txt[0].upper() + bad_txt[1:]

        good_txt = _reconstruct_sentence(doc)
        meta = {
            "window_start": segment_indices[0],
            "window_indices": segment_indices,
            "original_tokens": tokens_to_shuffle,
            "shuffled_tokens": shuffled_tokens,
        }

        return good_txt, bad_txt, meta
    return None

def dialogue_sentences(s: str):
    QUOTE_RE = re.compile(r'“([^”]+)”|\\\"([^\\\"]+)\\\"')
    parts = []
    for m in QUOTE_RE.finditer(s):
        piece = next(g for g in m.groups() if g)  # matched group
        parts.append(piece.strip())

    if parts: # stitch all quotes together
        utter = " ".join(parts)
        utter = re.sub(r"\s+", " ", utter).strip()
        if utter.endswith((",", "—", "–")):
            utter = utter[:-1].rstrip() + "."
    else: # no quotes
        utter = s.rstrip("\n").strip()

    seg = pysbd.Segmenter(language="en", clean=False)
    return [x.rstrip("\n").strip() for x in seg.segment(utter)]

def spacy_sent_seg(text, data_name):
    seg_text_path = PROJECT_ROOT / "data" / "cached" / f"{data_name}.pkl"
    if seg_text_path.exists():
        with open(seg_text_path, "rb") as f:
            return pickle.load(f)

    sentences = []
    seg_sbd = pysbd.Segmenter(language="en", clean=False, char_span=False)
    for doc in text:
        for sent in seg_sbd.segment(doc):           # sent is a str
            sentences.extend(dialogue_sentences(sent))

    with open(seg_text_path, "wb") as f:
        pickle.dump(sentences, f)

    return sentences

def read_wsj():
    if os.path.exists(str(Path(__file__).resolve().parent)+"/cached/ptb.pkl"):
        with open(str(Path(__file__).resolve().parent)+"/cached/ptb.pkl", "rb") as f:
            return pickle.load(f)
    all_lines = []
    dir_range = [f"0{num}" if num < 10 else str(num) for num in range(25)]
    dir_p = str(Path(__file__).resolve().parent)+"/treebank_3/raw/wsj/"
    for num in dir_range:
        d_name = dir_p + num
        sub_dir = [str(p) for p in Path(d_name).rglob("*") if p.is_file()]
        for pp in sub_dir:
            try:
                with open(pp, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            all_lines += [line.strip()]
            except UnicodeDecodeError:
                with open(pp, "r", encoding="latin-1") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            all_lines += [line.strip()]
    with open(str(Path(__file__).resolve().parent)+"/cached/ptb.pkl", "wb") as f:
        pickle.dump(all_lines, f)
    return all_lines
