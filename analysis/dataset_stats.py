"""Compute basic sentence statistics for key datasets.

This script loads each dataset used in the project and prints the total number
of sentences along with the proportion labeled as "good" (grammatical or
preferred). It relies on the dataset loaders defined in ``data.format_data``.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from data.format_data import (
    BLiMP,
    CoLA,
    Plausibility,
    ITACoLA,
    JCoLA,
    RuCoLA,
    SLING,
    Scala,
    SyntaxGym,
    BLiMP_nl,
    SynthData,
)


DatasetLoader = Callable[[], Dict[str, list]]


def _count_stats(data: Dict[str, list]) -> Tuple[int, float]:
    """Return total sentences and proportion of good sentences.

    ``data`` is expected to contain ``good_txt`` and ``bad_txt`` lists whose
    entries are either strings or ``None``. ``None`` entries are ignored for the
    counts.
    """

    good_count = sum(1 for sent in data.get("good_txt", []) if sent is not None)
    bad_count = sum(1 for sent in data.get("bad_txt", []) if sent is not None)
    total = good_count + bad_count
    proportion_good = good_count / total if total else 0.0
    return total, proportion_good


def _load_blimp() -> Dict[str, list]:
    ds = BLiMP().load_data()
    return ds


def _load_cola() -> Dict[str, list]:
    return CoLA().load_data()


def _load_syntaxgym() -> Dict[str, list]:
    return SyntaxGym().load_data()


def _load_plausibility() -> Dict[str, list]:
    return Plausibility().load_data()


def _load_scala() -> Dict[str, list]:
    return Scala().load_data()


def _load_nl_blimp() -> Dict[str, list]:
    return BLiMP_nl().load_data()


def _load_itacola() -> Dict[str, list]:
    return ITACoLA().load_data()


def _load_rucola() -> Dict[str, list]:
    return RuCoLA().load_data()


def _load_jcola() -> Dict[str, list]:
    return JCoLA().load_data()


def _load_sling() -> Dict[str, list]:
    return SLING().load_data()


DATASETS: Dict[str, DatasetLoader] = {
    # "blimp": _load_blimp,
    # "cola": _load_cola,
    # "syngym": _load_syntaxgym,
    # "plausibility": _load_plausibility,
    # "scala": _load_scala,
    # "nl-blimp": _load_nl_blimp,
    # "itacola": _load_itacola,
    # "rucola": _load_rucola,
    # "jcola": _load_jcola,
    # "sling": _load_sling,
    "synthetic": SynthData(data_name="synthetic", perturb="all").load_data
}


def main() -> None:
    print("Dataset sentence statistics\n-----------------------------")
    for name, loader in DATASETS.items():
        print(f"Loading {name}…")
        data = loader()
        total, proportion_good = _count_stats(data)
        print(
            f"{name}: total sentences = {total}, proportion good = {proportion_good:.4f}"
        )


if __name__ == "__main__":
    main()
