"""
examples.py — Load and split labeled JSON examples into correct / incorrect lists.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_examples(path: Path) -> tuple[list[dict], list[dict]]:
    """
    Read *path* (a JSON array of example objects) and split by label.

    Returns:
        (correct_examples, incorrect_examples)
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    correct   = [ex for ex in raw if ex.get("label", "").upper() == "CORRECT"]
    incorrect = [ex for ex in raw if ex.get("label", "").upper() == "INCORRECT"]
    return correct, incorrect
