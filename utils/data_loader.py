# data_loader.py
"""
Step 1: Load GSM8k dataset from Hugging Face.

Provides a function load_gsm8k(split) -> dataset list.
Each element is a dict with keys:
  - "question": str
  - "answer": str
"""

from typing import List, Dict
from datasets import load_dataset


def load_gsm8k(config: str = "main", split: str = "train") -> List[Dict[str, str]]:
    """
    Load GSM8k dataset from HuggingFace.

    Parameters:
      split : str
          Which split to load. Options include "train", "test", "main",
          "train_socratic". For most purposes, "train" or "test".

    Returns:
      A list of dicts with fields {"question", "answer"}.
    """
    # HuggingFace dataset id for GSM8k
    ds = load_dataset("openai/gsm8k", config, split=split)

    # Convert to list of dicts with only needed fields
    data = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]

    return data
