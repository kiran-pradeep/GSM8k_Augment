# data_loader.py
"""
Step 1: Load GSM8k dataset from Hugging Face.

Provides a function load_gsm8k(split) -> dataset list.
Each element is a dict with keys:
  - "question": str
  - "answer": str
"""

import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset


def load_gsm8k(config: str="main", split: str="train", source: str="gsm8k") -> List[Dict[str, str]]:
    """
    Load GSM8k dataset from HuggingFace.

    Parameters:
      config : str
          Which configuration of GSM8k to load. Options include "main",
          "socratic", "train_socratic". Default is "main".
          NOTE: No need to pass this if using a custom JSONL file.
      split : str
          Which split to load. Options include "train", "test", "main",
          "train_socratic". For most purposes, "train" or "test".
          NOTE: No need to pass this if using a custom JSONL file.
      source : str
          Either "gsm8k" to load from HuggingFace, or path to a JSONL file.

    Returns:
      A list of dicts with fields {"index", "question", "answer", "final_answer"}.
    """
    if source == "gsm8k":
      # HuggingFace dataset id for GSM8k
      ds = load_dataset("openai/gsm8k", config, split=split)

      # Convert to list of dicts with only needed fields
      data = [{"index": idx, "question": ex["question"], "answer": ex["answer"]} for idx, ex in enumerate(ds)]

      for instance in data:
          answer = instance["answer"]
          # Extract final numeric answer after "####"
          final_answer = None
          for line in answer.splitlines():
              if line.strip().startswith("####"):
                  final_answer = line.replace("####", "").strip()
                  break
          instance["final_answer"] = final_answer.replace(",", "")

    else:
        # Else: treat source as JSONL path
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Custom dataset not found: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                # Use augmented fields if available, else fallback
                question = ex.get("augmented_question")
                answer = "\n".join(ex.get("augmented_answer"))
                final_answer = ex.get("final_answer")
                data.append({
                    "index": ex.get("index"),
                    "question": question,
                    "answer": answer,
                    "final_answer": final_answer
                })

    return data
