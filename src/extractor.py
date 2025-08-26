# extractor.py
"""
Step 3: LLM-based metric extraction.

This module queries the LLM to extract measurable values and their units
from GSM8k problems (both question and answer). Each extracted element
should be tied to a unique "variable" so it can later be aligned with
templatization placeholders.
"""

import json
from typing import Any, Dict, List

from utils.llm_client import render_template


def extract_metrics_llm(chat, question: str, answer: str, templates_dir: str) -> List[Dict[str, Any]]:
    """
    Use LLM to extract measurable values + units from question and answer.

    Returns a list of dicts with fields:
      - variable: str (unique identifier or placeholder name, if possible)
      - value: float
      - unit: str (dollar, rupee, pound, kg, mile, km, etc.)
      - context: "question" or "answer"

    Example return:
    [
      {"variable": "wallet_cost", "value": 100, "unit": "dollar", "context": "question"},
      {"variable": "parents_gift", "value": 15, "unit": "dollar", "context": "answer"}
    ]
    """
    prompt = render_template(
        templates_dir, 
        "extract_metrics.txt", 
        question=question, 
        answer=answer
    )

    resp = chat.invoke(prompt)
    text = resp.content.strip()
    text = text.replace("```", "").replace("json", "")

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected list of JSON objects")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM output: {text}") from e
