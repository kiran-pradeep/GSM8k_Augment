# templatizer.py
"""
Step 5: LLM-based templatization of question and answer.

- Replace numeric values with placeholders.
- Produce a factual_assignment mapping placeholders -> numbers.
- Helps align with extracted metrics so conversions apply correctly.
"""

import json
from typing import Any, Dict, List

from utils.llm_client import render_template


def templatize_qa(chat, question: str, answer: str, variables: List[Dict[str, Any]], templates_dir: str) -> Dict[str, Any]:
    """
    Ask the LLM to templatize question and CoT answer.

    Expected JSON structure:
    {
      "templatized_question": "Betty is saving money for a wallet which costs {wallet_cost}.",
      "templatized_answer": [
        "In the beginning, Betty has only {wallet_cost} / 2 = {half_wallet}.",
        "Betty's grandparents gave her {parents_gift} * 2 = {grandparents_gift}.",
        "This means, Betty needs {wallet_cost} - {half_wallet} - {grandparents_gift} - {parents_gift} = {needed} more.",
        "#### {needed}"
      ],
      "factual_assignment": {
        "wallet_cost": 100,
        "half_wallet": 50,
        "parents_gift": 15,
        "grandparents_gift": 30,
        "needed": 5
      }
    }
    """
    prompt = render_template(
        templates_dir,
        "templatize.txt",
        question=question,
        answer=answer,
        variables=variables
    )
    resp = chat.invoke(prompt)
    text = resp.content.strip()
    text = text.replace("```json", "").replace("```", "")

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON dict")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM output: {text}") from e
