# styler.py
"""
Step 7: Style-preserving final CoT regeneration.

- Use the templatized Q/A and recomputed numeric assignments.
- Apply per-unit style policies (numbers_only vs show_unit) for ANY unit.
- Produce the final augmented question and answer lines, with the last line '#### <answer>'.

Inputs:
    - templated: dict with 'templatized_question', 'templatized_answer', 'factual_assignment'
    - recompute_result: dict mapping placeholder -> numeric value (post-conversion+recompute)
    - original_answer: original CoT text (to help the LLM match style)
    - unit_policies: dict {unit_name: "show_unit" | "numbers_only"}
    - templates_dir: folder with text templates

Output (dict):
    {
    "question": <augmented_question_str>,
    "answer_lines": [<step1>, <step2>, ..., "#### <final>"],
    "final_scalar": <numeric_final>,
    "style_meta": {
        "unit_policies": {...},
        "notes": "LLM regenerated answer preserving style"
    }
    }
"""

import json
from typing import Any, Dict

from src.utils.llm_client import render_template


def replace_entity(
    chat,
    original_question: str,
    templates_dir: str,
    country: str = "Western",
) -> Dict[str, Any]:
    """
    Ask the LLM to render the final Q/A text, respecting style + unit policies.
    """

    # Build prompt
    prompt = render_template(
        templates_dir,
        f"replace_entities_with/{country}.txt",
        question=original_question,
    )

    resp = chat.invoke(prompt)
    text = resp.content.strip()
    text = text.replace("```json", "").replace("```", "")

    # Expected JSON object:
    # {
    #   "new_question": " ... ",
    # }
    try:
        data = json.loads(text)
        # Basic checks
        if not isinstance(data, dict):
            raise ValueError("Expected an object with augmented question")
        if "new_question" not in data :
            raise ValueError("Missing required keys in adapated output")
        # return {
        #     "new_question": data["new_question"],
        # }
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from western adapted LLM output: {text}") from e
