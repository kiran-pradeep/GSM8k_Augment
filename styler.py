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

from utils.llm_client import render_template


def style_cot_answer(
    chat,
    templated: Dict[str, Any],
    recompute_result: Dict[str, Any],
    conversions: list,
    original_question: str,
    original_answer: str,
    unit_policies: Dict[str, str],
    templates_dir: str
) -> Dict[str, Any]:
    """
    Ask the LLM to render the final Q/A text, respecting style + unit policies.
    """

    # Build prompt
    prompt = render_template(
        templates_dir,
        "style_answer.txt",
        templatized=json.dumps(templated, indent=2, ensure_ascii=False),
        recomputed=json.dumps(recompute_result, indent=2, ensure_ascii=False),
        conversions=json.dumps(conversions, indent=2, ensure_ascii=False),
        original_question=original_question,
        original_answer=original_answer,
        unit_policies=json.dumps(unit_policies, indent=2, ensure_ascii=False)
    )

    resp = chat.invoke(prompt)
    text = resp.content.strip()
    text = text.replace("```json", "").replace("```", "")

    # Expected JSON object:
    # {
    #   "question": " ... ",
    #   "answer_lines": ["...", "...", "#### 415"],
    #   "final_scalar": 415
    # }
    try:
        data = json.loads(text)
        # Basic checks
        if not isinstance(data, dict):
            raise ValueError("Expected an object with question, answer_lines, final_scalar")
        if "question" not in data or "answer_lines" not in data or "final_scalar" not in data:
            raise ValueError("Missing required keys in styler output")
        lines = data["answer_lines"]
        if not lines or not isinstance(lines, list):
            raise ValueError("answer_lines must be a non-empty list")
        last = lines[-1]
        if not isinstance(last, str) or not last.strip().startswith("#### "):
            raise ValueError("The last line must be in the format '#### <answer>'")
        return {
            "question": data["question"],
            "answer_lines": data["answer_lines"],
            "final_scalar": data["final_scalar"],
            "style_meta": {
                "unit_policies": unit_policies,
                "notes": "LLM regenerated answer preserving original style and unit policies"
            }
        }
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from styler LLM output: {text}") from e
