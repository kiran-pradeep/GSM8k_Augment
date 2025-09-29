"""
cultural_adapter.py

Step 8: Replace culturally non-Indian entities (names, places, etc.)
with Indian equivalents in the FINAL augmented Q/A.

Inputs:
    - question (str)
    - answer_lines (list of str)
    - original_answer (str)
    - cultural_check (dict: output from cultural_filter)

Outputs:
    dict with:
    {
        "adapted_question": str,
        "adapted_answer": list of str,
        "adapted_final_scalar": number,
        "adaptation_notes": str
    }
"""

import json
from typing import Dict, Any, List
from src.utils.llm_client import render_template


def adapt_cultural_entities(
    chat,
    question: str,
    answer_lines: List[str],
    cultural_check: Dict[str, Any],
    templates_dir: str
) -> Dict[str, Any]:
    """Replace non-Indian cultural entities with Indian equivalents."""

    prompt = render_template(
        templates_dir,
        "cultural_adapter.txt",
        question=question,
        answer="\n".join(answer_lines),
        cultural_check=json.dumps(cultural_check, indent=2, ensure_ascii=False)
    )

    resp = chat.invoke(prompt)
    text = resp.content.strip()
    text = text.replace("```json", "").replace("```", "")

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object")
        return {
            "adapted_question": data["adapted_question"],
            "adapted_answer": data["adapted_answer"],
            "adaptation_notes": data.get("adaptation_notes", "")
        }
    except Exception as e:
        prompt = f"Correct the following JSON:\n{text}\n\nOnly return valid JSON.\n"
        resp = chat.invoke(prompt)
        text = resp.content.strip()
        text = text.replace("```json", "").replace("```", "")
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
            return {
                "adapted_question": data["adapted_question"],
                "adapted_answer": data["adapted_answer"],
                "adaptation_notes": data.get("adaptation_notes", "")
            }
        except Exception as e2:
            raise RuntimeError(f"Failed to parse JSON from cultural adapter LLM output: {text}") from e2
