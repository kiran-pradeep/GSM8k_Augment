"""
cultural_filter.py

Step 0: Check whether Q/A contains culturally specific references 
to regions other than India. 

Returns:
    {
        "is_cultural": bool,
        "matched_entities": [list of strings],
        "notes": str
    }
"""

import json
from typing import Dict, Any
from src.utils.llm_client import render_template

def check_cultural_bias(chat, question: str, answer: str, templates_dir: str) -> Dict[str, Any]:
    """Use LLM to classify cultural specificity."""

    prompt = render_template(
        templates_dir,
        "cultural_filter.txt",
        question=question,
        answer=answer
    )

    resp = chat.invoke(prompt)
    text = resp.content.strip()
    text = text.replace("```json", "").replace("```", "")

    try:
        result = json.loads(text)
        # Ensure structure
        return {
            "is_cultural": bool(result.get("is_cultural", False)),
            "matched_entities": result.get("matched_entities", []),
            "notes": result.get("notes", "")
        }
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from cultural filter LLM output: {text}") from e
