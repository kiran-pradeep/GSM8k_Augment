# recomputer.py
"""
Step 6: LLM-based recomputation with converted values.

- Generate Python code that uses the converted factual_assignment.
- Execute safely.
- Return updated assignments with recomputed values.
"""

import json
import traceback
from typing import Any, Dict

from src.utils.llm_client import render_template


def generate_recompute_code(
        chat, 
        templated: Dict[str, Any], 
        converted_assignment: Dict[str, Any], 
        templates_dir: str
    ) -> str:
    """
    Ask the LLM to generate Python code that recomputes all placeholders
    in the templatized answer using the converted_assignment values.

    The code must define a function `recompute()` returning a dict
    mapping placeholder -> numeric value.
    """
    prompt = render_template(
        templates_dir,
        "recompute_code.txt",
        templatized=json.dumps(templated, indent=2),
        converted=json.dumps(converted_assignment, indent=2)
    )
    resp = chat.invoke(prompt)
    return resp.content.strip().replace("```python", "").replace("```", "")



def run_recompute_code_safely(chat, code: str) -> Dict[str, Any]:
    """
    Execute the LLM-generated recomputation code in a restricted environment.

    The code must define a function recompute() returning a dict.
    """
    safe_globals = {}
    safe_locals = {}
    try:
        exec(code, safe_globals, safe_locals)
        if "recompute" not in safe_locals:
            raise RuntimeError("LLM code did not define recompute()")
        result = safe_locals["recompute"]()
        if not isinstance(result, dict):
            raise RuntimeError("recompute() did not return a dict")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        prompt = f"Fix the following Python code so that it runs without errors:\n{tb}\nCode:\n{code}\nOnly return the corrected code."
        resp = chat.invoke(prompt)
        code = resp.content.strip().replace("```python", "").replace("```", "")
        try:
            safe_globals = {}
            safe_locals = {}
            exec(code, safe_globals, safe_locals)
            if "recompute" not in safe_locals:
                raise RuntimeError("LLM code did not define recompute()")
            result = safe_locals["recompute"]()
            if not isinstance(result, dict):
                raise RuntimeError("recompute() did not return a dict")
            return result
        except Exception as e2:
            raise RuntimeError(f"Error running recompute code:\n{tb}\nCode:\n{code}") from e2
