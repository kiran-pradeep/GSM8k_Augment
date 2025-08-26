# converter.py
"""
Step 4: Conversion of extracted values to SI/Indian units.

- Use LLM to generate a Python function that converts each value.
- Execute the code in a restricted namespace.
- Return structured results (list of dicts).
"""

import json
import traceback
from typing import Any, Dict, List

from utils.llm_client import render_template


def generate_conversion_code(chat, extracted_metrics: List[Dict[str, Any]], templates_dir: str) -> str:
    """
    Ask the LLM to generate Python code that converts extracted values into SI/Indian equivalents.

    The extracted_metrics list has elements like:
      {"variable": "wallet_cost", "value": 100, "unit": "dollar", "context": "question"}

    The LLM should output only valid Python code that defines a function `convert_units()`
    which returns a list of dicts:
      [
        {"variable": "wallet_cost", "original_value": 100, "original_unit": "dollar",
         "converted_value": 8300, "converted_unit": "rupee"},
        ...
      ]
    """
    prompt = render_template(
        templates_dir,
        "conversion_code.txt",
        extracted=json.dumps(extracted_metrics, indent=2)
    )
    resp = chat.invoke(prompt)
    return resp.content.strip().replace("```python", "").replace("```", "")


def run_conversion_code_safely(code: str, metrics_list: list) -> list:
    safe_globals = {}
    safe_locals = {}
    try:
        exec(code, safe_globals, safe_locals)
        if "convert_units" not in safe_locals:
            raise RuntimeError("LLM code did not define convert_units()")
        # Pass metrics_list to the function
        result = safe_locals["convert_units"](metrics_list)
        if not isinstance(result, list):
            raise RuntimeError("convert_units() did not return a list")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Error running conversion code:\n{tb}\nCode:\n{code}") from e
