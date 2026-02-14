# src/bias_detection/cot.py
"""
Chain-of-Thought (CoT) reasoning.
"""
import json
import re
from typing import Dict

from utils.llm_client import get_chat_model, render_template

def extract_json_string(text: str) -> str:
    """
    Extract the JSON object from the given text using regex.
    Returns the JSON string or raises an error if not found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in model output.")

def sanitize_json_string(json_str: str) -> str:
    """
    Escape newlines and carriage returns inside JSON string values,
    so that the JSON parser can successfully decode it.
    Also removes other illegal control characters.
    """
    # First remove illegal control characters except \n, \r, \t
    json_str = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', json_str)
    
    # Then escape newlines inside quotes
    def escape_newlines(match):
        inner = match.group(1)
        inner = inner.replace('\n', '\\n').replace('\r', '\\r')
        return f'"{inner}"'
    
    # This regex matches strings inside double quotes (naive, but often works)
    json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_newlines, json_str)
    
    return json_str

def solve_with_cot(question: str, templates_dir: str = "templates") -> Dict[str, str]:
    """
    Solve using chain-of-thought prompting.

    Returns:
      {
        "chain-of-thought-reasoning": str,
        "final_answer": str
      }
    """
    chat = get_chat_model()

    # Render prompt
    prompt = render_template(templates_dir, "cot.txt", question=question)

    # Call model
    resp = chat.invoke(prompt)
    raw_text = resp.content.strip()

    # Extract text after </think>, if present
    if "</think>" in raw_text:
        text = raw_text.split("</think>")[1].strip()
    else:
        text = raw_text

    # Remove markdown formatting
    text = text.replace("```", "").replace("json", "").strip()

    # Try extracting and sanitizing JSON
    try:
        json_str = extract_json_string(text)
        clean_json_str = sanitize_json_string(json_str)
        data = json.loads(clean_json_str)

        try:
            cot = data["chain-of-thought-reasoning"]
        except KeyError:
            cot = None

        # In the data look for `final_answer` key. If not present, look for `answer` key. Else raise exception 
        if "final_answer" in data:
            final_answer = data["final_answer"]
        elif "answer" in data:
            final_answer = data["answer"]
        else:
            raise KeyError(f"Neither 'final_answer' nor 'answer' key found in JSON data: {raw_text}")
        
        pred = {}
        pred["chain-of-thought-reasoning"] = cot
        pred["final_answer"] = final_answer
        pred["raw_text"] = raw_text
        pred["digit_question"] = data["converted_question"] if "converted_question" in data else None

        return pred

    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM output. Raw output:\n{text}") from e

