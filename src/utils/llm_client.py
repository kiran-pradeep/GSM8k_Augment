# llm_client.py
"""
LLM client wrapper and template rendering utilities.

- Provides get_chat_model() which connects to your vLLM backend.
- Provides render_template() for loading + filling prompt templates.
"""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI  # pip install langchain-openai
from langchain_google_genai import ChatGoogleGenerativeAI # pip install langchain-google-genai
from dotenv import load_dotenv
# Load .env file
load_dotenv()

# ---------------- LLM Connector ---------------- #

def get_chat_model():
    """
    Initialize a LangChain ChatOpenAI client pointing to your vLLM server.
    Requires env vars:
      VLLM_BASE_URL : base URL of the vLLM API (e.g. http://localhost:8000/v1)
      VLLM_API_KEY  : API key if authentication is enabled, else "EMPTY"
      VLLM_MODEL    : model name string (must match your vLLM model id)

    Returns: ChatOpenAI object
    """
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    model_name = os.getenv("VLLM_MODEL", "gpt-4")  # default fallback

    if 'gemini' in model_name:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_api_key,
                temperature=0,
                # convert_system_message_to_human=True, # Helps with compatibility for some chat prompts
            )
    else:
        return ChatOpenAI(
            model=model_name,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0,
        )


# ---------------- Template Renderer ---------------- #

def render_template(templates_dir: str, filename: str, **kwargs) -> str:
    """
    Load a template file and substitute only {{var}} placeholders.
    Literal { and } inside template (like in JSON examples) are left untouched.
    """
    path = Path(templates_dir) / filename
    if not path.exists():
        print(f"[ERROR] Template not found: {path}")
        raise FileNotFoundError(f"Template not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Replace our {{var}} with str.format() placeholders
    for k, v in kwargs.items():
        text = text.replace(f"{{{{{k}}}}}", str(v))


    # Common variables for all templates
    # These can be set in .env file or will use the defaults below
    common_vars = {
        "times": os.getenv("TIMES_MULTIPLIER", 10.0)
    }

    for k, v in common_vars.items():
        text = text.replace(f"{{{{{k}}}}}", str(v))

    return text

