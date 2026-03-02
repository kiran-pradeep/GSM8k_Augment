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
            max_tokens=1024,
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

    if "FewS" in str(templates_dir):
        country = os.getenv("COUNTRY", "gsm8k")
        no_examples = int(os.getenv("FEWSHOT_EXAMPLES", "5"))
        if country == "unknown":
            print(f"[WARN] COUNTRY environment variable not set. Using 'gsm8k' as default for FewS-CoT examples.")
            country = "gsm8k"
        with open("templates/bias_detection/_examples/_examples.json", "r", encoding="utf-8") as f:
            import json
            all_examples = json.load(f)

        examples = all_examples.get(country, [])
        example_text = ""
        for ex in examples[:no_examples]:
            """
            Output Format to be appended to the example_text string is Question followed by a JSON object containing the CoT reasoning and the final answer. For example:
            Question:
            <question text>
            {
                "chain-of-thought-reasoning": "the full answer goes here",
                "final_answer": <The last line of the answer, it starts with #### followed by the answer itself>
            }
            """
            question = ex["question"]
            answer = ex["answer"]
            final_answer = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
            if "CoT" in templates_dir:
                example_text += f"Question:\n{question}\n{{\n  \"chain-of-thought-reasoning\": \"{answer}\",\n  \"final_answer\": \"{final_answer}\"\n}}\n\n"
            else:
                example_text += f"Question:\n{question}\n{{\n  \"answer\": \"{final_answer}\"\n}}\n\n"

        kwargs["examples"] = example_text.strip()


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

