import json
from pathlib import Path

from tqdm import tqdm
from src.utils.data_loader import load_gsm8k
from src.utils.llm_client import render_template

SPLIT = "test"

def prepare_rl_jsonl(sources=["gsm8k"], output_path="data/rl_prompts.jsonl"):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = "templates/bias_detection/ZS-CoT"
    all_records = []

    for source in tqdm(sources):
        print(f"Processing source: {source}")
        data = load_gsm8k(source=source, split=SPLIT)
        for item in data:
            question = item["question"]
            prompt = render_template(templates_dir, "cot.txt", question=question)
            # We only give the question to the model, the answer is for the reward function
            record = {
                    "prompt": prompt,
                    "label": item["final_answer"]
                }
            all_records.append(record)

    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(all_records)} records to {output_path}")

if __name__ == "__main__":
    sources = [
        "gsm8k",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Korea/cultural_values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Moldova/cultural_values/test.jsonl",
        ]
    prepare_rl_jsonl(sources=sources)