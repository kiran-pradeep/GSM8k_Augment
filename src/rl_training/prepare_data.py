import json
import random
import re
from pathlib import Path

from tqdm import tqdm

def prepare_rl_datasets(source_paths, output_dir="data/rl"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_records = []

    for path in tqdm(source_paths, desc="Processing files"):
        if not Path(path).exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    question = item.get("augmented_question", item.get("question"))
                    if not question: # Guard against missing questions
                        continue
                    
                    raw_answer = str(item.get("final_answer", item.get("gold_answer")))
                    # Clean currency, spaces, and trailing periods
                    clean_answer = re.sub(r"[,$₹]", "", raw_answer).strip().rstrip(".")
                    
                    if not clean_answer:
                        continue
                    float(clean_answer) # Ensure it's a valid number

                    prompt = f"""You are a helpful math reasoning assistant.
Solve the problem step by step, showing reasoning.

Question:
{question}

Output STRICT JSON:
{{
  "chain-of-thought-reasoning": "<step-by-step reasoning goes here>",
  "final_answer": <numeric answer only, no units, no extra text>
}}
"""
                    all_records.append({"prompt": prompt, "answer": clean_answer})
                except (ValueError, KeyError):
                    continue

    random.seed(42)
    random.shuffle(all_records)
    print(f"Loaded {len(all_records)} valid examples.")

    n = len(all_records)
    train_data = all_records[:int(n*0.9)]
    # val_data = all_records[int(n*0.8):int(n*0.9)]
    val_data = all_records[int(n*0.9):]
    # test_data = all_records[int(n*0.9):]

    for name, data in [("train", train_data), ("val", val_data)]:
        with open(f"{output_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    source_files = [
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/India/cultural_values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/India/values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/China/cultural_values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/China/values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Haiti/cultural_values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Haiti/values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Japan/cultural_values/test.jsonl",
        "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Japan/values/test.jsonl",
    ]
    prepare_rl_datasets(source_files)