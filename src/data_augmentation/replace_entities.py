# src/data_augmentation/augment_data.py
"""
Main orchestrator for GSM8K metric-conversion augmentation.

- Multiprocessing pool with configurable workers.
- Progressive saving of intermediates even if later steps fail.
- Timestamped output directories: out/{VLLM_MODEL}/{IST-timestamp}/...

Pipeline:
1) Load GSM8k dataset item
2) Extract Q + A (with CoT)
3) LLM-based augmentation to replace western names with other western names

Environment:
- VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL

Outputs:
- out/intermediate/{split}/{idx}.json
- out/augmented/{split}.jsonl

"""

from __future__ import annotations
import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from multiprocessing import Pool
from datetime import datetime
import pytz
from tqdm import tqdm

# ---- Local modules ----
from src.data_augmentation.replace_with_country import replace_entity
from src.utils.llm_client import get_chat_model
from src.utils.data_loader import load_gsm8k
from src.utils.io_utils import ensure_dir, dump_json, append_jsonl, log_error

# ---------------- Helper functions ---------------- #

load_dotenv()


def build_intermediate_record(
    idx: int,
    split: str,
    question: str,
    answer: str,
    final: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Structure for saving intermediate JSON for inspection (progressive)."""
    return {
        "index": idx,
        "split": split,
        "original": {"question": question, "answer": answer},
        "final": final,
    }


def remove_braces_from_keys(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(key, str) and key.startswith("{") and key.endswith("}"):
            new_key = key[1:-1]
        else:
            new_key = key
        new_dict[new_key] = value
    return new_dict


# ---------------- Worker ---------------- #

def process_item(args_tuple):
    i, row, args, intermediate_dir, augmented_path = args_tuple
    question = row["question"].strip()
    answer = row["answer"].strip()
    final_answer = row["final_answer"].strip()

    # Init LLM client per process
    chat = get_chat_model()

    intermediate_record = build_intermediate_record(
        idx=i, split=args.split, question=question, answer=answer
    )

    try:
        final = replace_entity(
            chat=chat,
            original_question=question,
            templates_dir=args.templates,
            country=args.country,
        )
        intermediate_record["final"] = final
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)

        # Save augmented JSONL
        augmented_record = {
            "index": i,
            "split": args.split,
            "augmented_question": final["new_question"],
            "augmented_answer": answer,
            "final_answer": final_answer,
        }
        append_jsonl(augmented_path, augmented_record)

        print(f"[OK] idx={i} saved")
    except Exception as e:
        print(traceback.format_exc())
        print(f"[ERROR] idx={i}: {e}")
        # still save partial progress
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        if args.failfast:
            raise
        tb = traceback.format_exc()
        print(tb)
        print(f"[ERROR] idx={i}: {e}")
        log_error(intermediate_dir.parent, i, str(e), tb)


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="main", choices=["main", "socratic"], help="Which prompt config to use.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split to process.")
    parser.add_argument("--limit", type=int, default=-1, help="-1 for all, else max number of items to process.")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based).")
    parser.add_argument("--country", type=str, default="India", help="Country style to replace with.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (LLM clients).")
    parser.add_argument("--templates", default="templates/data_augmentation", help="Directory with prompt templates.")
    parser.add_argument("--failfast", action="store_true", help="Whether to stop on first error.")

    args = parser.parse_args()

    # os.environ["TIMES_MULTIPLIER"] = args.times

    print(f"[INFO] Using config: {args.config}, split: {args.split}, start: {args.start}, limit: {args.limit}, workers: {args.workers}")
    print(f"[INFO] Country style: {args.country}")
    print(f"[INFO] Templates dir: {args.templates}")
    print(f"[INFO] Failfast: {args.failfast}")  

    # Make timestamped output directory
    model_name = os.getenv("VLLM_MODEL", "NA").replace("/", "_").replace(".", "_")
    if "snapshot" in model_name:
        model_name = model_name.split("_snapshots")[0]
        model_name = model_name.split("_models--")[1]
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    outdir = Path("out") / "replace_entities"/ args.country / model_name / timestamp
    intermediate_dir = outdir / "intermediate" / args.split
    augmented_path = outdir / "augmented" / f"{args.split}.jsonl"
    ensure_dir(intermediate_dir)
    ensure_dir(augmented_path.parent)

    # Load dataset
    ds = load_gsm8k(args.config, args.split)
    n = len(ds)
    if args.limit == -1:
        end = n
    else:
        end = min(args.start + args.limit, n)
    print(f"[INFO] Processing {args.split} from {args.start} to {end-1} (total {n})")

    tasks = [
        (i, ds[i], args, intermediate_dir, augmented_path)
        for i in range(args.start, end)
    ]

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for _ in tqdm(pool.imap_unordered(process_item, tasks), total=len(tasks), desc="Processing items"):
                pass
    else:
        for t in tqdm(tasks, desc="Processing items"):
            process_item(t)


    print(f"[DONE] Augmented JSONL: {augmented_path}")
    print(f"[DONE] Intermediates in: {intermediate_dir}")


if __name__ == "__main__":
    main()

