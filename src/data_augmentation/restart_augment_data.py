# src/data_augmentation/augment_data.py
"""
Main orchestrator for GSM8K metric-conversion augmentation.
Restartable from any item index. Checks for existing intermediates in the given output dir.

- Multiprocessing pool with configurable workers.
- Progressive saving of intermediates even if later steps fail.
- Timestamped output directories: out/{VLLM_MODEL}/{IST-timestamp}/...

Pipeline:
1) Load GSM8k dataset item
2) Extract Q + A (with CoT)
3) LLM-based metric extraction (dynamic units & quantities)
4) LLM-generated conversion code -> execute -> structured conversions
5) LLM-based templatization (Q, CoT, mapping)
6) LLM-generated recomputation code for new values -> execute safely
7) Style-preserving final CoT + save outputs

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
# from cultural_adapter import adapt_cultural_entities
from src.data_augmentation.cultural_adapter import adapt_cultural_entities
from src.data_augmentation.cultural_filter import check_cultural_bias
from src.data_augmentation.extractor import extract_metrics_llm
from src.data_augmentation.converter import generate_conversion_code, run_conversion_code_safely
from src.data_augmentation.templatizer import templatize_qa
from src.data_augmentation.recomputer import generate_recompute_code, run_recompute_code_safely
from src.data_augmentation.styler import style_cot_answer
from src.utils.llm_client import get_chat_model
from src.utils.data_loader import load_gsm8k
from src.utils.io_utils import ensure_dir, dump_json, append_jsonl, log_error


import json
import nltk
import os
from nltk.corpus import stopwords

nltk.download('stopwords')
english_stopwords = set(stopwords.words("english"))
nltk.download('stopwords')


# ---------------- Helper functions ---------------- #

load_dotenv()


def build_intermediate_record(
    idx: int,
    split: str,
    question: str,
    answer: str,
    cultural_check: Dict[str, Any] = None,
    metrics_extracted: List[Dict[str, Any]] = None,
    conversion_code: str = None,
    conversions_result: List[Dict[str, Any]] = None,
    merged_assignment: Dict[str, str] = None,
    templated: Dict[str, Any] = None,
    recompute_code: str = None,
    recompute_result: Dict[str, Any] = None,
    final: Dict[str, Any] = None,
    cultural_adapted: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Structure for saving intermediate JSON for inspection (progressive)."""
    return {
        "index": idx,
        "split": split,
        "original": {"question": question, "answer": answer},
        "cultural_check": cultural_check,
        "metrics_extracted": metrics_extracted,
        "conversion": {"code": conversion_code, "results": conversions_result},
        "templatization": templated,
        "merged_assignment": merged_assignment,
        "recompute": {"code": recompute_code, "results": recompute_result},
        "final": final,
        "cultural_adapted": cultural_adapted,
    }


def merge_conversions_with_assignments(factual_assignment, conversions_result):
    factual_assignment = dict(factual_assignment or {})
    for conv in conversions_result or []:
        var = conv.get("variable")
        conv_val = conv.get("converted_value")
        if var and var in factual_assignment:
            factual_assignment[var] = conv_val
    return factual_assignment


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
    i, row, args, intermediate_dir, restarted_path = args_tuple
    question = row["question"].strip()
    answer = row["answer"].strip()

    # Init LLM client per process
    chat = get_chat_model()

    with open(intermediate_dir / f"{i}.json", "r") as f:
        try:
            intermediate_record = json.load(f)
        except:
            print(f"[WARN] Corrupted intermediate JSON for idx={i}, rebuilding")
            intermediate_record = build_intermediate_record(
                idx=i, split=args.split, question=question, answer=answer
            )

    try:
        # Step 0: cultural filter
        if "cultural_check" not in intermediate_record or not intermediate_record["cultural_check"]:
            print(f"[INFO] idx={i} running cultural check")
            cultural_check = check_cultural_bias(
                chat=chat,
                question=question,
                answer=answer,
                templates_dir=args.templates
            )
            intermediate_record["cultural_check"] = cultural_check
            dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        cultural_check = intermediate_record["cultural_check"]

        # Step 1: metric extraction
        if "metrics_extracted" not in intermediate_record or not intermediate_record["metrics_extracted"]:
            print(f"[INFO] idx={i} running metric extraction")
            metrics_extracted = extract_metrics_llm(
                chat=chat, question=question, answer=answer, templates_dir=args.templates
            )
            intermediate_record["metrics_extracted"] = metrics_extracted
            dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        metrics_extracted = intermediate_record["metrics_extracted"]

        # Step 2: conversion
        if "conversion" not in intermediate_record or not intermediate_record["conversion"]["code"]:
            print(f"[INFO] idx={i} running conversion")
            conversion_code = generate_conversion_code(
                chat=chat, extracted_metrics=metrics_extracted, templates_dir=args.templates
            )
            conversions_result = run_conversion_code_safely(conversion_code, metrics_extracted)
            intermediate_record["conversion"] = {"code": conversion_code, "results": conversions_result}
            dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        conversion_code = intermediate_record["conversion"]["code"]
        conversions_result = intermediate_record["conversion"]["results"]

        # Step 3: templatize
        if "templatization" not in intermediate_record or not intermediate_record["templatization"]:
            print(f"[INFO] idx={i} running templatization")
            templated = templatize_qa(
                chat=chat,
                question=question,
                answer=answer,
                variables=metrics_extracted,
                templates_dir=args.templates,
            )
            intermediate_record["templatization"] = templated
            dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        templated = intermediate_record["templatization"]

        # Step 4: merge + recompute
        if "merged_assignment" not in intermediate_record or not intermediate_record["merged_assignment"]:
            print(f"[INFO] idx={i} running merging assignments")
            factual_assignment = remove_braces_from_keys(templated.get("factual_assignment", {}))
            merged_assignment = merge_conversions_with_assignments(factual_assignment, conversions_result)
            intermediate_record["merged_assignment"] = merged_assignment
            dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        
        factual_assignment = remove_braces_from_keys(templated.get("factual_assignment", {}))
        merged_assignment = merge_conversions_with_assignments(factual_assignment, conversions_result)

        if "recompute" not in intermediate_record or not intermediate_record["recompute"]["code"]:
            print(f"[INFO] idx={i} running recomputation")
            recompute_code = generate_recompute_code(
                chat=chat,
                templated={
                    "templatized_question": templated["templatized_question"],
                    "templatized_answer": templated["templatized_answer"],
                },
                converted_assignment=merged_assignment,
                templates_dir=args.templates,
            )
            recompute_result = run_recompute_code_safely(chat, recompute_code)
            intermediate_record["recompute"] = {"code": recompute_code, "results": recompute_result}
            dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        recompute_code = intermediate_record["recompute"]["code"]
        recompute_result = intermediate_record["recompute"]["results"]

        # Step 5: styling
        if "final" not in intermediate_record or not intermediate_record["final"]:
            print(f"[INFO] idx={i} running final styling")
            final = style_cot_answer(
                chat=chat,
                templated=templated,
                recompute_result=recompute_result,
                conversions=conversions_result,
                original_question=question,
                original_answer=answer,
                unit_policies=None,
                templates_dir=args.templates,
            )
            intermediate_record["final"] = final
            dump_json(intermediate_dir / f"{i}.json", intermediate_record)
        final = intermediate_record["final"]

        # Step 6: Adapting to specific cultural context
        cultural_adapted = adapt_cultural_entities(
            chat=chat,
            question=final["question"],
            answer_lines=final["answer_lines"],
            cultural_check=cultural_check["matched_entities"] if cultural_check["is_cultural"] else "None",
            templates_dir=args.templates
        )
        intermediate_record["cultural_adapted"] = cultural_adapted
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)


        # Save augmented JSONL
        augmented_record = {
            "index": i,
            "split": args.split,
            "is_cultural": cultural_check["is_cultural"],
            "augmented_question": cultural_adapted["adapted_question"],
            "augmented_answer": cultural_adapted["adapted_answer"],
            "final_answer": final["final_scalar"],
            # "style": final["style_meta"],
        }
        # append_jsonl(augmented_path, augmented_record)
        append_jsonl(restarted_path, augmented_record)


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
        # log_error(intermediate_dir.parent, i, str(e), tb)
        log_error(restarted_path.parent, i, str(e), tb)


def is_not_english(text):
    words = text.lower().split()
    overlap = sum(1 for w in words if w in english_stopwords)
    return overlap / max(len(words), 1) < 0.2  # less than 20% English stopwords


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="main", choices=["main", "socratic"], help="Which prompt config to use.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split to process.")
    parser.add_argument("--limit", type=int, default=-1, help="-1 for all, else max number of items to process.")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based).")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (LLM clients).")
    parser.add_argument("--templates", default="templates/data_augmentation", help="Directory with prompt templates.")
    parser.add_argument("--failfast", action="store_true", help="Whether to stop on first error.")
    parser.add_argument("--country", type=str, default="India", help="Country for cultural adaptation.")
    parser.add_argument("--demonym", type=str, default="Indian", help="Demonym for cultural adaptation.")
    parser.add_argument("--currency", type=str, default="rupee", help="Currency name for cultural adaptation.")
    parser.add_argument("--currency_symbol", type=str, default="₹", help="Currency symbol for cultural adaptation.")
    parser.add_argument("--currency_conversion_rate", type=float, default=87.0, help="Conversion rate to 1 USD.")
    parser.add_argument("--currency_abbreviation", type=str, default="INR", help="Currency abbreviation.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")

    args = parser.parse_args()

    os.environ["COUNTRY"] = args.country
    os.environ["DEMONYM"] = args.demonym
    os.environ["CURRENCY"] = args.currency
    os.environ["CURRENCY_SYMBOL"] = args.currency_symbol
    os.environ["CURRENCY_CONVERSION_RATE"] = str(args.currency_conversion_rate)
    os.environ["CURRENCY_ABBREVIATION"] = args.currency_abbreviation 

    print(f"[INFO] Using config: {args.config}, split: {args.split}, start: {args.start}, limit: {args.limit}, workers: {args.workers}")
    print(f"[INFO] Cultural context: country={args.country}, demonym={args.demonym}, currency={args.currency}, currency_symbol={args.currency_symbol}, currency_conversion_rate={args.currency_conversion_rate}, currency_abbreviation={args.currency_abbreviation}")
    print(f"[INFO] Templates dir: {args.templates}")
    print(f"[INFO] Failfast: {args.failfast}")  

    # Make timestamped output directory
    model_name = os.getenv("VLLM_MODEL", "NA").replace("/", "_").replace(".", "_")
    if "snapshot" in model_name:
        model_name = model_name.split("_snapshots")[0]
        model_name = model_name.split("_models--")[1]
    # ist = pytz.timezone("Asia/Kolkata")
    # timestamp = datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    # outdir = Path("out") / "augmented_data"/ model_name / timestamp
    outdir = Path(args.output_dir)
    intermediate_dir = outdir / "intermediate" / args.split
    augmented_path = outdir / "cultural_values" / f"{args.split}.jsonl"
    restarted_path = outdir / "english_restarted" / f"{args.split}.jsonl"
    ensure_dir(intermediate_dir)
    ensure_dir(augmented_path.parent)
    ensure_dir(restarted_path.parent)

    # Create a list of already processed indices to skip from the augmented_path
    error_indices = set()
    if augmented_path.exists():
        with open(augmented_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if is_not_english(record["augmented_question"]):
                        error_indices.add(record["index"])
                except json.JSONDecodeError:
                    continue
    print(f"[INFO] Found {len(error_indices)} already processed items in {augmented_path}")

    # Load dataset
    ds = load_gsm8k(args.config, args.split)
    n = len(ds)
    if args.limit == -1:
        end = n
    else:
        end = min(args.start + args.limit, n)
    print(f"[INFO] Processing total {end - args.start - len(error_indices)} items from index {args.start} to {end-1} (out of {n})")

    tasks = [
        (i, ds[i], args, intermediate_dir, restarted_path)
        for i in range(args.start, end) if i in error_indices
    ]

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for _ in tqdm(pool.imap_unordered(process_item, tasks), total=len(tasks), desc="Processing items"):
                pass
    else:
        for t in tqdm(tasks, desc="Processing items"):
            process_item(t)


    print(f"[DONE] Augmented JSONL: {restarted_path}")
    print(f"[DONE] Intermediates in: {intermediate_dir}")


if __name__ == "__main__":
    main()

