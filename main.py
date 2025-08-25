#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main orchestrator for GSM8K metric-conversion augmentation.

Pipeline:
1) Load GSM8k dataset item
2) Extract Q + A (with CoT)
3) LLM-based metric extraction (dynamic units & quantities)
4) LLM-generated conversion code -> execute -> structured conversions
5) LLM-based templatization (Q, CoT, mapping)
6) LLM-generated recomputation code for new values -> execute safely
7) Style-preserving final CoT (unit policy) + save outputs

Environment:
- VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL

Outputs:
- out/intermediate/{split}/{idx}.json
- out/augmented/{split}.jsonl

Now supports:
- Multiprocessing pool with configurable workers.
- Progressive saving of intermediates even if later steps fail.
- Timestamped output directories: out/{VLLM_MODEL}/{IST-timestamp}/...
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

# ---- Local modules ----
from cultural_adaptor import adapt_cultural_entities
from cultural_filter import check_cultural_bias
from utils.llm_client import get_chat_model
from utils.data_loader import load_gsm8k
from extractor import extract_metrics_llm
from converter import generate_conversion_code, run_conversion_code_safely
from templatizer import templatize_qa
from recomputer import generate_recompute_code, run_recompute_code_safely
from styler import style_cot_answer
from utils.io_utils import ensure_dir, dump_json, append_jsonl, log_error


# ---------------- Helper functions ---------------- #

load_dotenv()


def build_intermediate_record(
    idx: int,
    split: str,
    question: str,
    answer: str,
    metrics_extracted: List[Dict[str, Any]] = None,
    conversion_code: str = None,
    conversions_result: List[Dict[str, Any]] = None,
    merged_assignment: Dict[str, str] = None,
    templated: Dict[str, Any] = None,
    recompute_code: str = None,
    recompute_result: Dict[str, Any] = None,
    final: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Structure for saving intermediate JSON for inspection (progressive)."""
    return {
        "index": idx,
        "split": split,
        "original": {"question": question, "answer": answer},
        "metrics_extracted": metrics_extracted,
        "conversion": {"code": conversion_code, "results": conversions_result},
        "templatization": templated,
        "merged_assignment": merged_assignment,
        "recompute": {"code": recompute_code, "results": recompute_result},
        "final": final,
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
    i, row, args, intermediate_dir, augmented_path = args_tuple
    question = row["question"].strip()
    answer = row["answer"].strip()

    # Init LLM client per process
    chat = get_chat_model()

    intermediate_record = build_intermediate_record(
        idx=i, split=args.split, question=question, answer=answer
    )

    try:
        # Step 0: cultural filter
        cultural_check = check_cultural_bias(
            chat=chat,
            question=question,
            answer=answer,
            templates_dir=args.templates
        )
        intermediate_record["cultural_check"] = cultural_check
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)

        # Step 1: metric extraction
        metrics_extracted = extract_metrics_llm(
            chat=chat, question=question, answer=answer, templates_dir=args.templates
        )
        intermediate_record["metrics_extracted"] = metrics_extracted
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)

        # Step 2: conversion
        conversion_code = generate_conversion_code(
            chat=chat, extracted_metrics=metrics_extracted, templates_dir=args.templates
        )
        conversions_result = run_conversion_code_safely(conversion_code, metrics_extracted)
        intermediate_record["conversion"] = {"code": conversion_code, "results": conversions_result}
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)

        # Step 3: templatize
        templated = templatize_qa(
            chat=chat,
            question=question,
            answer=answer,
            variable_names=[info["variable"] for info in conversions_result],
            templates_dir=args.templates,
        )
        intermediate_record["templatization"] = templated
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)

        # Step 4: merge + recompute
        factual_assignment = remove_braces_from_keys(templated.get("factual_assignment", {}))
        merged_assignment = merge_conversions_with_assignments(factual_assignment, conversions_result)
        intermediate_record["merged_assignment"] = merged_assignment
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)

        recompute_code = generate_recompute_code(
            chat=chat,
            templated={
                "templatized_question": templated["templatized_question"],
                "templatized_answer": templated["templatized_answer"],
            },
            converted_assignment=merged_assignment,
            templates_dir=args.templates,
        )
        recompute_result = run_recompute_code_safely(recompute_code)
        intermediate_record["recompute"] = {"code": recompute_code, "results": recompute_result}
        dump_json(intermediate_dir / f"{i}.json", intermediate_record)

        # Step 5: styling
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

        # Step 6: Adapting to specific cultural context
        cultural_adapted = adapt_cultural_entities(
            chat=chat,
            question=final["question"],
            answer_lines=final["answer_lines"],
            cultural_check=cultural_check["matched_entities"] if cultural_check["is_cultural"] else "None",
            templates_dir=args.templates
        )

        # Save augmented JSONL
        augmented_record = {
            "index": i,
            "split": args.split,
            "cultural_check": cultural_check,
            "augmented_question": cultural_adapted["question"],
            "augmented_answer": cultural_adapted["answer_lines"],
            "final_answer": final["final_scalar"],
            # "style": final["style_meta"],
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
    parser.add_argument("--config", default="main", choices=["main", "socratic"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--failfast", action="store_true")
    args = parser.parse_args()

    # Make timestamped output directory
    model_name = os.getenv("VLLM_MODEL", "NA").replace("/", "_").replace(".", "_")
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    outdir = Path("out") / model_name / timestamp
    intermediate_dir = outdir / "intermediate" / args.split
    augmented_path = outdir / "augmented" / f"{args.split}.jsonl"
    ensure_dir(intermediate_dir)
    ensure_dir(augmented_path.parent)

    # Load dataset
    ds = load_gsm8k(args.config, args.split)
    n = len(ds)
    end = min(args.start + args.limit, n)
    print(f"[INFO] Processing {args.split} from {args.start} to {end-1} (total {n})")

    tasks = [
        (i, ds[i], args, intermediate_dir, augmented_path)
        for i in range(args.start, end)
    ]

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            pool.map(process_item, tasks)
    else:
        for t in tasks:
            process_item(t)

    print(f"[DONE] Augmented JSONL: {augmented_path}")
    print(f"[DONE] Intermediates in: {intermediate_dir}")


if __name__ == "__main__":
    main()

