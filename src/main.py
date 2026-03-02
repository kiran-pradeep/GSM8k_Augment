# main.py
"""
Main runner for CoT evaluation over GSM8k (or a custom JSONL source).

Features:
- Multiprocessing with configurable workers
- Progressive saving of per-instance JSON into intermediate/<split>/<index>.json
- Aggregation of final stats into -final_results.json
- Error logging.
"""

from __future__ import annotations
import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
from multiprocessing import Pool
import pytz
from tqdm import tqdm
from dotenv import load_dotenv

# Local imports
from bias_detection.cot import solve_with_cot
from bias_detection.evaluator import evaluate_instance, evaluate_all
from utils.data_loader import load_gsm8k
from utils.io_utils import ensure_dir, dump_json, append_jsonl, log_error

load_dotenv()


def build_result_record(
    idx: int,
    split: str,
    question: str,
    gold_cot: str,
    gold_answer: float,
    prediction: Dict[str, Any] = None,
    evaluation: Dict[str, Any] = None,
    error: str | None = None
) -> Dict[str, Any]:
    """Structure for saving per-instance intermediate JSON."""
    return {
        "index": idx,
        "split": split,
        "question": question,
        "gold_cot": gold_cot,
        "gold_answer": gold_answer,
        "pred_cot": prediction.get("chain-of-thought-reasoning") if prediction else None,
        "pred_answer": prediction.get("final_answer") if prediction else None,
        "evaluation": evaluation,
        "error": error,
        "time": datetime.now().astimezone(pytz.timezone("Asia/Kolkata")).isoformat(),
    }


def process_item(args_tuple: Tuple[int, Dict[str, Any], argparse.Namespace, Path, Path]) -> Dict[str, Any]:
    """
    Worker function called in multiprocessing pool.

    args_tuple: (i, instance, args, outdir, intermediate_dir)
    """
    i, instance, args, outdir, intermediate_dir = args_tuple
    idx = instance.get("index", i)
    question = (instance.get("question") or "").strip()
    if isinstance(instance["answer"], list):
        gold_cot = "\n".join(instance["answer"])
    else:
        gold_cot = (instance.get("answer") or "").strip()
    # gold_cot = (instance.get("answer") or "").strip()
    gold_answer = instance.get("final_answer", "")

    intermediate_path = intermediate_dir / f"{idx}.json"
    record = build_result_record(idx=idx, split=args.split, question=question, gold_cot=gold_cot, gold_answer=float(gold_answer))

    try:
        # Run CoT
        pred = solve_with_cot(question, templates_dir=args.templates)

        # Evaluate
        eval_result = evaluate_instance(pred, gold_answer)

        # Fill record
        # record["prediction"] = pred
        if pred["digit_question"] is not None:
            record["digit_question"] = pred["digit_question"]
        record["pred_cot"] = pred.get("chain-of-thought-reasoning")
        record["pred_answer"] = float(pred.get("final_answer"))
        record["evaluation"] = eval_result
        record["prompt"] = pred.get("prompt")

        # Save per-instance result (progressive)
        dump_json(intermediate_path, record)

        return eval_result
    except Exception as e:
        tb = traceback.format_exc()
        record["error"] = str(e)
        # Save partial progress
        try:
            dump_json(intermediate_path, record)
        except Exception:
            # best-effort: if saving fails, write to stdout
            print(f"[WARN] Failed to save intermediate for idx={idx}")

        # Log error in a central errors.jsonl next to intermediate parent
        try:
            log_error(intermediate_dir.parent, idx, str(e), tb)
        except Exception:
            print(f"[WARN] Failed to log error for idx={idx}")

        if args.failfast:
            raise

        return {"predicted": None, "gold": str(gold_answer), "correct": False, "error": str(e)}


def parse_data_aug_source(source: str) -> Tuple[str, str]:
    """
    Try to parse data augmentation model identifier and country from a source path like:
        out/augmented_data/<model>/<country>/augmented/<split>.jsonl
        out/replace_entities/India/meta-llama--Llama-3_1-70B-Instruct/20260104_184259/augmented/test.jsonl
    Returns (data_aug_model, country) or (basename_of_source, "unknown").
    """
    try:
        parts = Path(source).parts
        # look for 'augmented_data' or 'augmented' in path
        if "OnlyCulturalEntities_Dataset" in parts:
            i = parts.index("OnlyCulturalEntities_Dataset")
            country = parts[i + 1] if len(parts) > i + 1 else "unknown"
            data_aug_model = "OnlyCulturalEntities_Dataset"
            return data_aug_model, country
        if "replace_entities" in parts:
            i = parts.index("replace_entities")
            data_aug_model = parts[i + 2] if len(parts) > i + 2 else Path(source).stem
            country = parts[i + 1] if len(parts) > i + 2 else "unknown"
            return data_aug_model, country
        if "augmented_data" in parts:
            i = parts.index("augmented_data")
            data_aug_model = parts[i + 1] if len(parts) > i + 1 else Path(source).stem
            country = parts[i + 2] if len(parts) > i + 2 else "unknown"
            return data_aug_model, country
        # fallback: if path has at least two parts, try to use them
        if len(parts) >= 2:
            return parts[-4], parts[-3] if len(parts) >= 4 else "unknown"
    except Exception:
        pass
    return (Path(source).stem, "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="main", choices=["main", "socratic"], help="Which prompt config to use.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to process.")
    parser.add_argument("--source", type=str, default="gsm8k", help="Either `gsm8k` to load from HuggingFace, or path to a custom JSONL file.")
    parser.add_argument("--limit", type=int, default=-1, help="-1 for all, else max number of items to process.")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based).")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (LLM clients).")
    parser.add_argument("--templates", default="templates/bias_detection", help="Directory with prompt templates.")
    parser.add_argument(("--fewshot-examples"), type=int, default=5, help="Number of few-shot examples to include in prompt (if applicable).")
    parser.add_argument("--failfast", action="store_true", help="Whether to stop on first error.")
    args = parser.parse_args()

    print(f"[INFO] Arguments: {args}")

    # Prepare model-aware timestamped outdir
    model_name = os.getenv("VLLM_MODEL", "NA").replace("/", "_").replace(".", "_")
    if "snapshot" in model_name:
        try:
            model_name = model_name.split("_snapshots")[0]
            model_name = model_name.split("_models--")[1]
        except Exception:
            pass

    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y%m%d_%H%M%S")

    # Load dataset
    data = load_gsm8k(config=args.config, split=args.split, source=args.source)
    n = len(data)
    start = max(0, args.start)
    end = n if args.limit == -1 else min(start + args.limit, n)
    if start >= end:
        print(f"[WARN] start ({start}) >= end ({end}). Nothing to process.")
        return

    print(f"[INFO] Processing split={args.split}, items {start}..{end-1} (total available: {n})")

    # Prepare output directories
    data_aug_model, country = parse_data_aug_source(args.source)
    os.environ["COUNTRY"] = country
    os.environ["FEWSHOT_EXAMPLES"] = str(args.fewshot_examples)

    if country != "unknown":
        outdir = Path("out") / "bias_detection" / country / f"data_aug_{data_aug_model}" / model_name / timestamp
    else:
        print(f"[WARN] Could not determine country from source path: {args.source}")
        outdir = Path("out") / "bias_detection" / args.source / model_name / timestamp

    intermediate_dir = outdir / "intermediate" / args.split
    ensure_dir(intermediate_dir)
    ensure_dir(outdir)

    # Save args for future reference
    dump_json(outdir / "-args.json", vars(args))

    print(f"[INFO] Output directory: {outdir}")
    print(f"[INFO] Intermediate directory: {intermediate_dir}")
    tasks = [(i, data[i], args, outdir, intermediate_dir) for i in range(start, end)]

    # Run tasks in parallel or sequentially with progress
    results: List[Dict[str, Any]] = []
    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_item, tasks), total=len(tasks), desc="Processing items"):
                results.append(res)
    else:
        for t in tqdm(tasks, desc="Processing items"):
            res = process_item(t)
            results.append(res)

    # Aggregate final statistics
    final_stats = evaluate_all(results)
    

    # Save final results and summary
    dump_json(outdir / "-final_results.json", final_stats)
    # Optionally also save raw results list
    try:
        dump_json(outdir / "-raw_results.json", {"results": results})
    except Exception:
        pass

    print(f"[DONE] Final results saved to: {outdir / '-final_results.json'}")
    print(f"[DONE] Intermediate per-instance JSONs in: {intermediate_dir}")
    print("Accuracy / stats:", final_stats)


if __name__ == "__main__":
    main()
