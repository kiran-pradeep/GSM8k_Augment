# evaluator.py
"""
Evaluator for comparing LLM answers with ground truth.
"""

from typing import Dict, List


def evaluate_instance(pred: Dict[str, str], gold: str) -> Dict[str, any]:
    """
    Compare a single prediction against ground truth.

    Args:
      pred : dict with "final_answer"
      gold : str (ground truth numeric answer)

    Returns:
      {
        "predicted": str,
        "gold": str,
        "correct": bool
      }
    """
    try:
        pred_ans = pred.get("final_answer", "").strip()
    except AttributeError:
        pred_ans = pred["final_answer"]
    
    gold_ans = str(gold).strip()

    correct = False
    try:
        # Compare numerically
        correct = round(float(pred_ans), 2) == round(float(gold_ans), 2)
    except Exception:
        # Fall back to string equality
        correct = pred_ans == gold_ans

    return {
        "predicted": round(float(pred_ans), 2),
        "gold": round(float(gold_ans), 2),
        "correct": correct
    }


def evaluate_all(results: List[Dict]) -> Dict[str, any]:
    """
    Aggregate accuracy across results.
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": acc
    }
