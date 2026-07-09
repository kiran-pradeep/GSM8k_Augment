import torch
import re
import json


def extract_number(text):
    """
    Extract the first numeric value from a string.
    Handles cases like:
    '42', '42.', '42 dollars', 'The answer is 42'
    """
    nums = re.findall(r"-?\d+\.?\d*", str(text))
    return nums[-1] if nums else None


def reward_func(queries, prompts, labels):
    rewards = []
    scores = []

    for query, label in zip(queries, labels):

        reward = 0.0
        pred_val = None
        reasoning_text = ""

        # --------------------------------------------------
        # 1️⃣ PRIMARY ATTEMPT: JSON PARSING
        # --------------------------------------------------
        try:

            json_start = query.find("{")

            if json_start != -1:
                json_str = query[json_start:]

                # Trim trailing junk after JSON
                json_end = json_str.rfind("}")
                if json_end != -1:
                    json_str = json_str[: json_end + 1]

                json_obj = json.loads(json_str)

                # Reasoning reward
                if "chain-of-thought-reasoning" in json_obj:
                    reasoning_text = str(json_obj["chain-of-thought-reasoning"])
                    reward += 0.05

                # JSON structure reward
                if "final_answer" in json_obj:
                    pred_val = extract_number(json_obj["final_answer"])
                    reward += 0.10

        except Exception:
            pass

        # --------------------------------------------------
        # 2️⃣ FALLBACK EXTRACTION
        # --------------------------------------------------
        if pred_val is None:

            # Pattern A: malformed JSON key
            match_key = re.search(
                r'"final_answer":\s*"?(-?[\d,.]+)"?', query
            )

            if match_key:
                pred_val = match_key.group(1).replace(",", "")

            else:
                # Pattern B: last number in the output
                nums = re.findall(r"-?\d+\.?\d*", query)
                if nums:
                    pred_val = nums[-1]

        # --------------------------------------------------
        # 3️⃣ REASONING QUALITY HEURISTICS
        # --------------------------------------------------
        if reasoning_text:

            # Encourage multi-step reasoning
            steps = reasoning_text.split("\n")
            if len(steps) >= 3:
                reward += 0.03

            # Encourage equation usage
            if any(sym in reasoning_text for sym in ["=", "+", "-", "*", "/"]):
                reward += 0.02

        # --------------------------------------------------
        # 4️⃣ CORRECTNESS REWARD
        # --------------------------------------------------
        try:

            if pred_val is not None:

                if abs(float(pred_val) - float(label)) < 1e-6:
                    reward += 1.0

                else:
                    reward += 0.0

            else:
                # No number produced
                reward -= 0.1

        except Exception:
            reward -= 0.1

        # --------------------------------------------------
        # 5️⃣ REWARD CLIPPING
        # --------------------------------------------------
        reward = max(min(reward, 1.2), -0.2)

        rewards.append(reward)
        scores.append(1.0 if reward > 0.9 else 0.0)

    # --------------------------------------------------
    # 6️⃣ CONVERT TO TENSORS
    # --------------------------------------------------
    rewards_t = torch.tensor(rewards)
    scores_t = torch.tensor(scores)

    return {
        "rewards": rewards_t,
        "scores": scores_t,
        "extra_logs": {
            "mean_reward": rewards_t,
            "accuracy": scores_t,
        },
    }