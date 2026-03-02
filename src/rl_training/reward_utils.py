import re

def extract_answer(text: str):
    """To find the numeric answer in the JSON."""
    # Look for "final_answer": 123 or similar
    match = re.findall(r'"final_answer":\s*([\d\.]+)', text)
    if not match:
        # Fallback to general number extraction at the end
        match = re.findall(r"([\d\.]+)\s*$", text.strip())
    
    try:
        return float(match[-1]) if match else None
    except:
        return None

def soft_format_reward(completions):
    """Rewards the model for following the JSON format."""
    rewards = []
    for content in completions:
        if '"chain-of-thought-reasoning":' in content and '"final_answer":' in content:
            rewards.append(0.2) # Small bonus for format
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward(completions, gold_answers):
    """Rewards the model for the correct numeric answer."""
    rewards = []
    for content, gold in zip(completions, gold_answers):
        pred = extract_answer(content)
        if pred is not None and abs(float(pred) - float(gold)) < 0.01:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards