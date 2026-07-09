from pathlib import Path
from typing import Tuple

def parse_data_aug_source(source: str) -> Tuple[str, str]:
    """
    Try to parse data augmentation model identifier and country from a source path like:
      out/augmented_data/<model>/<country>/augmented/<split>.jsonl
    Returns (data_aug_model, country) or (basename_of_source, "unknown").
    """
    try:
        parts = Path(source).parts
        # look for 'augmented_data' or 'augmented' in path
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

if __name__ == "__main__":
    source = "out/constrained_exps_data/meta-llama--Llama-3_1-70B-Instruct/10.0/augmented/test.jsonl"
    data_aug_model, country = parse_data_aug_source(source)
    outdir = Path("out") / "bias_detection" / country / f"data_aug_{data_aug_model}"
    print(outdir)
    