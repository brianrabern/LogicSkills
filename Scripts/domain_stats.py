import argparse
import json
import statistics
import sys
from typing import Any, Dict, List


def read_models(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object mapping id -> model dict")
    return data


def extract_domain_sizes(models: Dict[str, Dict[str, Any]]) -> List[int]:
    sizes: List[int] = []
    for model_id, model in models.items():
        domain = model.get("Domain")
        if isinstance(domain, list):
            sizes.append(len(domain))
            continue
        # Fallbacks (if ever needed)
        for alt_key in ("D", "U"):
            alt_val = model.get(alt_key)
            if isinstance(alt_val, list):
                sizes.append(len(alt_val))
                break
            if isinstance(alt_val, int):
                sizes.append(alt_val)
                break
    return sizes


def compute_stats(sizes: List[int]) -> Dict[str, float]:
    if not sizes:
        return {"min_size": 0, "max_size": 0, "average_size": 0.0}
    return {
        "min_size": int(min(sizes)),
        "max_size": int(max(sizes)),
        "average_size": float(statistics.mean(sizes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute domain size stats from countermodels JSON")
    parser.add_argument(
        "--input",
        "-i",
        default="countermodels.json",
        help="Path to countermodels JSON (default: countermodels.json in repo root)",
    )
    args = parser.parse_args()

    try:
        models = read_models(args.input)
        sizes = extract_domain_sizes(models)
        stats = compute_stats(sizes)
        json.dump(stats, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
