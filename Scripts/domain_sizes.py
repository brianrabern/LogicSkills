import argparse
import json
import sys
from typing import Any, Dict


def read_models(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object mapping id -> model dict")
    return data


def domain_size(model: Dict[str, Any]) -> int:
    domain = model.get("Domain")
    if isinstance(domain, list):
        return len(domain)
    for alt_key in ("D", "U"):
        val = model.get(alt_key)
        if isinstance(val, list):
            return len(val)
        if isinstance(val, int):
            return val
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce a map: id -> domain size from countermodels JSON")
    parser.add_argument(
        "--input",
        "-i",
        default="countermodels.json",
        help="Path to countermodels JSON (default: countermodels.json in repo root)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Path to write JSON map (default: stdout)",
    )
    args = parser.parse_args()

    models = read_models(args.input)
    mapping: Dict[str, int] = {model_id: domain_size(model) for model_id, model in models.items()}

    if args.output and args.output != "-":
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        json.dump(mapping, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
