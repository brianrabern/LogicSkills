import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional


ARG_PREFIX_RE = re.compile(r"^\s*Argument:\s*\n*", re.IGNORECASE)


def clean_question_text(text: str) -> str:
    """Remove leading "Argument:" and surrounding whitespace from a question string."""
    if not isinstance(text, str):
        return ""
    cleaned = ARG_PREFIX_RE.sub("", text)
    return cleaned.strip()


def transform_items(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    transformed: List[Dict[str, str]] = []
    for item in items:
        item_id = item.get("id", "")
        question = clean_question_text(item.get("question", ""))
        transformed.append(
            {
                "id": item_id,
                "question": question,
            }
        )
    return transformed


def read_input(path: Optional[str]) -> List[Dict[str, Any]]:
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)


def write_output(data: Any, path: Optional[str]) -> None:
    if path and path != "-":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cleaned questions (id, question) from input JSON list.")
    parser.add_argument("--input", "-i", help="Path to input JSON file (list of objects). Use '-' for stdin.")
    parser.add_argument("--output", "-o", help="Path to output JSON file. Use '-' for stdout.")
    args = parser.parse_args()

    try:
        items = read_input(args.input)
        if not isinstance(items, list):
            raise ValueError("Input JSON must be a list of objects")
        result = transform_items(items)
        write_output(result, args.output)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
