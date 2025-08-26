#!/usr/bin/env python3
"""Build formatted validity questions (English).

Reads the validity system prompt and `questions_validity_english.json`, then
prints a JSON object mapping each question id to a single formatted string of
the system prompt, a blank line, and the question text. Also writes the result
to `Assessors/human/formatted_validity_questions_english.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def resolve_repo_paths() -> Dict[str, Path]:
    this_file = Path(__file__).resolve()
    assessors_dir = this_file.parent.parent
    validity_dir = assessors_dir / "validity"

    default_questions = validity_dir / "questions_validity_english.json"
    default_system = validity_dir / "prompts" / "system.py"
    default_output = assessors_dir / "human" / "formatted_validity_questions_english.json"

    return {
        "questions_path": default_questions,
        "system_path": default_system,
        "output_path": default_output,
    }


def load_system_prompt_from_file(system_py_path: Path) -> str:
    if not system_py_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_py_path}")
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("_validity_system_en", str(system_py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {system_py_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "system_prompt"):
        raise AttributeError(f"`system_prompt` not found in {system_py_path}.")
    system_prompt_value = getattr(module, "system_prompt")
    if not isinstance(system_prompt_value, str):
        raise TypeError("`system_prompt` must be a string")
    return system_prompt_value


def read_questions(questions_json_path: Path) -> List[dict]:
    if not questions_json_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_json_path}")
    with questions_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Questions JSON must be a list of objects")
    return data


def build_formatted_mapping(system_prompt: str, questions: List[dict]) -> Dict[str, str]:
    formatted_by_id: Dict[str, str] = {}
    system_prompt_stripped = system_prompt.strip()
    for item in questions:
        question_id = item.get("id")
        question_text = item.get("question")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Each question item must contain a non-empty string `id`.")
        if not isinstance(question_text, str) or not question_text:
            raise ValueError(f"Question {question_id!r} must contain a non-empty string `question`.")
        formatted_text = f"{system_prompt_stripped}\n\n{question_text.strip()}\n"
        formatted_by_id[question_id] = formatted_text
    return formatted_by_id


if __name__ == "__main__":
    paths = resolve_repo_paths()
    system_prompt = load_system_prompt_from_file(paths["system_path"])
    questions = read_questions(paths["questions_path"])
    formatted_map = build_formatted_mapping(system_prompt, questions)

    output_path = paths["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(formatted_map, f, ensure_ascii=False, indent=2)

    print(json.dumps(formatted_map, ensure_ascii=False, indent=2))
