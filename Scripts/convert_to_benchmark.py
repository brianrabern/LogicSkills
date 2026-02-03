import json
import re
from pathlib import Path
from typing import Dict, Any, List


def load_system_prompt(prompt_file: Path) -> str:
    """Load system prompt from a Python file."""
    with open(prompt_file, "r") as f:
        content = f.read()
    # Extract the system_prompt variable value
    match = re.search(r'system_prompt\s*=\s*"""(.*?)"""', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Could not extract system_prompt from {prompt_file}")


def convert_symbolization_question(question: Dict[str, Any], system_prompt: str, language: str) -> Dict[str, Any]:
    """Convert symbolization question to normalized format."""
    return {
        "id": str(question["id"]),
        "task": "symbolization",
        "language": language,
        "input": f"{system_prompt}\n\n{question['question']}",
        "answer": question["form"],
    }


def convert_validity_question(question: Dict[str, Any], system_prompt: str, language: str) -> Dict[str, Any]:
    """Convert validity question to normalized format."""
    # Convert answer string to array of integers
    answer_str = question["correct_answer"]
    if answer_str:
        # Handle comma-separated answers like "1,2" or "4"
        answer_list = [int(x.strip()) for x in answer_str.split(",") if x.strip()]
    else:
        answer_list = []

    return {
        "id": question["id"],
        "task": "validity",
        "language": language,
        "input": f"{system_prompt}\n\n{question['question']}",
        "answer": answer_list,
    }


def convert_countermodel_question(question: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """Convert countermodel question to normalized format."""
    return {
        "id": question["id"],
        "task": "countermodel",
        "language": "formal",
        "input": f"{system_prompt}\n\n{question['question']}",
        "answer": None,
    }


def main():
    base_dir = Path("Assessors")
    output_dir = Path("logicskills/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load system prompts
    symbolization_system = load_system_prompt(base_dir / "symbolization" / "prompts" / "system.py")
    validity_system = load_system_prompt(base_dir / "validity" / "prompts" / "system.py")
    countermodel_system = load_system_prompt(base_dir / "countermodel" / "prompts" / "system.py")

    # Process symbolization questions
    symbolization_questions = []

    # Carroll symbolization
    carroll_file = base_dir / "symbolization" / "questions_symbolization_carroll.json"
    if carroll_file.exists():
        with open(carroll_file, "r") as f:
            carroll_questions = json.load(f)
        for q in carroll_questions:
            symbolization_questions.append(convert_symbolization_question(q, symbolization_system, "carroll"))

    # English symbolization
    english_file = base_dir / "symbolization" / "questions_symbolization_english.json"
    if english_file.exists():
        with open(english_file, "r") as f:
            english_questions = json.load(f)
        for q in english_questions:
            symbolization_questions.append(convert_symbolization_question(q, symbolization_system, "english"))

    # Process validity questions
    validity_questions = []

    # Carroll validity
    carroll_file = base_dir / "validity" / "questions_validity_carroll.json"
    if carroll_file.exists():
        with open(carroll_file, "r") as f:
            carroll_questions = json.load(f)
        for q in carroll_questions:
            validity_questions.append(convert_validity_question(q, validity_system, "carroll"))

    # English validity
    english_file = base_dir / "validity" / "questions_validity_english.json"
    if english_file.exists():
        with open(english_file, "r") as f:
            english_questions = json.load(f)
        for q in english_questions:
            validity_questions.append(convert_validity_question(q, validity_system, "english"))

    # Process countermodel questions
    countermodel_questions = []
    countermodel_file = base_dir / "countermodel" / "questions_countermodel.json"
    if countermodel_file.exists():
        with open(countermodel_file, "r") as f:
            countermodel_data = json.load(f)
        for q in countermodel_data:
            countermodel_questions.append(convert_countermodel_question(q, countermodel_system))

    # Write JSONL files
    with open(output_dir / "symbolization.jsonl", "w") as f:
        for q in symbolization_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with open(output_dir / "validity.jsonl", "w") as f:
        for q in validity_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with open(output_dir / "countermodel.jsonl", "w") as f:
        for q in countermodel_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Created {len(symbolization_questions)} symbolization questions")
    print(f"Created {len(validity_questions)} validity questions")
    print(f"Created {len(countermodel_questions)} countermodel questions")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

