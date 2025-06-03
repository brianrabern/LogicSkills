from Database.DB import db
from Database.models import Sentence
from Utils.normalize import unescape_logical_form
import json
from pathlib import Path

# initialize database session
session = db.session


def get_logical_entailment(premise_ids, conclusion_id, domain_constraint_id=None):
    """Construct logical entailment string from premise and conclusion IDs."""
    premise_forms = [unescape_logical_form(session.get(Sentence, pid).form) for pid in premise_ids]
    conclusion_form = unescape_logical_form(session.get(Sentence, conclusion_id).form)
    domain_form = (
        unescape_logical_form(session.get(Sentence, domain_constraint_id).form) if domain_constraint_id else ""
    )

    return f"{domain_form + ', ' if domain_form else ''}{', '.join(premise_forms)} |= {conclusion_form}"


def verify_answers(results_file):
    """Verify logical forms for both model answers and correct answers."""
    with open(results_file, "r") as f:
        data = json.load(f)

    results = data["results"]
    verification_results = []

    for result in results:
        if "error" in result:
            continue

        question_id = result["question_id"]
        premise_ids = result["premise_ids"]
        domain_constraint_id = result["domain_constraint_id"]

        # get correct answer logical form
        correct_answer_ids = result["correct_answer_sentence_ids"]
        correct_entailments = []
        for conclusion_id in correct_answer_ids:
            entailment = get_logical_entailment(premise_ids, conclusion_id, domain_constraint_id)
            correct_entailments.append(entailment)

        # get model answer logical forms
        model_answer_ids = result["model_answer_sentence_ids"]
        model_entailments = []
        for conclusion_id in model_answer_ids:
            entailment = get_logical_entailment(premise_ids, conclusion_id, domain_constraint_id)
            model_entailments.append(entailment)

        verification_results.append(
            {
                "question_id": question_id,
                "is_correct": result["is_correct"],
                "correct_entailments": correct_entailments,
                "model_entailments": model_entailments,
                "model_answer": result["extracted_answer"],
                "correct_answer": result["correct_answer"],
            }
        )

    return verification_results


def save_verification_results(results, output_file):
    """Save verification results to a JSON file."""
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved verification results to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python verify_answers.py <results_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    verification_results = verify_answers(results_file)

    # Save results to a new file with '_verified' suffix
    output_file = str(Path(results_file).with_suffix("")) + "_verified.json"
    save_verification_results(verification_results, output_file)
