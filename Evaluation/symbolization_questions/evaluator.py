import json
import logging
from typing import Dict, Any
from Evaluation.model import Model
from Evaluation.symbolization_questions.prompts.extractor_prompt import extractor_prompt
from Evaluation.symbolization_questions.prompts.evaluation_subject_prompt import evaluation_subject_prompt
from Evaluation.symbolization_questions.symbolization_checker import check_equivalence
from config import EXTRACTOR_MODEL


class SymbolizationEvaluator:
    def __init__(self, model: Model):
        self.model = model
        self.extractor_model = Model(EXTRACTOR_MODEL)
        self.results = []

    def evaluate_question(self, question):
        """Evaluate a single question."""

        # get model's response to the question
        response = self.model.query(question["question"])
        print(f"\nQuestion {question['id']}:")
        print(f"Model response: {response}")

        # extract the answer using our extractor model
        extraction = self.extractor_model.query(extractor_prompt(response), parse_json=True)
        print(f"Extracted answer: {extraction}")

        if extraction["formula"] == question["form"]:
            is_correct = True
            comments = "Exact match: LLM symbolization is identical to DB symbolization"
        else:
            # check to see if the LLM's symbolization is logically equivalent to the DB symbolization
            is_correct = check_equivalence(extraction["formula"], question["form"])
            if is_correct is None:
                is_correct = "UNKNOWN"
                comments = f"Failed to parse the model's symbolization: {extraction['formula']}"
            elif is_correct:
                comments = "Logically equivalent: LLM symbolization is logically equivalent to DB symbolization"
            else:
                comments = "Not logically equivalent"

        # log whether the answer was correct
        print(
            f"Answer was {'correct' if is_correct is True else 'incorrect' if is_correct is False else 'indeterminate'}"
        )

        # prepare result
        result = {
            "question_id": question["id"],
            "question_type": "symbolization",
            "question": question["question"],
            "model_response": response,
            "extracted_answer": extraction if extraction else "None",
            "is_correct": is_correct,
            "comments": comments,
        }

        self.results.append(result)
        return result

    def evaluate_questions(self, questions):
        """Evaluate a list of questions."""
        for question in questions:
            try:
                self.evaluate_question(question)
            except Exception as e:
                logging.error(f"Error evaluating question {question.get('id')}: {e}")
                self.results.append({"question_id": question.get("id"), "error": str(e)})

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of evaluation results."""
        if not self.results:
            return {"error": "No results to summarize"}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct", False) is True)
        errors = sum(1 for r in self.results if "error" in r)
        unknown = sum(1 for r in self.results if r.get("is_correct", False) is None)

        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "errors": errors,
            "unknown": unknown,
        }

    def save_results(self, filepath: str):
        """Save evaluation results to a JSON file."""
        with open(filepath, "w") as f:
            json.dump({"results": self.results, "summary": self.get_summary()}, f, indent=2)


if __name__ == "__main__":
    from pathlib import Path

    evaluation_subject = Model("openai/gpt-4o-mini", system_prompt=evaluation_subject_prompt)
    evaluator = SymbolizationEvaluator(evaluation_subject)
    # quesiton are in same directory as this file
    questions_path = Path(__file__).parent / "questions_symbolization_carroll.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)

    results = evaluator.evaluate_questions(questions)
    print(results)
