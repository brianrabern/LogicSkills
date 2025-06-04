import json
import logging
from typing import Dict, Any
from Evaluation.model import Model
from Evaluation.countermodel_questions.prompts.extractor_prompt import extractor_prompt
from Evaluation.countermodel_questions.prompts.evaluation_subject_prompt import evaluation_subject_prompt
from config import EXTRACTOR_MODEL
from Evaluation.countermodel_questions.countermodel_checker import check_countermodel
from Utils.helpers import ast_from_json


class CountermodelEvaluator:
    def __init__(self, model: Model):
        self.model = model
        self.extractor_model = Model(EXTRACTOR_MODEL)
        self.results = []

    def evaluate_question(self, question):
        """Evaluate a single question."""

        # get model's response to the question
        response = self.model.query(question["argument_form"])
        print(f"\nQuestion {question['id']}:")
        print(f"Model response: {response}")

        # extract the answer using our extractor model
        extraction = self.extractor_model.query(extractor_prompt(response), parse_json=True)
        print(f"Extracted answer: {extraction}")

        is_correct, comments = check_countermodel(extraction, ast_from_json(question["argument_ast"]))

        # log whether the answer was correct
        print(f"Answer was {'correct' if is_correct else 'incorrect' if is_correct is False else 'indeterminate'}")

        # prepare result
        result = {
            "question_id": question["id"],
            "question_type": "countermodel",
            "question": question["argument_form"],
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
        correct = sum(1 for r in self.results if r.get("is_correct", False))
        errors = sum(1 for r in self.results if "error" in r)

        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "errors": errors,
        }

    def save_results(self, filepath: str):
        """Save evaluation results to a JSON file."""
        with open(filepath, "w") as f:
            json.dump({"results": self.results, "summary": self.get_summary()}, f, indent=2)


if __name__ == "__main__":
    from pathlib import Path

    evaluation_subject = Model("meta-llama/llama-3.2-3b-instruct", system_prompt=evaluation_subject_prompt)
    evaluator = CountermodelEvaluator(evaluation_subject)
    # quesiton are in same directory as this file
    questions_path = Path(__file__).parent / "questions_invalid_arguments.json"
    with open(questions_path, "r") as f:
        questions = json.load(f)

    results = evaluator.evaluate_questions(questions)
    print(results)
