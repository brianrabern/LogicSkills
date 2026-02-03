import json
from typing import Dict, Any
from Assessors.core.evaluation_engine import EvaluationEngine
from Assessors.validity.prompts.extractor import extractor_prompt
from config import EXTRACTOR_MODEL


class ValidityEvaluator:
    def __init__(
        self, model: EvaluationEngine, question_type: str = "validity", language: str = None, model_name: str = None
    ):
        self.model = model
        self.extractor_model = EvaluationEngine(EXTRACTOR_MODEL)
        self.results = []
        self.question_type = question_type
        self.language = language
        self.model_name = model_name

    def evaluate_response(self, response):
        """Evaluate a single question."""

        # get model's response to the question
        raw_response = response["response"]["raw_response"]

        # extract the answer using our extractor model
        extraction = self.extractor_model.query(extractor_prompt(raw_response), parse_json=True)
        print(f"Extracted answer: {extraction.get('answer') if extraction else 'None'}")

        # get the sentence ids for model's answer and correct answer
        model_answer_sentence_ids = []
        correct_answer_sentence_ids = []

        if extraction and extraction.get("answer"):
            for num in extraction.get("answer").split(","):
                num = num.strip()
                if num in response["question"].get("option_to_sentence_id", {}):
                    model_answer_sentence_ids.append(response["question"]["option_to_sentence_id"][num])

        if response["question"]["correct_answer"]:
            for num in response["question"]["correct_answer"].split(","):
                num = num.strip()
                if num in response["question"].get("option_to_sentence_id", {}):
                    correct_answer_sentence_ids.append(response["question"]["option_to_sentence_id"][num])

        # log whether the answer was correct
        is_correct = extraction.get("answer") == response["question"]["correct_answer"] if extraction else None
        print(f"Answer was {'correct' if is_correct else 'incorrect' if is_correct is False else 'indeterminate'}")

        # prepare result
        result = {
            "question_id": response["question"]["id"],
            "question_type": "validity",
            "question": response["question"]["question"],
            "model_response": response["response"]["raw_response"],
            "extracted_answer": extraction.get("answer") if extraction else "None",
            "model_metadata": response["response"]["inference_metadata"],
            "is_correct": is_correct,
            "comments": {
                "correct_answer": response["question"]["correct_answer"],
                "model_answer_sentence_ids": model_answer_sentence_ids,
                "correct_answer_sentence_ids": correct_answer_sentence_ids,
            },
        }

        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of evaluation results."""
        if not self.results:
            return {"error": "No results to summarize"}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct", False))
        errors = sum(1 for r in self.results if "error" in r)

        summary = {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "errors": errors,
        }

        # Add metadata if available
        if self.question_type:
            summary["question_type"] = self.question_type
        if self.language:
            summary["language"] = self.language
        if self.model_name:
            summary["model_name"] = self.model_name

        return summary

    def save_results(self, filepath: str):
        """Save evaluation results to a JSON file."""
        with open(filepath, "w") as f:
            json.dump({"results": self.results, "summary": self.get_summary()}, f, indent=2)


if __name__ == "__main__":
    engine = EvaluationEngine("openai/gpt-4o-mini")
    evaluator = ValidityEvaluator(engine)
    with open("validity_inference_results.json", "r") as f:
        responses = json.load(f)

    for response in responses:
        evaluator.evaluate_response(response)

    print(evaluator.get_summary())
    evaluator.save_results("validity_inference_results_evaluated.json")
