import json
from typing import Dict, Any
from Assessors.core.evaluation_engine import EvaluationEngine
from Assessors.symbolization.prompts.extractor import extractor_prompt
from Assessors.symbolization.checker import check_equivalence
from config import EXTRACTOR_MODEL


class SymbolizationEvaluator:
    def __init__(self, model: EvaluationEngine):
        self.model = model
        self.extractor_model = EvaluationEngine(EXTRACTOR_MODEL)
        self.results = []

    def evaluate_response(self, response):
        """Evaluate a single question."""
        try:
            # get model's response to the question
            raw_response = response["response"]["raw_response"]

            # extract the answer using our extractor model
            extraction = self.extractor_model.query(extractor_prompt(raw_response), parse_json=True)
            print(f"Extracted answer: {extraction}")

            if extraction["formula"] == response["question"]["form"]:
                is_correct = True
                comments = "Exact match: LLM symbolization is identical to DB symbolization"
            else:
                # check to see if the LLM's symbolization is logically equivalent to the DB symbolization
                is_correct = check_equivalence(extraction["formula"], response["question"]["form"])
                if is_correct is None:
                    is_correct = "UNKNOWN"
                    comments = f"Failed to parse the model's symbolization: {extraction['formula']}"
                elif is_correct:
                    comments = "Logically equivalent: LLM symbolization is logically equivalent to DB symbolization"
                else:
                    comments = "Not logically equivalent"

            # prepare result
            result = {
                "question_id": response["question"]["id"],
                "question_type": "symbolization",
                "question": response["question"]["question"],
                "model_response": response["response"]["raw_response"],
                "extracted_answer": extraction if extraction else "None",
                "model_metadata": response["response"]["inference_metadata"],
                "is_correct": is_correct,
                "comments": {
                    "correct_answer": response["question"]["form"],
                    "assessment": comments,
                },
            }

            self.results.append(result)
            return result

        except Exception as e:
            # Create a result entry for failed evaluation
            result = {
                "question_id": response["question"]["id"],
                "question_type": "symbolization",
                "question": response["question"]["question"],
                "model_response": response["response"]["raw_response"],
                "extracted_answer": "ERROR",
                "model_metadata": response["response"]["inference_metadata"],
                "is_correct": "UNKNOWN",
                "comments": {
                    "correct_answer": response["question"]["form"],
                    "assessment": f"Evaluation failed: {str(e)}",
                },
            }

            self.results.append(result)
            print(f"Error evaluating question {response['question']['id']}: {e}")
            return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of evaluation results."""
        if not self.results:
            return {"error": "No results to summarize"}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct", False) is True)
        errors = sum(1 for r in self.results if "error" in r)
        unknown = sum(1 for r in self.results if r.get("is_correct", False) == "UNKNOWN")

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
    engine = EvaluationEngine("openai/gpt-4o-mini")
    evaluator = SymbolizationEvaluator(engine)

    with open("questions_symbolization_carroll.json", "r") as f:
        responses = json.load(f)

    for response in responses:
        evaluator.evaluate_response(response)

    print(evaluator.get_summary())
    evaluator.save_results("symbolization_inference_results_evaluated.json")
