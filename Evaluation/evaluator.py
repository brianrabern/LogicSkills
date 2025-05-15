import json
import logging
from typing import List, Dict, Any
from Evaluation.model import Model
from Evaluation.prompts.extractor_prompt import extractor_prompt
from Evaluation.prompts.evaluation_subject_prompt import evaluation_subject_prompt
from config import EXTRACTOR_MODEL


class Evaluator:
    def __init__(self, model: Model):
        self.model = model
        self.extractor_model = Model(EXTRACTOR_MODEL)
        self.results = []

    def evaluate_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question."""

        # get model's response to the question
        response = self.model.query(question["text"])
        print(f"\nQuestion {question['id']}:")
        print(f"Model response: {response}")

        # extract the answer using our extractor model
        extraction = self.extractor_model.query(extractor_prompt(response), parse_json=True)
        print(f"Extracted answer: {extraction.get('answer') if extraction else 'None'}")

        # get the sentence ids for model's answer and correct answer
        model_answer_sentence_ids = []
        correct_answer_sentence_ids = []

        if extraction and extraction.get("answer"):
            for num in extraction.get("answer").split(","):
                num = num.strip()
                if num in question.get("option_to_sentence_id", {}):
                    model_answer_sentence_ids.append(question["option_to_sentence_id"][num])

        if question["correct_answer"]:
            for num in question["correct_answer"].split(","):
                num = num.strip()
                if num in question.get("option_to_sentence_id", {}):
                    correct_answer_sentence_ids.append(question["option_to_sentence_id"][num])

        # log whether the answer was correct
        is_correct = extraction.get("answer") == question["correct_answer"] if extraction else None
        print(f"Answer was {'correct' if is_correct else 'incorrect' if is_correct is False else 'indeterminate'}")

        # prepare result
        result = {
            "question_id": question["id"],
            "question": question["text"],
            "model_response": response,
            "extracted_answer": extraction.get("answer") if extraction else "None",
            "model_answer_sentence_ids": model_answer_sentence_ids,
            "correct_answer": question["correct_answer"],
            "correct_answer_sentence_ids": correct_answer_sentence_ids,
            "is_correct": is_correct,
            "premise_ids": question.get("premise_ids", []),
            "domain_constraint_id": question.get("domain_constraint_id"),
        }

        self.results.append(result)
        return result

    def evaluate_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    evaluation_subject = Model("meta-llama/llama-3.2-3b-instruct", system_prompt=evaluation_subject_prompt)
    evaluator = Evaluator(evaluation_subject)
    with open("questions.json", "r") as f:
        questions = json.load(f)

    results = evaluator.evaluate_question(questions[1])
    print(results)
