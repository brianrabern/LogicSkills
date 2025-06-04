import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from Evaluation.model import Model
from Evaluation.validity_questions.evaluator import Evaluator
from Evaluation.validity_questions.prompts.evaluation_subject_prompt import evaluation_subject_prompt


def load_questions(filepath: str) -> List[Dict[str, Any]]:
    """Load questions from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def run_evaluation(model_name: str, questions_file: str):
    """Run evaluation for a single model."""
    # setup logging
    setup_logging()
    logging.info(f"Starting evaluation for model: {model_name}")

    # create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # load questions
    try:
        questions = load_questions(questions_file)
        logging.info(f"Loaded {len(questions)} questions")
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return

    # initialize model and evaluator
    model = Model(model_name, system_prompt=evaluation_subject_prompt)
    evaluator = Evaluator(model)

    # run evaluation
    evaluator.evaluate_questions(questions)

    # save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    output_path = output_dir / f"{safe_model_name}_{timestamp}.json"
    evaluator.save_results(str(output_path))

    # log summary
    summary = evaluator.get_summary()
    logging.info(f"Results for {model_name}:")
    logging.info(f"  Accuracy: {summary['accuracy']:.2%}")
    logging.info(f"  Correct: {summary['correct_answers']}/{summary['total_questions']}")
    logging.info(f"  Errors: {summary['errors']}")
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # configure these values
    MODEL_NAME = "mistralai/mixtral-8x7b-instruct"
    QUESTIONS_FILE = "questions.json"

    run_evaluation(MODEL_NAME, QUESTIONS_FILE)

# evaluation subjects
# meta-llama/llama-3-8b-instruct
# meta-llama/llama-3-70b-instruct
# mistralai/mixtral-8x7b-instruct
# openai/gpt-4o-mini
