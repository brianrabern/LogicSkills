"""
Evaluation pipeline that works with existing evaluators.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from Assessors.core.evaluation_engine import EvaluationEngine
from Assessors.validity.evaluator import ValidityEvaluator
from Assessors.symbolization.evaluator import SymbolizationEvaluator
from Assessors.countermodel.evaluator import CountermodelEvaluator
from Assessors.settings import get_evaluation_filename_from_inference
from Utils.logging_config import setup_logging

# Set up logging
setup_logging("evaluation_pipeline")


class EvaluationPipeline:

    def __init__(self, question_type: str, results_file: str, model_name: str):
        self.question_type = question_type
        self.results_file = results_file
        self.model_name = model_name

        # Initialize evaluation engine
        self.evaluation_engine = EvaluationEngine(model_name)

        # Initialize appropriate evaluator
        self.evaluator = self._create_evaluator()

        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_evaluator(self):
        """Create the appropriate evaluator based on question type."""
        if self.question_type == "validity":
            return ValidityEvaluator(self.evaluation_engine)
        elif self.question_type == "symbolization":
            return SymbolizationEvaluator(self.evaluation_engine)
        elif self.question_type == "countermodel":
            return CountermodelEvaluator(self.evaluation_engine)
        else:
            raise ValueError(f"Unsupported question type: {self.question_type}")

    def load_inference_results(self):
        """Load inference results from JSON file."""
        filepath = Path(self.results_file)
        if not filepath.exists():
            raise FileNotFoundError(f"Inference results file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.logger.info(f"Loaded {len(results)} inference results from {filepath}")
        return results

    def run_evaluation(self, inference_results):
        """Run evaluation on inference results using the existing evaluator."""
        self.logger.info(f"Starting evaluation for {len(inference_results)} {self.question_type} responses")

        # Use the existing evaluator's interface
        for i, response in enumerate(inference_results, 1):
            self.logger.debug(f"Evaluating response {i}/{len(inference_results)}")
            try:
                self.evaluator.evaluate_response(response)
            except Exception as e:
                self.logger.error(f"Error evaluating response {i}: {e}")

        self.logger.info("Evaluation completed")
        return self.evaluator.results

    def run_pipeline(self) -> str:
        """Run the complete evaluation pipeline."""
        self.logger.info(f"Starting evaluation pipeline for {self.question_type}")

        # Load inference results
        inference_results = self.load_inference_results()

        # Run evaluation
        self.run_evaluation(inference_results)

        # Generate evaluation filename
        output_file = get_evaluation_filename_from_inference(self.results_file)

        # Save results using the evaluator's save method
        self.evaluator.save_results(output_file)

        # Log summary
        summary = self.evaluator.get_summary()
        self.logger.info(f"Evaluation Summary: {summary}")
        self.logger.info(f"Evaluation results saved to: {output_file}")

        return output_file

    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        return self.evaluator.get_summary()


def run_evaluation_for_type(question_type: str, inference_results_file: str, model_name: str = "openai/gpt-4o-mini"):
    """Convenience function to run evaluation for a specific question type."""
    inference_results_file = f"results/inference/{question_type}/{inference_results_file}"
    pipeline = EvaluationPipeline(question_type, inference_results_file, model_name)
    return pipeline.run_pipeline()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    CONFIG = {
        "validity": [
            # {
            #     "language": "carroll",
            #     "file": "english_meta-llama_Llama-3.2-3B-Instruct_20250625_182607.json",
            #     "run": True,
            # },
            {
                "language": "english",
                "file": "english_meta-llama_Llama-3.2-3B-Instruct_20250625_182607.json",
                "run": True,
            },
        ],
        # "symbolization": [
        #     {"language": "carroll", "file": "carroll_openai_gpt-4o-mini_20250620_155605.json", "run": True},
        #     {"language": "english", "file": "english_openai_gpt-4o-mini_20250620_155611.json", "run": True},
        # ],
        # "countermodel": [{"language": "default", "file": "openai_gpt-4o-mini_20250620_160856.json", "run": True}],
    }

    record = {}
    for question_type, configs in CONFIG.items():
        for config in configs:
            if config["run"]:
                print(f"Running evaluation for {question_type} ({config['language']})...")
                evaluation_file = run_evaluation_for_type(question_type, config["file"])
                print(f"Evaluation results saved to: {evaluation_file}")
                record[f"{question_type}_{config['language']}"] = evaluation_file
            else:
                print(f"Skipping evaluation for {question_type} ({config['language']})...")

    print(f"Evaluation results saved to: {record}")
