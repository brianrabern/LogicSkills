import json
import logging
from pathlib import Path
from datetime import datetime
from Assessors.core.response_engine import ResponseEngine
from Assessors.settings import get_question_type_config, list_question_types


class InferencePipeline:
    """Generalized pipeline for running inference on different question types."""

    def __init__(self, question_type: str):
        self.config = get_question_type_config(question_type)

        # Initialize inference engine
        self.engine = ResponseEngine(
            model_name=self.config.model_name,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        self.question_type = question_type
        self.results = []

    def load_questions(self):
        """Load questions from the configured file."""
        filepath = Path(self.config.question_file)
        if not filepath.exists():
            raise FileNotFoundError(f"Question file not found: {filepath}")

        with open(filepath, "r") as f:
            questions = json.load(f)

        # Limit questions if specified
        if self.config.max_questions:
            questions = questions[: self.config.max_questions]

        logging.info(f"Loaded {len(questions)} questions from {filepath}")
        return questions

    def run_inference(self):
        """Run inference on questions."""

        questions = self.load_questions()

        logging.info(f"Starting inference for {len(questions)} {self.question_type} questions")
        start_time = datetime.now()

        results = []
        for i, question in enumerate(questions):
            logging.info(f"Processing question {i+1}/{len(questions)}")

            # Run inference
            response = self.engine.query(question["question"])

            # Create result entry
            result = {
                "question": question,
                "response": response,
            }
            results.append(result)

        end_time = datetime.now()
        duration = end_time - start_time

        logging.info(f"Inference completed in {duration}")
        logging.info(f"Processed {len(results)} questions")
        logging.info(f"Successful: {sum(1 for r in results if r['response']['success'])}")
        logging.info(f"Failed: {sum(1 for r in results if not r['response']['success'])}")

        self.results = results
        return results

    def save_results(self):
        """Save inference results to JSON file."""

        output_file = self.config.get_timestamped_output_file()

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=4)

        logging.info(f"Results saved to {output_file}")
        return output_file

    def run_pipeline(self):
        """Run the complete inference pipeline: load questions, run inference, save results."""

        self.load_questions()
        self.run_inference()
        file = self.save_results()
        return file


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Available question types:", list_question_types())

    # Run inference for validity questions
    print("\nRunning inference for validity questions...")
    inference_pipeline = InferencePipeline("validity")
    validity_inference_file = inference_pipeline.run_pipeline()

    # Run inference for symbolization questions
    print("\nRunning inference for symbolization questions...")
    inference_pipeline = InferencePipeline("symbolization")
    symbolization_inference_file = inference_pipeline.run_pipeline()

    # Run inference for countermodel questions
    print("\nRunning inference for countermodel questions...")
    inference_pipeline = InferencePipeline("countermodel")
    countermodel_inference_file = inference_pipeline.run_pipeline()

    print(f"Validity responses saved to: {validity_inference_file}")
    print(f"Symbolization responses saved to: {symbolization_inference_file}")
    print(f"Countermodel responses saved to: {countermodel_inference_file}")
