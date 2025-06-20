"""
Settings for different question types in the assessment pipeline.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from datetime import datetime
from Assessors.countermodel.prompts.system import system_prompt as countermodel_system_prompt
from Assessors.symbolization.prompts.system import system_prompt as symbolization_system_prompt
from Assessors.validity.prompts.system import system_prompt as validity_system_prompt


@dataclass
class QuestionTypeConfig:
    """Configuration for a specific question type."""

    name: str
    question_file: str
    system_prompt: str
    output_file: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 2500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_questions: Optional[int] = None  # None = all questions

    def get_timestamped_output_file(self) -> str:
        """Generate timestamped output filename for inference results."""
        # Extract language from question file name
        # e.g., "questions_validity_carroll.json" -> "carroll"
        question_filename = Path(self.question_file).name
        language = question_filename.split("_")[-1].replace(".json", "")

        # Ensure results/inference/{question_type} directory exists
        Path(f"results/inference/{self.name}").mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename: {language}_{model_name}_{timestamp}.json
        # For countermodel, skip language prefix since it doesn't have language variants
        model_name_safe = self.model_name.replace("/", "_")
        if self.name == "countermodel":
            filename = f"{model_name_safe}_{timestamp}.json"
        else:
            filename = f"{language}_{model_name_safe}_{timestamp}.json"

        return f"results/inference/{self.name}/{filename}"


# Question type configurations
VALIDITY_CONFIG = QuestionTypeConfig(
    name="validity",
    model_name="openai/gpt-4o-mini",
    question_file="Assessors/validity/questions_validity_carroll.json",
    system_prompt=validity_system_prompt,
    output_file="inference_results_validity_carroll.json",
    max_questions=2,
)

SYMBOLIZATION_CONFIG = QuestionTypeConfig(
    name="symbolization",
    model_name="openai/gpt-4o-mini",
    question_file="Assessors/symbolization/questions_symbolization_carroll.json",
    system_prompt=symbolization_system_prompt,
    output_file="inference_results_symbolization_carroll.json",
    max_questions=2,
)

COUNTERMODEL_CONFIG = QuestionTypeConfig(
    name="countermodel",
    model_name="openai/gpt-4o-mini",
    question_file="Assessors/countermodel/questions_countermodel.json",
    system_prompt=countermodel_system_prompt,
    output_file="inference_results_countermodel.json",
    max_questions=2,
)

# Registry of all question types
QUESTION_TYPES = {
    "validity": VALIDITY_CONFIG,
    "symbolization": SYMBOLIZATION_CONFIG,
    "countermodel": COUNTERMODEL_CONFIG,
}


def get_question_type_config(question_type: str) -> QuestionTypeConfig:
    """Get configuration for a specific question type."""
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Unknown question type: {question_type}. Available: {list(QUESTION_TYPES.keys())}")
    return QUESTION_TYPES[question_type]


def list_question_types() -> list:
    """List all available question types."""
    return list(QUESTION_TYPES.keys())


def get_evaluation_filename_from_inference(inference_file: str) -> str:
    """Generate evaluation filename based on inference filename."""
    inference_path = Path(inference_file)

    # Extract components from inference filename
    # e.g., "results/inference/validity/carroll_openai_gpt-4o-mini_20250620_144423.json"
    # -> question_type = "validity", filename = "carroll_openai_gpt-4o-mini_20250620_144423.json"

    question_type = inference_path.parent.name
    filename = inference_path.name

    # Ensure results/evaluation/{question_type} directory exists
    Path(f"results/evaluation/{question_type}").mkdir(parents=True, exist_ok=True)

    # Return evaluation filename
    return f"results/evaluation/{question_type}/{filename}"
