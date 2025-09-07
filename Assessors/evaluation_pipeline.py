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

# Model-specific configurations
llama3_1_8B = {
    "validity": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.1-8B-Instruct_20250704_111415.json", "run": False},
        {"language": "english", "file": "english_meta-llama_Llama-3.1-8B-Instruct_20250704_111146.json", "run": False},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.1-8B-Instruct_20250704_110812.json", "run": True},
        {"language": "english", "file": "english_meta-llama_Llama-3.1-8B-Instruct_20250704_110742.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "meta-llama_Llama-3.1-8B-Instruct_20250704_111745.json", "run": False}
    ],
}

llama3_1_70B = {
    "validity": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.1-70B-Instruct_20250704_163947.json", "run": True},
        {"language": "english", "file": "english_meta-llama_Llama-3.1-70B-Instruct_20250704_115857.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.1-70B-Instruct_20250704_113420.json", "run": True},
        {"language": "english", "file": "english_meta-llama_Llama-3.1-70B-Instruct_20250704_113044.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "meta-llama_Llama-3.1-70B-Instruct_20250704_174747.json", "run": True}
    ],
}

llama3_2_3B = {
    "validity": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.2-3B-Instruct_20250704_111807.json", "run": True},
        {"language": "english", "file": "english_meta-llama_Llama-3.2-3B-Instruct_20250625_182607.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.2-3B-Instruct_20250704_111411.json", "run": True},
        {"language": "english", "file": "english_meta-llama_Llama-3.2-3B-Instruct_20250704_111411.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "meta-llama_Llama-3.2-3B-Instruct_20250704_114729.json", "run": True}
    ],
}

qwen2_5_32B = {
    "validity": [
        {"language": "carroll", "file": "carroll_Qwen_Qwen2.5-32B-Instruct_20250704_143900.json", "run": True},
        {"language": "english", "file": "english_Qwen_Qwen2.5-32B-Instruct_20250704_143949.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_Qwen_Qwen2.5-32B-Instruct_20250704_142339.json", "run": True},
        {"language": "english", "file": "english_Qwen_Qwen2.5-32B-Instruct_20250704_142322.json", "run": True},
    ],
    "countermodel": [{"language": "default", "file": "Qwen_Qwen2.5-32B-Instruct_20250704_144822.json", "run": True}],
}

qwen2_5_72B = {
    "validity": [
        {"language": "carroll", "file": "carroll_Qwen_Qwen2.5-72B-Instruct_20250704_150117.json", "run": True},
        {"language": "english", "file": "english_Qwen_Qwen2.5-72B-Instruct_20250704_145450.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_Qwen_Qwen2.5-72B-Instruct_20250704_142617.json", "run": True},
        {"language": "english", "file": "english_Qwen_Qwen2.5-72B-Instruct_20250704_142643.json", "run": True},
    ],
    "countermodel": [{"language": "default", "file": "Qwen_Qwen2.5-72B-Instruct_20250704_235036.json", "run": True}],
}

qwen3_32B = {
    "validity": [
        {"language": "carroll", "file": "carroll_Qwen_Qwen3-32B_20250704_215121.json", "run": True},
        {"language": "english", "file": "english_Qwen_Qwen3-32B_20250704_211833.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_Qwen_Qwen3-32B_20250704_172030.json", "run": True},
        # Note: English symbolization file for Qwen3-32B doesn't exist
    ],
    "countermodel": [{"language": "default", "file": "Qwen_Qwen3-32B_20250704_214308.json", "run": True}],
}

# gpt4o_mini = {
#     "validity": [
#         {"language": "carroll", "file": "carroll_openai_gpt-4o-mini_20250623_150504.json", "run": True},
#         {"language": "english", "file": "english_openai_gpt-4o-mini_20250623_152128.json", "run": True},
#     ],
#     "symbolization": [
#         {"language": "carroll", "file": "carroll_openai_gpt-4o-mini_20250624_104629.json", "run": True},
#         {"language": "english", "file": "english_openai_gpt-4o-mini_20250624_104752.json", "run": True},
#     ],
#     "countermodel": [{"language": "default", "file": "openai_gpt-4o-mini_20250627_112938.json", "run": True}],
# }

claude_3_7_sonnet = {
    "validity": [
        {"language": "carroll", "file": "carroll_anthropic_claude-3.7-sonnet_20250715_090919.json", "run": True},
        {"language": "english", "file": "english_anthropic_claude-3.7-sonnet_20250715_085332.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_anthropic_claude-3.7-sonnet_20250715_091602.json", "run": True},
        {"language": "english", "file": "english_anthropic_claude-3.7-sonnet_20250715_094020.json", "run": True},
    ],
    "countermodel": [{"language": "default", "file": "anthropic_claude-3.7-sonnet_20250715_105308.json", "run": True}],
}


# Latest explicit config for anthropic/claude-3.7-sonnet (2025-08-22 runs)
claude_3_7_sonnet_20250822 = {
    "validity": [
        {"language": "carroll", "file": "carroll_anthropic_claude-3.7-sonnet_20250822_194011.json", "run": True},
        {"language": "english", "file": "english_anthropic_claude-3.7-sonnet_20250822_190250.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_anthropic_claude-3.7-sonnet_20250822_195739.json", "run": True},
        {"language": "english", "file": "english_anthropic_claude-3.7-sonnet_20250822_194859.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "anthropic_claude-3.7-sonnet_20250822_210348.json", "run": True},
    ],
}

gemini_2_5_flash = {
    "validity": [
        {"language": "carroll", "file": "carroll_google_gemini-2.5-flash_20250715_130916.json", "run": True},
        {"language": "english", "file": "english_google_gemini-2.5-flash_20250715_125326.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_google_gemini-2.5-flash_20250715_131652.json", "run": True},
        {"language": "english", "file": "english_google_gemini-2.5-flash_20250715_133755.json", "run": True},
    ],
    "countermodel": [{"language": "default", "file": "google_gemini-2.5-flash_20250715_142745.json", "run": True}],
}


gpt5_mini = {
    "countermodel": [{"language": "default", "file": "openai_gpt-5-mini_20250816_165658.json", "run": True}],
}

# Placeholders for upcoming models — set file names and flip run=True when ready
qwen2_5_math_72b = {
    "validity": [
        {"language": "carroll", "file": "", "run": False},
        {"language": "english", "file": "english_Qwen_Qwen2.5-Math-72B-Instruct_20250823_201432.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_Qwen_Qwen2.5-Math-72B-Instruct_20250822_144613.json", "run": True},
        {"language": "english", "file": "english_Qwen_Qwen2.5-Math-72B-Instruct_20250821_191428.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "", "run": False},
    ],
}

microsoft_phi_4 = {
    "validity": [
        {"language": "carroll", "file": "carroll_microsoft_phi-4_20250819_234021.json", "run": True},
        {"language": "english", "file": "english_microsoft_phi-4_20250819_232501.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_microsoft_phi-4_20250819_231353.json", "run": True},
        {"language": "english", "file": "english_microsoft_phi-4_20250819_231259.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "microsoft_phi-4_20250820_000522.json", "run": True},
    ],
}

openthinker2_32b = {
    "validity": [
        {"language": "carroll", "file": "carroll_open-thoughts_OpenThinker2-32B_20250822_085511.json", "run": False},
        {"language": "english", "file": "english_open-thoughts_OpenThinker2-32B_20250822_003253.json", "run": False},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_open-thoughts_OpenThinker2-32B_20250821_161949.json", "run": False},
        {"language": "english", "file": "english_open-thoughts_OpenThinker2-32B_20250821_080442.json", "run": False},
    ],
    "countermodel": [
        {"language": "default", "file": "open-thoughts_OpenThinker2-32B_20250822_171802.json", "run": True},
    ],
}

gpt_oss_120b = {
    "validity": [
        {"language": "carroll", "file": "", "run": False},
        {"language": "english", "file": "", "run": False},
    ],
    "symbolization": [
        {"language": "carroll", "file": "", "run": False},
        {"language": "english", "file": "", "run": False},
    ],
    "countermodel": [
        {"language": "default", "file": "", "run": False},
    ],
}


# Latest explicit config for google/gemini-2.5-flash (2025-08-22 runs)
gemini_2_5_flash_20250822 = {
    "validity": [
        {"language": "carroll", "file": "carroll_google_gemini-2.5-flash_20250822_145432.json", "run": True},
        {"language": "english", "file": "english_google_gemini-2.5-flash_20250822_142558.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_google_gemini-2.5-flash_20250822_150206.json", "run": True},
        {"language": "english", "file": "english_google_gemini-2.5-flash_20250822_145748.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "google_gemini-2.5-flash_20250822_151018.json", "run": True},
    ],
}

gpt_4o = {
    "validity": [
        {"language": "carroll", "file": "carroll_openai_gpt-4o_20250905_160039.json", "run": True},
        {"language": "english", "file": "english_openai_gpt-4o_20250905_152608.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_openai_gpt-4o_20250905_161030.json", "run": True},
        {"language": "english", "file": "english_openai_gpt-4o_20250905_160544.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "openai_gpt-4o_20250905_164818.json", "run": True},
    ],
}

deepseek_chat = {
    "validity": [
        {"language": "carroll", "file": "carroll_deepseek_deepseek-chat_20250906_134412.json", "run": True},
        {"language": "english", "file": "english_deepseek_deepseek-chat_20250906_114800.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_deepseek_deepseek-chat_20250906_140004.json", "run": True},
        {"language": "english", "file": "english_deepseek_deepseek-chat_20250906_135216.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "deepseek_deepseek-chat_20250906_160610.json", "run": True},
    ],
}

# Latest explicit config for meta-llama/Llama-3.1-8B-Instruct (2025-08-19 runs)
llama3_1_8B_20250819 = {
    "validity": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.1-8B-Instruct_20250819_003756.json", "run": True},
        {"language": "english", "file": "english_meta-llama_Llama-3.1-8B-Instruct_20250819_001821.json", "run": True},
    ],
    "symbolization": [
        {"language": "carroll", "file": "carroll_meta-llama_Llama-3.1-8B-Instruct_20250819_000139.json", "run": True},
        {"language": "english", "file": "english_meta-llama_Llama-3.1-8B-Instruct_20250819_000059.json", "run": True},
    ],
    "countermodel": [
        {"language": "default", "file": "meta-llama_Llama-3.1-8B-Instruct_20250819_005902.json", "run": True},
    ],
}


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

    def extract_metadata_from_results(self, inference_results):
        """Extract language and model information from inference results."""
        if not inference_results:
            return None, None

        # Extract language from the first result
        language = None
        model_name = None

        if len(inference_results) > 0:
            first_result = inference_results[0]

            # Get language from question (only for validity and symbolization)
            if "question" in first_result and "language" in first_result["question"]:
                language = first_result["question"]["language"]
            elif self.question_type == "countermodel":
                # Countermodel doesn't have language variants
                language = "default"

            # Get model name from inference metadata
            if "response" in first_result and "inference_metadata" in first_result["response"]:
                model_name = first_result["response"]["inference_metadata"].get("model_name")

        return language, model_name

    def run_evaluation(self, inference_results):
        """Run evaluation on inference results using the existing evaluator."""
        self.logger.info(f"Starting evaluation for {len(inference_results)} {self.question_type} responses")

        # Extract metadata and update evaluator
        language, model_name = self.extract_metadata_from_results(inference_results)
        if language:
            self.evaluator.language = language
        if model_name:
            self.evaluator.model_name = model_name

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

        # Add to main results file
        self.add_to_main_results(summary, output_file)

        return output_file

    def add_to_main_results(self, summary: Dict[str, Any], evaluation_file: str):
        """Add evaluation summary to the main results file."""
        main_results_file = "results/main_evaluation_results.json"

        # Load existing results or create new
        if Path(main_results_file).exists():
            with open(main_results_file, "r", encoding="utf-8") as f:
                main_results = json.load(f)
        else:
            main_results = {"evaluations": [], "last_updated": None}

        # Create evaluation entry
        evaluation_entry = {
            "evaluation_file": evaluation_file,
            "timestamp": Path(evaluation_file).stem.split("_")[-1],  # Extract timestamp from filename
            "summary": summary,
        }

        # Check if this evaluation already exists (based on file path)
        existing_indices = [
            i
            for i, eval_entry in enumerate(main_results["evaluations"])
            if eval_entry["evaluation_file"] == evaluation_file
        ]

        if existing_indices:
            # Update existing entry
            main_results["evaluations"][existing_indices[0]] = evaluation_entry
            self.logger.info(f"Updated existing evaluation in main results: {evaluation_file}")
        else:
            # Add new entry
            main_results["evaluations"].append(evaluation_entry)
            self.logger.info(f"Added new evaluation to main results: {evaluation_file}")

        # Update timestamp
        from datetime import datetime

        main_results["last_updated"] = datetime.now().isoformat()

        # Save main results
        Path(main_results_file).parent.mkdir(parents=True, exist_ok=True)
        with open(main_results_file, "w", encoding="utf-8") as f:
            json.dump(main_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Main results updated: {main_results_file}")

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

    # Available configs:
    # - llama3_1_8B /done
    # - llama3_1_8B_20250819
    # - llama3_1_70B \done
    # - llama3_2_3B \done
    # - qwen2_5_32B \done
    # - qwen3_32B \done
    # - claude_3_7_sonnet \done
    # - gemini_2_5_flash
    # - gemini_2_5_flash_20250822
    # - gpt5_mini
    # - gpt_4o
    # - deepseek_chat
    # Select which model config to use
    # CONFIG = gemini_2_5_flash_20250822  # Change this to any of the model configs above
    # CONFIG = claude_3_7_sonnet_20250822
    CONFIG = llama3_1_8B_20250819

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
