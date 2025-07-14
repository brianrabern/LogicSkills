import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal

from tqdm import tqdm
from transformers import AutoTokenizer

from Assessors.core.prompt_prep import build_api_chat_prompts, build_local_chat_prompts
from Assessors.countermodel.prompts.system import system_prompt as countermodel_system_prompt
from Assessors.symbolization.prompts.system import system_prompt as symbolization_system_prompt
from Assessors.validity.prompts.system import system_prompt as validity_system_prompt
from Models.load_configs import load_model_config
from Models.model_type import LOCAL_MODEL
from Models.model_wrapper import ModelWrapper
from Utils.helpers import set_seed
from Utils.logging_config import setup_logging

MODEL_ARG_PATH = "Models/model_config"

# Set up logging
setup_logging("inference_pipeline")


class InferencePipeline:
    """Generalized pipeline for running inference on different question types."""

    def __init__(
        self,
        question_type: Literal["validity", "symbolization", "countermodel"],
        language: Literal["english", "carroll"],
        max_questions: int,
        model_name: str,
        model_arg_path: str,
        model_path: str | None,
        backend: Literal["transformers", "vllm", "api"],
        num_gpus: int = 1,
    ) -> None:
        self.question_type = question_type
        self.language = language
        self.max_questions = max_questions
        self.model_name = model_name
        self.model_path = model_path
        self.backend = backend
        self.num_gpus = num_gpus

        # Get system prompt
        self.system_prompt = self.load_system_prompt()

        # Get model config
        model_config_file = os.path.join(model_arg_path, f"{model_name.replace('/', '_')}.yaml")
        self.model_args, self.tokenizer_kwargs, self.prompt_args = load_model_config(yaml_file_path=model_config_file)

        # Logging
        self.inference_log = []

    def load_system_prompt(self) -> str:
        """
        Loads the system prompt string for the given question type.

        Returns:
            A string containing the system prompt.

        Raises:
            ValueError: If the question type is unknown.
        """
        if self.question_type == "validity":
            return validity_system_prompt
        elif self.question_type == "symbolization":
            return symbolization_system_prompt
        elif self.question_type == "countermodel":
            return countermodel_system_prompt
        else:
            raise ValueError(
                f"Unknown question type: {self.question_type}. Available: ['validity', 'symbolization', 'countermodel']"
            )

    def load_questions(self) -> List[Dict[str, str]]:
        """
        Load questions from the configured file.

        Returns:
            A list of dictionaries representing the questions.

        Raises:
            FileNotFoundError: If the question file does not exist.
        """
        # For countermodel, use the default path since it doesn't have language variants
        if self.question_type == "countermodel":
            question_file = f"Assessors/{self.question_type}/questions_{self.question_type}.json"
        else:
            # Use language parameter to construct question file path for other question types
            question_file = f"Assessors/{self.question_type}/questions_{self.question_type}_{self.language}.json"

        filepath = Path(question_file)
        if not filepath.exists():
            raise FileNotFoundError(f"Question file not found: {filepath}")

        with open(filepath, "r") as f:
            questions = json.load(f)

        # Limit questions if specified
        if self.max_questions > 0:
            questions = questions[: self.max_questions]

        logging.info(f"Loaded {len(questions)} questions from {filepath}")
        return questions

    def load_chat_prompts(self, questions: List[str], tokenizer: AutoTokenizer | None) -> List[str] | List[List[Dict]]:
        """
        Assembles prompts for either local or API models, depending on the model name.

        Args:
            questions (List[str]): The list of questions to assemble prompts for.
            tokenizer (AutoTokenizer): The tokenizer to use for converting the prompts to chat prompts.

        Returns:
            List[str] | List[List[Dict]]: A list of strings, each representing a chat prompt for a local model,
                or a list of lists of dictionaries, each representing a chat prompt for the API.
        """
        if LOCAL_MODEL[self.model_name]:
            return build_local_chat_prompts(
                prompts=questions,
                tokenizer=tokenizer,
                accepts_system_prompt=self.model_args.system_message,
                system_prompt=self.system_prompt,
            )

        return build_api_chat_prompts(
            prompts=questions, accepts_system_prompt=self.model_args.system_message, system_prompt=self.system_prompt
        )

    def init_model_wrapper(
        self,
        cache_dir: str | None,
        backend: Literal["transformers", "vllm", "api"],
        num_gpus: int = 1,
    ) -> ModelWrapper:
        """
        Initializes the model wrapper with the specified model name, path, and initialization arguments.

        Args:
            cache_dir (str | None): The path to the model.
            backend (Literal['transformers', 'vllm', 'api']): The inference backend to use. Can be 'transformers', 'vllm', or 'api'.
            num_gpus (int, optional): Number of GPUs used for model inference. Defaults to 1.

        Returns:
            ModelWrapper: The initialized model wrapper.
        """
        if not LOCAL_MODEL[self.model_name]:
            assert backend == "api", f"Non-local model must be called via API backend! Specify backend as 'api'."
            model_path = None
            tokenizer_path = None
        else:
            model_path = os.path.join(cache_dir, "model", self.model_name)
            tokenizer_path = os.path.join(cache_dir, "tokenizer", self.model_name)

        model_wrapper = ModelWrapper(
            model_name=self.model_name,
            model_path=model_path,
            model_init_kwargs=self.model_args.init_kwargs,
            tokenizer_path=tokenizer_path,
            tokenizer_init_kwargs=self.tokenizer_kwargs,
            backend=backend,
            num_gpus=num_gpus,
        )

        return model_wrapper

    def run_inference(
        self,
        question_dicts: str,
        chat_prompts: List[str] | List[Dict[str, str]],
        model_wrapper: ModelWrapper,
        batch_size: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Runs inference on the given prompts and questions using the specified model wrapper.

        Args:
            original_questions (str): The original questions to be processed.
            chat_prompts (List[str] | List[Dict[str, str]]): The chat prompts to be processed.
                If the model is a local model, this should be a list of strings, each representing a chat prompt.
                If the model is a non-local model, this should be a list of dictionaries, each representing a chat prompt.
            model_wrapper (ModelWrapper): The model wrapper to use for inference.
            batch_size (int, optional): The batch size to use for inference. Defaults to 1.

        Returns:
            List[Dict[str, str]]: A list of results, each containing the original question and the generated response.
        """
        logging.info(f"Starting inference for {len(chat_prompts)} {self.question_type} questions")
        num_batches = len(chat_prompts) // batch_size + (len(chat_prompts) % batch_size > 0)
        results: List[Dict[str, str]] = []

        start_time = datetime.now()
        for i in tqdm(range(num_batches), desc="Processing Batches"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            chat_prompt_batch = chat_prompts[start_idx:end_idx]
            question_dict_batch = question_dicts[start_idx:end_idx]

            batched_response = model_wrapper.forward(
                prompts=chat_prompt_batch, inference_kwargs=self.model_args.inference_kwargs
            )

            for question, response in zip(question_dict_batch, batched_response):
                inference_metadata = {
                    "model_name": self.model_name,
                    "system_prompt": self.system_prompt,
                    "model_parameters": self.model_args.inference_kwargs,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "success": response is not None,
                }

                result = {
                    "raw_response": response,
                    "inference_metadata": inference_metadata,
                    "success": response is not None,
                }

                results.append({"question": question, "response": result})
                self.inference_log = results

        end_time = datetime.now()
        duration = end_time - start_time

        logging.info(f"Inference completed in {duration}")
        logging.info(f"Processed {len(results)} questions")
        logging.info(f"Successful: {sum(1 for r in results if r['response']['success'])}")
        logging.info(f"Failed: {sum(1 for r in results if not r['response']['success'])}")

        return results

    def get_inference_log(self) -> List[Dict]:
        """
        Get the log of all inference operations.

        Returns:
            List[Dict]: List of inference logs.
        """
        return self.inference_log

    def get_timestamped_output_file(self) -> str:
        """
        Generate timestamped output filename for inference results.

        Returns:
            str: The path to the output file.
        """
        # Ensure results/inference/{question_type} directory exists
        Path(f"results/inference/{self.question_type}").mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename: {language}_{model_name}_{timestamp}.json
        # For countermodel, skip language prefix since it doesn't have language variants
        model_name_safe = self.model_name.replace("/", "_")
        if self.question_type == "countermodel":
            filename = f"{model_name_safe}_{timestamp}.json"
        else:
            filename = f"{self.language}_{model_name_safe}_{timestamp}.json"

        return f"results/inference/{self.question_type}/{filename}"

    def save_results(self, results: List[Dict[str, str]]):
        """
        Saves the inference results to a JSON file in the "results/inference" directory.

        Args:
            results (List[Dict[str, str]]): The list of results to be saved.
        """
        Path("results/inference").mkdir(parents=True, exist_ok=True)
        Path("results/evaluation").mkdir(parents=True, exist_ok=True)

        output_file = self.get_timestamped_output_file()

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        # For countermodel, don't include language in the log message
        if self.question_type == "countermodel":
            logging.info(f"{self.question_type} responses saved to: {output_file}")
        else:
            logging.info(f"{self.language} {self.question_type} responses saved to: {output_file}")

    def run_pipeline(self, batch_size: int = 1):
        """
        Runs the entire inference pipeline from start to finish.

        Args:
            batch_size (int): The batch size.
        """
        # Fetch raw data
        question_dicts = self.load_questions()

        # Load model and prep data
        model_wrapper = self.init_model_wrapper(
            cache_dir=self.model_path, backend=self.backend, num_gpus=self.num_gpus
        )
        chat_dataset = self.load_chat_prompts(
            questions=[question_dict["question"] for question_dict in question_dicts], tokenizer=model_wrapper.tokenizer
        )

        # Run Inference
        results = self.run_inference(
            question_dicts=question_dicts, chat_prompts=chat_dataset, model_wrapper=model_wrapper, batch_size=batch_size
        )

        # Save results
        self.save_results(results)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Run LLM inference on JabberBench samples.")

    # General configs
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (relevant for vLLM).")

    # Dataset configs
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "carroll"],
        help="The languahe of the samples.",
    )
    parser.add_argument(
        "--question_type",
        type=str,
        default="validity",
        choices=["validity", "symbolization", "countermodel"],
        help="The subtask to evaluate.",
    )
    parser.add_argument(
        "--max_questions", type=int, default=-1, help="Number of questions to assess. Default -1 means all."
    )

    # Configs about model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Large Language Model to use.",
    )
    parser.add_argument(
        "--model_arg_path",
        type=str,
        default="Models/model_config",
        help="Path to model arguments.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="hf_models",
        help="Cache directory for models and tokenizer.",
    )

    # Inference configs
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["transformers", "vllm", "api"],
        help="The inference backend to use.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    set_seed(args.seed)

    # Check command-line arguments
    if args.model not in LOCAL_MODEL.keys():
        raise ValueError(f"Unsupported model: {args.model}! Currently only supports:\n{LOCAL_MODEL.keys()}")

    if not LOCAL_MODEL[args.model]:
        assert (
            args.backend == "api"
        ), f"Backend must be 'api' if model is not a local model! Currently, it is: {args.backend}"
        logging.info("Batch size set to 1 due to API model.")
        args.batch_size = 1
        args.num_gpus = 0

    # Run inference for questions
    logging.info(f"\nRunning inference for {args.language} {args.question_type} questions...")
    inference_pipeline = InferencePipeline(
        question_type=args.question_type,
        language=args.language,
        max_questions=args.max_questions,
        model_name=args.model,
        model_arg_path=args.model_arg_path,
        model_path=args.cache_dir,
        backend=args.backend,
        num_gpus=args.num_gpus,
    )
    inference_pipeline.run_pipeline(batch_size=args.batch_size)
