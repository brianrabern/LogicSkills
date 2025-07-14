"""
Model wrapper around HF and VLLM models
"""

import logging
from typing import Any, Dict, List, Literal, Tuple

from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# make vllm optional since it requires CUDA
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from Models.model_type import LOCAL_MODEL

BASE_URL = OPENROUTER_BASE_URL
API_KEY = OPENROUTER_API_KEY


logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(
        self,
        model_name: str,
        model_path: str | None,
        model_init_kwargs: Dict[str, Any],
        tokenizer_path: str | None,
        tokenizer_init_kwargs: Dict[str, Any],
        backend: Literal["transformers", "vllm", "api"],
        num_gpus: int = 1,
    ) -> None:
        assert (
            backend == "api" if not LOCAL_MODEL[model_name] else True
        ), f"Backend must be 'api' if model is not a local model! Currently, it is: {backend}"
        self.model_name = model_name
        self.backend = backend

        self.model = self._init_model(model_name, model_path, model_init_kwargs, backend, num_gpus)
        self.tokenizer = self._init_tokenizer(model_name, tokenizer_path, tokenizer_init_kwargs, backend)

    def _init_model(
        self,
        model_name: str,
        model_path: str,
        model_init_kwargs: Dict,
        backend: Literal["transformers", "vllm", "api"],
        num_gpus: int = 1,
    ) -> PreTrainedModel | LLM | OpenAI:
        """
        Initialize the model with the specified name, path, and initialization arguments.

        Args:
            model_name (str): The name of the model to be initialized.
            model_path (str): The path to the model.
            model_init_kwargs (Dict): Additional keyword arguments for model initialization.
            backend (Literal['transformers', 'vllm', 'api']): The inference backend to use. Can be 'transformers', 'vllm', or 'api'.
            num_gpus (int, Optional): Number of GPUs used for model inference.

        Returns:
            PreTrainedModel: An instance of PreTrainedModel.
        """
        try:
            if backend == "transformers":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=model_path,
                    device_map="auto",
                    **model_init_kwargs,
                )
            elif backend == "vllm":
                if not VLLM_AVAILABLE:
                    raise ImportError("vllm is not available. It requires CUDA. ")
                model_init_kwargs["dtype"] = model_init_kwargs.pop("torch_dtype")
                model = LLM(
                    model=model_name,
                    tokenizer=model_name,
                    trust_remote_code=True,
                    download_dir=model_path,
                    tensor_parallel_size=num_gpus,
                    gpu_memory_utilization=0.90,
                    **model_init_kwargs,
                )
            elif backend == "api":
                model = OpenAI(base_url=BASE_URL, api_key=API_KEY)
            else:
                raise ValueError(f"Invalid backend: {backend}")
            logger.info(f"Model {model_name} initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise

    def _init_tokenizer(
        self,
        model_name: str,
        tokenizer_path: str,
        tokenizer_init_kwargs: Dict,
        backend: Literal["transformers", "vllm", "api"],
    ) -> PreTrainedTokenizerBase | None:
        """
        Initialize the tokenizer with the specified model name, tokenizer path, and initialization arguments.

        Args:
            model_name (str): The name of the model to be used for tokenization.
            tokenizer_path (str): The path to the tokenizer.
            tokenizer_init_kwargs (Dict): Additional initialization keyword arguments for the tokenizer.
            backend (Literal['transformers', 'vllm', 'api']): The inference backend to use. Can be 'transformers', 'vllm', or 'api'.

        Returns:
            PreTrainedTokenizerBase: The initialized tokenizer.
        """
        print(f"tokenizer_path: {tokenizer_path}")
        try:
            if backend == "api":
                return None
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=tokenizer_path, **tokenizer_init_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer for {model_name} initialized successfully.")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer for {model_name}: {str(e)}")
            raise

    @staticmethod
    def format_output(decoded_answer: List[str]) -> Tuple[List[str], List[str]]:
        """
        Method to format the output for extracting the answer tokens only.

        Args:
            decoded_answer (List[str]): The list of decoded answers to be formatted.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists, the formatted decoded input and decoded output.
        """
        instruction_start_tokens: List[str] = ["[INST]", "user\n\n", "user\n"]
        assistant_start_tokens = [
            "[/INST] ",
            "<|im_start|> assistant\n",
            "<|assistant|>",
            "assistant\n\n",
            "assistant\n",
            "model\n",
        ]
        decoded_input = []
        decoded_output = []

        for sample in decoded_answer:
            # get final shot
            finished_sample = False
            for instr_start_token in instruction_start_tokens:
                if instr_start_token in sample:
                    all_shots = sample.split(instr_start_token)
                    few_shots = f"{instr_start_token}".join(all_shots[:-1]) + instr_start_token
                    final_shot = all_shots[-1]

                    # get model answer
                    for ass_start_token in assistant_start_tokens:
                        if ass_start_token in final_shot:
                            parts = final_shot.split(ass_start_token)
                            assert (
                                len(parts) == 2
                            ), f"More than one instruction token found!\n{sample}\nWith token: {ass_start_token}"
                            decoded_input.append(few_shots + parts[0] + ass_start_token)
                            decoded_output.append(parts[1])
                            finished_sample = True
                            break

                    if finished_sample:
                        break

        return decoded_input, decoded_output

    def forward_transformers(
        self,
        prompts: List[str],
        inference_kwargs: dict[str, Any],
    ) -> List[str]:
        """
        Perform forward inference using the encoded input provided and inference arguments.

        Args:
            prompts (List[str]): A list of input prompts for the model.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for the inference.

        Returns:
            List[str]: The decoded output as lists of strings.
        """
        assert (
            self.backend == "transformers"
        ), f"Invalid 'forward_transformers' function call for backend: {self.backend}"

        encoded_input_dict = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**encoded_input_dict, **inference_kwargs)  # type: ignore
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        _, decoded_output = self.format_output(outputs)

        return decoded_output

    def forward_vllm(
        self,
        prompts: List[str],
        inference_kwargs: dict[str, Any],
    ) -> List[str]:
        """
        Perform forward inference using the VLLM model with the given prompts and inference parameters.

        Args:
            prompts (List[str]): A list of input prompts for the model.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for inference, used to create sampling parameters.

        Returns:
            List[str]: The decoded output as lists of strings.
        """
        sampling_params = SamplingParams(**inference_kwargs)

        outputs = self.model.generate(prompts, sampling_params)  # type: ignore
        decoded_output = [o.outputs[0].text for o in outputs]

        return decoded_output

    def forward_api(
        self,
        prompts: List[List[Dict]],
        inference_kwargs: dict[str, Any],
    ) -> List[str]:
        """
        Perform forward inference using the API model with the given prompts and inference parameters.

        Args:
            prompts (List[List[Dict]]): A list of input prompts for the model, each represented as a list of dictionaries.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for inference, passed to the API model.

        Returns:
            List[str]: The decoded output as lists of strings.
        """
        model_responses: List[str] = []

        for prompt in prompts:
            try:
                completion = self.model.chat.completions.create(
                    model=self.model_name, messages=prompt, **inference_kwargs
                )
                model_responses.append(completion.choices[0].message.content)
            except Exception as e:
                print(f"Error prompting model: {e}")
                model_responses.append(None)

        return model_responses

    def forward(
        self,
        prompts: List[str],
        inference_kwargs: dict[str, Any],
    ) -> List[str]:
        """
        Perform forward inference using the provided prompts and inference arguments.

        Args:
            prompts (List[str]): A list of input prompts for the model.
            inference_kwargs (dict[str, Any]): Additional keyword arguments for the inference.

        Returns:
            List[str]: The decoded output as lists of strings.
        """
        if self.backend == "transformers":
            return self.forward_transformers(prompts, inference_kwargs)
        elif self.backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vllm is not available. It requires CUDA. ")
            return self.forward_vllm(prompts, inference_kwargs)
        elif self.backend == "api":
            return self.forward_api(prompts, inference_kwargs)
        else:
            raise ValueError(f"Invalid backend: {self.backend}")
