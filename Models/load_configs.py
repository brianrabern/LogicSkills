"""
Functions and methods to initialize models locally.
"""

import logging
from typing import Any, Dict, Tuple

import torch

from Models.model_args import ModelArgs, PromptArgs
from Utils.helpers import read_yaml_file


def load_model_args(model_args_dict: Dict[str, Any]) -> ModelArgs:
    """
    Load the model arguments from the given dictionary.

    Args:
        model_args_dict (Dict[str, Any]): The dictionary containing the model arguments.

    Returns:
        ModelArgs: The model arguments object.
    """

    if "init" not in model_args_dict:
        logging.warning("No 'init' argument provided in model-config.yaml file. Using no additional parameters...")
        init_kwargs = {}
    else:
        init_kwargs = model_args_dict["init"]

        # account for torch_dtype
        if "torch_dtype" in init_kwargs:
            mapping = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }
            if init_kwargs["torch_dtype"] in mapping.keys():
                init_kwargs["torch_dtype"] = mapping[init_kwargs["torch_dtype"]]

    if "inference" not in model_args_dict:
        logging.warning("No 'inference' argument provided in model-config.yaml file. Using no additional parameters...")
        inference_kwargs = {}
    else:
        inference_kwargs = model_args_dict["inference"]

    return ModelArgs(init_kwargs=init_kwargs, inference_kwargs=inference_kwargs)


def load_prompt_args(prompt_args_dict: Dict[str, Any]) -> PromptArgs:
    """
    Create a PromptArgs object from the given prompt_args_dict.

    Args:
        prompt_args_dict (Dict[str, Any]): A dictionary containing prompt arguments.

    Returns:
        PromptArgs: The PromptArgs object created from the prompt_args_dict.
    """
    return PromptArgs(**prompt_args_dict)


def load_model_config(
    yaml_file_path: str,
) -> Tuple[ModelArgs, Dict[str, Any], PromptArgs]:
    """
    Loads the model configuration from a YAML file.

    Args:
        yaml_file_path (str): The path to the YAML file.

    Returns:
        Tuple[ModelArgs, Dict[str, Any]]: A tuple containing the model and tokenizer arguments.
    """
    if not yaml_file_path.endswith(".yaml"):
        error_message = f"Provided file path must point to YAML file: {yaml_file_path}"
        logging.error(error_message)
        raise ValueError(error_message)

    model_config_dict = read_yaml_file(yaml_file_path)

    # model kwargs
    model_config = model_config_dict["model"] if "model" in model_config_dict else {}
    model_args = load_model_args(model_config)

    # tokenizer kwargs
    tokenizer_kwargs = model_config_dict["tokenizer"] if "tokenizer" in model_config_dict else {}

    # prompt kwargs
    promt_config = model_config_dict["prompt"] if "prompt" in model_config_dict else {}
    prompt_args = load_prompt_args(promt_config)

    return model_args, tokenizer_kwargs, prompt_args
