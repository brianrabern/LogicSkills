from typing import Dict, Any, List
import logging
import yaml
import hashlib
import random
import numpy as np
import torch
import json

def ast_from_json(data):
    if isinstance(data, list):
        return tuple(ast_from_json(x) for x in data)
    return data


def canonical_premise_str(premise_ids):
    return ",".join(map(str, sorted(premise_ids)))


def generate_argument_id(premise_ids, conclusion_id):
    full_key = canonical_premise_str(premise_ids) + f",{conclusion_id}"
    return hashlib.sha256(full_key.encode()).hexdigest()[:16]


def set_seed(seed: int = 0) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation. Defaults to 0.

    Returns:
        None
    """
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its content.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing the content of the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as exc:
                logging.error(f"Error parsing YAML file at {file_path}: {exc}")
                raise
    except FileNotFoundError:
        logging.error(f"YAML file not found at {file_path}.")
        raise FileNotFoundError(f"YAML file not found at {file_path}.")
    

def load_json_rows(path: str) -> List[Dict]:
    """
    Loads a JSON file and returns its content as a list of dictionaries.

    Args:
        path (str): The path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries containing the content of the JSON file.

    Raises:
        AssertionError: If the top-level JSON is not a list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert isinstance(data, list), "Top-level JSON must be a list!"
        return data