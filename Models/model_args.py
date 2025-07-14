"""
Data class to handle model arguments.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PromptArgs:
    system_message: bool = False


@dataclass
class ModelArgs:
    init_kwargs: Dict[str, Any]
    inference_kwargs: Dict[str, Any]
    system_message: bool = False
