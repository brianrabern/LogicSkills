"""
Data class and modules for SFT data.
"""

from typing import Any, Dict, List, Literal

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from Assessors.core.prompt_prep import assemble_chat_messages
from Assessors.countermodel.prompts.system import system_prompt as countermodel_system_prompt
from Assessors.symbolization.prompts.system import system_prompt as symbolization_system_prompt
from Assessors.validity.prompts.system import system_prompt as validity_system_prompt


class SFTDataset(Dataset):
    """
    Builds inputs so that labels are masked for the prompt and only the assistant answer tokens contribute to the loss.
    """

    def __init__(
        self,
        rows: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int,
        question_type: Literal["validity", "symbolization", "countermodel"],
        accepts_system_prompt: bool = True,
    ):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.accepts_system_prompt = accepts_system_prompt
        self.system_prompt = self.load_system_prompt(question_type)

    def load_system_prompt(self, question_type: Literal["validity", "symbolization", "countermodel"]) -> str:
        """
        Loads the system prompt string for the given question type.

        Returns:
            A string containing the system prompt.

        Raises:
            ValueError: If the question type is unknown.
        """
        if question_type == "validity":
            return validity_system_prompt
        elif question_type == "symbolization":
            return symbolization_system_prompt
        elif question_type == "countermodel":
            return countermodel_system_prompt
        else:
            raise ValueError(
                f"Unknown question type: {question_type}. Available: ['validity', 'symbolization', 'countermodel']"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_example(self, task: str, answer: str) -> Dict[str, Any]:
        """
        Encodes a single example from the dataset.

        Args:
            task: str
                The task to encode.
            answer: str
                The answer to encode.

        Returns:
            A dictionary containing the encoded example.
        """
        messages = assemble_chat_messages(
            prompt=task, accepts_system_prompt=self.accepts_system_prompt, system_prompt=self.system_prompt
        )

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tok(answer, add_special_tokens=False).input_ids

        if len(prompt_ids) + len(answer_ids) > self.max_length:
            raise ValueError(
                f"prompt_ids+answer_ids > max_length! {len(prompt_ids) + len(answer_ids)} > {self.max_length}"
            )

        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        attention_mask = [1] * len(input_ids)

        # Labels: mask prompt tokens and padding with -100
        labels = [-100] * len(prompt_ids) + answer_ids
        labels = labels[: self.max_length]

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tok.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.rows[idx]
        return self._encode_example(row["task"], row["answer"])
