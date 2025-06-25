"""
Functions and methods to prepare prompts for model evaluation.
"""

from typing import List, Dict
from transformers import PreTrainedTokenizerBase


def assemble_chat_messages(
    prompt: str,
    accepts_system_prompt: bool,
    system_prompt: str | None,
) -> List[Dict[str, str]]:
    """
    Assembles chat messages for a conversation prompt, optionally including a system prompt.

    Args:
        prompt (str): The main user prompt to be included in the chat messages.
        accepts_system_prompt (bool): Flag indicating if the system prompt should be included.
        system_prompt (str | None): The system prompt to include if `accepts_system_prompt` is True.

    Returns:
        List[Dict[str, str]]: A list of chat message dictionaries, each containing a "role" and "content",
                              with the user prompt and optionally the system prompt.
    """
    chat_prompt = [{"role": "user", "content": prompt}]

    # handle system prompt
    if system_prompt is not None:
        if accepts_system_prompt:
            system_chat_prompt = [
                {"role": "system", "content": system_prompt}
            ]
            chat_prompt = system_chat_prompt + chat_prompt
        else:
            chat_prompt[0][
                "content"
            ] = f"{system_prompt}\n\n{chat_prompt[0]['content']}"

    return chat_prompt    


def build_api_chat_prompts(
    prompts: List[str],
    accepts_system_prompt: bool,
    system_prompt: str | None,
) -> List[str]:
    """
    Assembles a list of chat prompts to be passed to the OpenAI API.

    Args:
        prompts (List[str]): The list of prompts to be converted to chat prompts.
        accepts_system_prompt (bool): Flag indicating if the system prompt should be included.
        system_prompt (str | None): The system prompt to include if `accepts_system_prompt` is True.

    Returns:
        List[str]: A list of strings, each representing a chat prompt for the OpenAI API.
    """
    chat_prompts: List[str] = []

    for prompt in prompts:
        chat_messages = assemble_chat_messages(
            prompt=prompt,
            accepts_system_prompt=accepts_system_prompt,
            system_prompt=system_prompt
        )
        chat_prompts.append(chat_messages)
    
    return chat_prompts


def build_local_chat_prompts(
    prompts: List[str],
    tokenizer: PreTrainedTokenizerBase | None,
    accepts_system_prompt: bool,
    system_prompt: str | None,
) -> List[str]:
    """
    Assembles a list of chat prompts to be passed to a local model.

    Args:
        prompts (List[str]): The list of prompts to be converted to chat prompts.
        tokenizer (PreTrainedTokenizerBase | None): The tokenizer to use for converting the
            prompts to chat prompts.
        accepts_system_prompt (bool): Flag indicating if the system prompt should be included.
        system_prompt (str | None): The system prompt to include if `accepts_system_prompt` is True.

    Returns:
        List[str]: A list of strings, each representing a chat prompt for a local model.
    """
    chat_prompts: List[str] = []

    for prompt in prompts:
        chat_messages = assemble_chat_messages(
            prompt=prompt,
            accepts_system_prompt=accepts_system_prompt,
            system_prompt=system_prompt
        )

        chat_template_prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors=False,
        )
        chat_prompts.append(chat_template_prompt)
    
    return chat_prompts