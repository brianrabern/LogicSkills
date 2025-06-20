import logging
from typing import Dict, Any, Optional
from datetime import datetime
from Assessors.core.llm import prompt_model


class ResponseEngine:
    """Handles the inference stage - getting raw responses from models."""

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2500,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.inference_log = []

    def query(self, prompt: str) -> Dict[str, Any]:
        """Get raw response from model."""

        start_time = datetime.now()

        try:
            raw_response = prompt_model(
                self.model_name,
                prompt,
                self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )

            inference_metadata = {
                "model_name": self.model_name,
                "system_prompt": self.system_prompt,
                "model_parameters": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                },
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "success": raw_response is not None,
            }

            result = {
                "raw_response": raw_response,
                "inference_metadata": inference_metadata,
                "success": raw_response is not None,
            }

            # Log inference
            self.inference_log.append({"prompt": prompt, "result": result})

            return result

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return {
                "prompt": prompt,
                "raw_response": None,
                "inference_metadata": inference_metadata,
                "success": False,
            }

    def get_inference_log(self) -> list:
        """Get the log of all inference operations."""
        return self.inference_log


# Test section - run this file directly to test the InferenceEngine
if __name__ == "__main__":
    print("Testing InferenceEngine...")
    print("Test 1: Simple query with default parameters")
    engine = ResponseEngine("openai/gpt-4o-mini")
    result = engine.query("What is 2+2?")

    print(f"Prompt: {result['prompt']}")
    print(f"Success: {result['success']}")
    print(f"Raw Response: {result['raw_response']}")
    print(f"Model: {result['inference_metadata']['model_name']}")
    print(f"Temperature: {result['inference_metadata']['model_parameters']['temperature']}")
    print(f"Max Tokens: {result['inference_metadata']['model_parameters']['max_tokens']}")
    print(f"Start Time: {result['inference_metadata']['start_time']}")
    print(f"End Time: {result['inference_metadata']['end_time']}")
    print()
