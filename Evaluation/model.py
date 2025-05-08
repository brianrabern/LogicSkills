import json
import logging
from Evaluation.llm import prompt_model
from Evaluation.prompts.json_fixer_prompt import fixer_prompt
from config import JSON_FIXER_MODEL


class Model:
    def __init__(self, model_name, system_prompt=None):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def query(self, prompt, parse_json=False):
        response = prompt_model(self.model_name, prompt)
        if not parse_json:
            return response

        json_data = self.parse_json(response)
        if json_data is not None:
            return json_data

        logging.info("Initial JSON parse failed. Attempting to fix...")
        return self.fix_json_response(response)

    def parse_json(self, response_text):
        if not response_text:
            print("No response text to parse.")
            return {"raw_response": None}
        try:
            if "```json" in response_text:
                start = response_text.find("```json") + len("```json")
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.info(f"Error decoding JSON: {e}")
            return None

    def fix_json_response(self, broken_output: str):
        fixed = prompt_model(JSON_FIXER_MODEL, fixer_prompt(broken_output))
        print(f"[{JSON_FIXER_MODEL} fixed] {fixed}")

        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logging.error(f"Could not fix JSON: {e}")
            return {"raw_response": fixed}
