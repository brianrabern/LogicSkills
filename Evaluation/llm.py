from openai import OpenAI
from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

BASE_URL = OPENROUTER_BASE_URL
API_KEY = OPENROUTER_API_KEY
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def prompt_model(model_name: str, prompt: str, system_prompt: str = None):
    try:
        print(f"{model_name} is thinking...")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        print(f"System: {system_prompt}" if system_prompt else "No system prompt")
        print(f"User: {prompt}")

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=2500,
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error prompting model: {e}")
        return None
