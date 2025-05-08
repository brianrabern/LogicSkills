from openai import OpenAI
from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

BASE_URL = OPENROUTER_BASE_URL
API_KEY = OPENROUTER_API_KEY
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def prompt_model(model_name: str, prompt: str):
    try:
        print(f"{model_name} is thinking...")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,  # Give more space for logical reasoning
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error prompting model: {e}")
        return None
