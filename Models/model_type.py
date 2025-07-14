"""
Specifies the type of LLM we consider (locally deployed vs. API called).
"""

LOCAL_MODEL = {
    "meta-llama/Llama-3.1-70B-Instruct": True,
    "meta-llama/Llama-3.1-8B-Instruct": True,
    "meta-llama/Llama-3.2-3B-Instruct": True,
    "google/gemma-3-27b-it": True,
    "Qwen/Qwen2.5-32B-Instruct": True,
    "Qwen/Qwen2.5-72B-Instruct": True,
    "Qwen/Qwen3-32B": True,
    "openai/gpt-4o-mini": False,
}
