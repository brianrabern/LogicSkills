"""
Specifies the type of LLM we consider (locally deployed vs. API called).
"""

LOCAL_MODEL = {
    "meta-llama/Llama-3.1-70B-Instruct": True,
    "meta-llama/Llama-3.1-8B-Instruct": True,
    "meta-llama/Llama-3.2-3B-Instruct": True,
    "meta-llama/Llama-3.2-3B-Instruct-sft-countermodel": True,
    "meta-llama/Llama-3.2-3B-Instruct-sft-countermodel-symbolization": True,
    "meta-llama/Llama-3.2-3B-Instruct-sft-symbolization": True,
    "google/gemma-3-27b-it": True,
    "Qwen/Qwen2.5-3B-Instruct": True,
    "Qwen/Qwen2.5-7B-Instruct": True,
    "Qwen/Qwen2.5-32B-Instruct": True,
    "Qwen/Qwen2.5-72B-Instruct": True,
    "Qwen/Qwen2.5-Math-72B-Instruct": True,
    "Qwen/Qwen3-32B": True,
    "microsoft/phi-4": True,
    "open-thoughts/OpenThinker2-32B": True,
    "openai/gpt-oss-20b": True,
    "openai/gpt-oss-120b": True,
    "openai/gpt-4o-mini": False,
    "anthropic/claude-3.7-sonnet": False,
    "google/gemini-2.5-flash": False,
    "openai/gpt-5-mini": False,
}
