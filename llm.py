# llm.py
"""
LLM module.
Swap LLM_PROVIDER in config.py to change the backend.
Contract: generate(prompt: str) -> str
"""

import requests
import config


def generate(prompt: str) -> str:
    if config.LLM_PROVIDER == "ollama":
        return _generate_ollama(prompt)
    elif config.LLM_PROVIDER == "openai_compatible":
        return _generate_openai_compatible(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


# ── Backend A: Ollama ──────────────────────────────────
def _generate_ollama(prompt: str) -> str:
    url = f"{config.LLM_BASE_URL}/api/chat"
    response = requests.post(url, json={
        "model":  config.LLM_MODEL,
        # "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_MAX_TOKENS,
        },
    })
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ── Backend B: Any OpenAI-compatible endpoint ──────────
# Works with: LM Studio, vLLM, Together AI, Groq, etc.
def _generate_openai_compatible(prompt: str) -> str:
    import os
    url = f"{config.LLM_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'none')}"}
    response = requests.post(url, headers=headers, json={
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    })
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()