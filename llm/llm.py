# llm.py
"""
LLM module.
Swap LLM_PROVIDER in config.py to change the backend.
Contract: generate(prompt: str) -> str
"""
import logging
import time
import requests
import config


logger = logging.getLogger(__name__)

def generate(prompt: str) -> str:
    # logger.debug("LLM generate: provider=%s  model=%s  prompt_len=%d chars",
    #             config.LLM_PROVIDER, config.LLM_MODEL, len(prompt))

    start = time.perf_counter()
    if config.LLM_PROVIDER == "ollama":
        answer = _generate_ollama(prompt)
    elif config.LLM_PROVIDER == "openai_compatible":
        answer = _generate_openai_compatible(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")

    elapsed = time.perf_counter() - start
    # logger.info("LLM generate complete: model=%s  response_len=%d chars  elapsed=%.2fs",
    #             config.LLM_MODEL, len(answer), elapsed)
    logger.debug("LLM response preview: %s", answer[:200])
    return answer


# ── Backend A: Ollama ──────────────────────────────────
def _generate_ollama(prompt: str) -> str:
    url = f"{config.LLM_BASE_URL}/api/chat"
    logger.debug("POST %s  model=%s", url, config.LLM_MODEL)

    try:
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
    except requests.exceptions.HTTPError as e:
        logger.error("Ollama generate request failed: %s", e)
        raise

    return response.json()["message"]["content"].strip()


# ── Backend B: Any OpenAI-compatible endpoint ──────────
# Works with: LM Studio, vLLM, Together AI, Groq, etc.
def _generate_openai_compatible(prompt: str) -> str:
    import os
    llm_base_url = os.environ['LLM_BASE_URL']
    llm_api_key = os.environ['LLM_API_KEY']
    llm_model = os.environ['LLM_MODEL']
    url = f"{llm_base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {llm_api_key}"}
    logger.debug("POST %s  model=%s", url, llm_model)

    try:
        for attempt in range(3):
            response = requests.post(url, headers=headers, json={
                "model": llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
            }   )
            if response.status_code == 429:
                wait = 12 * (attempt + 1)  # 12s, 24s, 36s — stays within 5 RPM window
                logger.warning("429 from LLM Provider — waiting %ds before retry (attempt %d/3)", wait, attempt + 1)
                time.sleep(wait)
                continue
            response.raise_for_status()
            break        
    except requests.exceptions.HTTPError as e:
        logger.error("OpenAI-compatible generate request failed: %s", e)
        raise

    # return response.json()["choices"][0]["message"]["content"].strip()
    try:
        return _extract_content(response.json())
    except Exception as e:
        logger.error("Failed to parse LLM response: %s\nRaw: %s", e, response.text[:500])
        raise
    
def _extract_content(response_json: dict) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        logger.error("Empty choices in LLM response: %s", response_json)
        return "LLM returned no content."
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        # Google AI Studio returns content as list of parts when filtered
        logger.warning("LLM content is a list — extracting text parts: %s", content)
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return (content or "").strip()