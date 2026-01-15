import httpx

from common.config import Settings, require_openai


def generate(prompt: str, settings: Settings) -> str:
    require_openai(settings)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=settings.timeout_s)
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected response from OpenAI.") from exc
    return content.strip()