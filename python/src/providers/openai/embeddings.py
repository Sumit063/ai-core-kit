from typing import List

import httpx

from common.config import Settings, require_openai


def embed_texts(texts: List[str], settings: Settings) -> List[List[float]]:
    require_openai(settings)
    if not texts:
        return []

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_embed_model,
        "input": texts,
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=settings.timeout_s)
    response.raise_for_status()
    data = response.json()
    try:
        embeddings = [item["embedding"] for item in data["data"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Unexpected response from OpenAI embeddings.") from exc
    return embeddings