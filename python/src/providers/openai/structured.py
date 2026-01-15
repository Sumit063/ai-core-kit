import json
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ValidationError

from common.config import Settings, require_openai

_SYSTEM_PROMPT = (
    "Return ONLY valid JSON with keys: title, summary, keywords. "
    "Do not wrap the JSON in markdown or add extra text."
)


class StructuredOutput(BaseModel):
    title: str
    summary: str
    keywords: list[str]


def _call_openai(messages: list[dict[str, str]], settings: Settings) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "messages": messages,
        "temperature": 0,
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=settings.timeout_s)
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected response from OpenAI.") from exc
    return content.strip()


def _strip_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _validate_output(data: Any) -> StructuredOutput:
    try:
        return StructuredOutput.model_validate(data)
    except AttributeError:
        return StructuredOutput.parse_obj(data)


def structured_json(prompt: str, settings: Settings, max_retries: int = 2) -> dict[str, Any]:
    require_openai(settings)
    base_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    last_error: Optional[Exception] = None
    messages = base_messages
    for attempt in range(max_retries + 1):
        content = _call_openai(messages, settings)
        try:
            payload = json.loads(_strip_json(content))
            validated = _validate_output(payload)
            try:
                return validated.model_dump()
            except AttributeError:
                return validated.dict()
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            correction = (
                "The previous response was invalid JSON or did not match the schema. "
                f"Error: {exc}. Return ONLY valid JSON with keys: title, summary, keywords."
            )
            messages = base_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": correction},
            ]

    raise ValueError(f"Structured output validation failed: {last_error}")
