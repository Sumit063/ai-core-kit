import os
from dataclasses import dataclass
from typing import Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str]
    openai_model: str
    openai_embed_model: str
    timeout_s: float


def get_settings() -> Settings:
    timeout_raw = os.getenv("REQUEST_TIMEOUT_SECONDS", "30")
    try:
        timeout_s = float(timeout_raw)
    except ValueError as exc:
        raise ValueError("REQUEST_TIMEOUT_SECONDS must be a number.") from exc

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        timeout_s=timeout_s,
    )


def require_openai(settings: Settings) -> None:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for this command.")
