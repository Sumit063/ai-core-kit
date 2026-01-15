import json
import math
from pathlib import Path
from typing import Any, Iterable, Union


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding sizes do not match.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_store(path: Union[str, Path]) -> list[dict[str, Any]]:
    store_path = Path(path)
    if not store_path.exists():
        raise FileNotFoundError(f"Vector store not found: {store_path}")
    return json.loads(store_path.read_text(encoding="utf-8"))


def save_store(path: Union[str, Path], entries: Iterable[dict[str, Any]]) -> None:
    store_path = Path(path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(list(entries), indent=2), encoding="utf-8")


def search(
    entries: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for entry in entries:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored.append({
            "id": entry["id"],
            "source": entry["source"],
            "text": entry["text"],
            "score": score,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
