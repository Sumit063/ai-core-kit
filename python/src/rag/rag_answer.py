import re
from typing import Any

from common.config import Settings
from providers.openai.generate import generate


def build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    context_lines = []
    for chunk in chunks:
        tag = f"[{chunk['source']}:{chunk['id']}]"
        context_lines.append(f"{tag} {chunk['text']}")
    context_block = "\n\n".join(context_lines)

    return (
        "Answer the question using only the context below. "
        "Cite sources in-line as [source:id]. If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def extract_citations(answer: str) -> list[str]:
    citations = re.findall(r"\[([^\[\]]+?:\d+)\]", answer)
    return sorted(set(citations))


def answer_with_citations(
    query: str,
    chunks: list[dict[str, Any]],
    settings: Settings,
) -> tuple[str, list[str]]:
    prompt = build_prompt(query, chunks)
    answer = generate(prompt, settings)
    citations = extract_citations(answer)
    return answer, citations