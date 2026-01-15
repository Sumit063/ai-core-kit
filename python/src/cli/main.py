import argparse
import json
from pathlib import Path
from typing import Iterable

from common.config import get_settings, require_openai
from providers.openai.embeddings import embed_texts
from providers.openai.generate import generate
from providers.openai.structured import structured_json
from rag.chunking import chunk_text
from rag.rag_answer import answer_with_citations
from rag.vectorstore import load_store, save_store, search


def _iter_text_files(docs_dir: Path) -> Iterable[Path]:
    for path in docs_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".markdown"}:
            yield path


def _batched(items: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def cmd_generate(args: argparse.Namespace) -> None:
    settings = get_settings()
    require_openai(settings)
    output = generate(args.prompt, settings)
    print(output)


def cmd_structured(args: argparse.Namespace) -> None:
    settings = get_settings()
    require_openai(settings)
    payload = structured_json(args.prompt, settings)
    print(json.dumps(payload, indent=2))


def cmd_index(args: argparse.Namespace) -> None:
    settings = get_settings()
    require_openai(settings)

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    chunk_records: list[dict[str, str]] = []
    for path in _iter_text_files(docs_dir):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_text(text):
            chunk_records.append({"source": path.name, "text": chunk})

    if not chunk_records:
        raise ValueError("No text files found to index.")

    entries = []
    next_id = 0
    for batch in _batched(chunk_records, size=32):
        embeddings = embed_texts([item["text"] for item in batch], settings)
        for item, embedding in zip(batch, embeddings):
            entries.append({
                "id": next_id,
                "source": item["source"],
                "text": item["text"],
                "embedding": embedding,
            })
            next_id += 1

    save_store(args.out, entries)
    print(f"Indexed {len(entries)} chunks -> {args.out}")


def cmd_search(args: argparse.Namespace) -> None:
    settings = get_settings()
    require_openai(settings)

    entries = load_store(args.store)
    query_embedding = embed_texts([args.query], settings)[0]
    results = search(entries, query_embedding, top_k=args.top_k)
    print(json.dumps(results, indent=2))


def cmd_rag(args: argparse.Namespace) -> None:
    settings = get_settings()
    require_openai(settings)

    entries = load_store(args.store)
    query_embedding = embed_texts([args.query], settings)[0]
    results = search(entries, query_embedding, top_k=args.top_k)
    answer, citations = answer_with_citations(args.query, results, settings)
    print(answer)
    if citations:
        print("\nCitations:")
        print(json.dumps(citations, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ai-core-kit Python CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate text")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.set_defaults(func=cmd_generate)

    structured_parser = subparsers.add_parser("structured", help="Structured JSON output")
    structured_parser.add_argument("--prompt", required=True)
    structured_parser.set_defaults(func=cmd_structured)

    index_parser = subparsers.add_parser("index", help="Index a docs folder")
    index_parser.add_argument("--docs_dir", default="./sample_docs")
    index_parser.add_argument("--out", default="./data/vectorstore.json")
    index_parser.set_defaults(func=cmd_index)

    search_parser = subparsers.add_parser("search", help="Search a vector store")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--store", default="./data/vectorstore.json")
    search_parser.add_argument("--top_k", type=int, default=5)
    search_parser.set_defaults(func=cmd_search)

    rag_parser = subparsers.add_parser("rag", help="Answer with citations")
    rag_parser.add_argument("--query", required=True)
    rag_parser.add_argument("--store", default="./data/vectorstore.json")
    rag_parser.add_argument("--top_k", type=int, default=5)
    rag_parser.set_defaults(func=cmd_rag)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()