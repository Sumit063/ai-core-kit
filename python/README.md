# Python

## Setup
- Create a `.env` at the repo root with `OPENAI_API_KEY` (see `.env.example`).
- From `python/`:
  - `python -m venv .venv`
  - `./.venv/Scripts/activate` (Windows) or `source .venv/bin/activate`
  - `pip install -e .`

## Commands
- `python -m cli.main generate --prompt "Write a one-line slogan."`
- `python -m cli.main structured --prompt "Describe a product idea."`
- `python -m cli.main index --docs_dir ./sample_docs --out ./data/vectorstore.json`
- `python -m cli.main search --query "What is this project?" --store ./data/vectorstore.json --top_k 5`
- `python -m cli.main rag --query "Summarize the docs." --store ./data/vectorstore.json --top_k 5`

If you skip `pip install -e .`, prefix commands with `PYTHONPATH=src`.