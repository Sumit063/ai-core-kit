# Go

## Setup
- Set `OPENAI_API_KEY` in your environment.
- Optional: set `OPENAI_MODEL`, `OPENAI_EMBED_MODEL`, `REQUEST_TIMEOUT_SECONDS`.

## Commands
- `go run ./cmd/cli generate --prompt "Write a one-line slogan."`
- `go run ./cmd/cli structured --prompt "Describe a product idea."`
- `go run ./cmd/cli index --docs_dir ./sample_docs --store ./data/vectorstore.json`
- `go run ./cmd/cli search --query "What is this project?" --store ./data/vectorstore.json --top_k 5`
- `go run ./cmd/cli rag --query "Summarize the docs." --store ./data/vectorstore.json --top_k 5`
