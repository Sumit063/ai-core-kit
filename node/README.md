# Node.js

## Setup
- Create a `.env` at the repo root with `OPENAI_API_KEY` (see `.env.example`).
- From `node/`:
  - `npm install`

## Commands
- `node src/cli.mjs generate --prompt "Write a one-line slogan."`
- `node src/cli.mjs structured --prompt "Describe a product idea."`
- `node src/cli.mjs index --docs_dir ./sample_docs --store ./data/vectorstore.json`
- `node src/cli.mjs search --query "What is this project?" --store ./data/vectorstore.json --top_k 5`
- `node src/cli.mjs rag --query "Summarize the docs." --store ./data/vectorstore.json --top_k 5`