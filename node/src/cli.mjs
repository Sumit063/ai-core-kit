import fs from "fs/promises";
import path from "path";

import { loadConfig, requireOpenAI } from "./common/config.js";
import { embedTexts } from "./providers/openai/embeddings.mjs";
import { generate } from "./providers/openai/generate.mjs";
import { structuredJson } from "./providers/openai/structured.mjs";
import { chunkText } from "./rag/chunking.mjs";
import { answerWithCitations } from "./rag/rag_answer.mjs";
import { loadStore, saveStore, search } from "./rag/vectorstore.mjs";

const DEFAULT_DOCS_DIR = "./sample_docs";
const DEFAULT_STORE = "./data/vectorstore.json";
const DEFAULT_TOP_K = 5;

const [command, ...rest] = process.argv.slice(2);

if (!command || command === "--help" || command === "-h" || command === "help") {
  printUsage();
  process.exit(0);
}

const options = parseArgs(rest);

try {
  if (command === "generate") {
    await runGenerate(options);
  } else if (command === "structured") {
    await runStructured(options);
  } else if (command === "index") {
    await runIndex(options);
  } else if (command === "search") {
    await runSearch(options);
  } else if (command === "rag") {
    await runRag(options);
  } else {
    throw new Error(`Unknown command: ${command}`);
  }
} catch (err) {
  exitWithError(err.message || String(err));
}

async function runGenerate(options) {
  const prompt = (options.prompt || "").trim();
  if (!prompt) {
    throw new Error("generate: --prompt is required");
  }

  const settings = loadSettings();
  const output = await generate(prompt, settings);
  console.log(output);
}

async function runStructured(options) {
  const prompt = (options.prompt || "").trim();
  if (!prompt) {
    throw new Error("structured: --prompt is required");
  }

  const settings = loadSettings();
  const payload = await structuredJson(prompt, settings);
  console.log(JSON.stringify(payload, null, 2));
}

async function runIndex(options) {
  const docsDir = options.docs_dir || DEFAULT_DOCS_DIR;
  const store = options.store || DEFAULT_STORE;

  const settings = loadSettings();
  const records = await collectChunks(docsDir);
  if (records.length === 0) {
    throw new Error("No text files found to index.");
  }

  const entries = [];
  let nextId = 0;
  const batchSize = 32;
  for (let i = 0; i < records.length; i += batchSize) {
    const batch = records.slice(i, i + batchSize);
    const embeddings = await embedTexts(
      batch.map((record) => record.text),
      settings,
    );

    batch.forEach((record, idx) => {
      entries.push({
        id: nextId,
        source: record.source,
        text: record.text,
        embedding: embeddings[idx],
      });
      nextId += 1;
    });
  }

  await saveStore(store, entries);
  console.log(`Indexed ${entries.length} chunks -> ${store}`);
}

async function runSearch(options) {
  const query = (options.query || "").trim();
  const store = options.store || DEFAULT_STORE;
  const topK = Number(options.top_k || DEFAULT_TOP_K);

  if (!query) {
    throw new Error("search: --query is required");
  }

  const settings = loadSettings();
  const entries = await loadStore(store);
  const [queryEmbedding] = await embedTexts([query], settings);
  const results = search(entries, queryEmbedding, topK);
  console.log(JSON.stringify(results, null, 2));
}

async function runRag(options) {
  const query = (options.query || "").trim();
  const store = options.store || DEFAULT_STORE;
  const topK = Number(options.top_k || DEFAULT_TOP_K);

  if (!query) {
    throw new Error("rag: --query is required");
  }

  const settings = loadSettings();
  const entries = await loadStore(store);
  const [queryEmbedding] = await embedTexts([query], settings);
  const results = search(entries, queryEmbedding, topK);
  const { answer, citations } = await answerWithCitations(query, results, settings);

  console.log(answer);
  if (citations.length) {
    console.log("\nCitations:");
    console.log(JSON.stringify(citations, null, 2));
  }
}

function loadSettings() {
  const settings = loadConfig();
  requireOpenAI(settings);
  return settings;
}

function parseArgs(args) {
  const options = {};
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (!arg.startsWith("--")) {
      continue;
    }
    const key = arg.slice(2);
    const next = args[i + 1];
    if (!next || next.startsWith("--")) {
      options[key] = true;
    } else {
      options[key] = next;
      i += 1;
    }
  }
  return options;
}

async function collectChunks(docsDir) {
  const files = await walkDir(docsDir);
  const records = [];

  for (const filePath of files) {
    const ext = path.extname(filePath).toLowerCase();
    if (![".txt", ".md", ".markdown"].includes(ext)) {
      continue;
    }
    const content = await fs.readFile(filePath, "utf-8");
    const chunks = chunkText(content, 800, 150);
    chunks.forEach((chunk) => {
      records.push({
        source: path.basename(filePath),
        text: chunk,
      });
    });
  }

  return records;
}

async function walkDir(dirPath) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      const nested = await walkDir(fullPath);
      files.push(...nested);
    } else {
      files.push(fullPath);
    }
  }

  return files;
}

function printUsage() {
  console.log("Usage: node src/cli.mjs <command> [options]");
  console.log("Commands: generate, structured, index, search, rag");
  console.log("Example: node src/cli.mjs generate --prompt \"Hello\"");
}

function exitWithError(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}