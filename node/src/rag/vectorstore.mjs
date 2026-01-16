import fs from "fs/promises";
import path from "path";

// loadStore reads a JSON vector store from disk.
export async function loadStore(storePath) {
  const data = await fs.readFile(storePath, "utf-8");
  return JSON.parse(data);
}

// saveStore writes entries to a JSON vector store.
export async function saveStore(storePath, entries) {
  await fs.mkdir(path.dirname(storePath), { recursive: true });
  const payload = JSON.stringify(entries, null, 2);
  await fs.writeFile(storePath, payload, "utf-8");
}

// search returns the top-k most similar entries.
export function search(entries, queryEmbedding, topK = 5) {
  const scored = entries.map((entry) => ({
    id: entry.id,
    source: entry.source,
    text: entry.text,
    score: cosineSimilarity(queryEmbedding, entry.embedding),
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, topK));
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}