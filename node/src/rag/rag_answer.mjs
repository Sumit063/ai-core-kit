import { generate } from "../providers/openai/generate.mjs";

const CITATION_PATTERN = /\[([^\[\]]+?:\d+)\]/g;

// answerWithCitations builds a prompt and extracts inline citations.
export async function answerWithCitations(query, chunks, settings) {
  const prompt = buildPrompt(query, chunks);
  const answer = await generate(prompt, settings);
  const citations = extractCitations(answer);
  return { answer, citations };
}

function buildPrompt(query, chunks) {
  const contextLines = chunks.map((chunk) => {
    const tag = `[${chunk.source}:${chunk.id}]`;
    return `${tag} ${chunk.text}`;
  });
  const contextBlock = contextLines.join("\n\n");

  return (
    "Answer the question using only the context below. " +
    "Cite sources in-line as [source:id]. If the answer is not in the context, say you do not know.\n\n" +
    `Context:\n${contextBlock}\n\n` +
    `Question: ${query}\n` +
    "Answer:"
  );
}

function extractCitations(answer) {
  const citations = new Set();
  let match = CITATION_PATTERN.exec(answer);
  while (match) {
    citations.add(match[1]);
    match = CITATION_PATTERN.exec(answer);
  }
  return Array.from(citations).sort();
}