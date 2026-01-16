// chunkText splits text into overlapping chunks for embedding.
export function chunkText(text, chunkSize = 800, overlap = 150) {
  const cleaned = text.split(/\s+/).filter(Boolean).join(" ");
  if (!cleaned) {
    return [];
  }

  const chunks = [];
  let start = 0;
  while (start < cleaned.length) {
    const end = Math.min(start + chunkSize, cleaned.length);
    chunks.push(cleaned.slice(start, end));
    if (end >= cleaned.length) {
      break;
    }
    start = Math.max(0, end - overlap);
  }

  return chunks;
}