import { requireOpenAI } from "../../common/config.js";

const EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings";

/**
 * embedTexts returns embeddings for a batch of texts.
 */
export async function embedTexts(texts, settings) {
  requireOpenAI(settings);
  if (!texts || texts.length === 0) {
    return [];
  }

  const payload = {
    model: settings.openaiEmbedModel,
    input: texts,
  };

  const data = await postJson(EMBEDDINGS_URL, payload, settings);
  const embeddings = data?.data?.map((item) => item.embedding);
  if (!embeddings) {
    throw new Error("Unexpected response from OpenAI embeddings.");
  }
  return embeddings;
}

async function postJson(url, payload, settings) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), settings.timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${settings.openaiApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const text = await response.text();
    if (!response.ok) {
      throw new Error(`OpenAI request failed (${response.status}): ${text}`);
    }

    return JSON.parse(text);
  } finally {
    clearTimeout(timeoutId);
  }
}