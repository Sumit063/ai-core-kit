import { requireOpenAI } from "../../common/config.js";

const CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions";

/**
 * generate sends a single-turn prompt to OpenAI chat completions.
 */
export async function generate(prompt, settings) {
  requireOpenAI(settings);
  const payload = {
    model: settings.openaiModel,
    messages: [{ role: "user", content: prompt }],
    temperature: 0.2,
  };

  const data = await postJson(CHAT_COMPLETIONS_URL, payload, settings);
  const content = data?.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("Unexpected response from OpenAI.");
  }
  return content.trim();
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