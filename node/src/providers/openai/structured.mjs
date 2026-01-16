import { requireOpenAI } from "../../common/config.js";

const CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions";
const SYSTEM_PROMPT =
  "Return ONLY valid JSON with keys: title, summary, keywords. Do not wrap the JSON in markdown or add extra text.";

/**
 * structuredJson requests a schema-validated JSON response.
 */
export async function structuredJson(prompt, settings, maxRetries = 2) {
  requireOpenAI(settings);
  const baseMessages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: prompt },
  ];

  let lastError = null;
  let messages = baseMessages;
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const content = await chatCompletion(messages, settings, 0);
    try {
      const payload = JSON.parse(stripJsonFence(content));
      validateStructured(payload);
      return payload;
    } catch (err) {
      lastError = err;
      if (attempt >= maxRetries) {
        break;
      }
      const correction =
        `The previous response was invalid JSON or did not match the schema. ` +
        `Error: ${err}. Return ONLY valid JSON with keys: title, summary, keywords.`;
      messages = [
        ...baseMessages,
        { role: "assistant", content },
        { role: "user", content: correction },
      ];
    }
  }

  throw new Error(`Structured output validation failed: ${lastError}`);
}

async function chatCompletion(messages, settings, temperature) {
  const payload = {
    model: settings.openaiModel,
    messages,
    temperature,
  };

  const data = await postJson(CHAT_COMPLETIONS_URL, payload, settings);
  const content = data?.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("Unexpected response from OpenAI.");
  }
  return content.trim();
}

function stripJsonFence(text) {
  let cleaned = text.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/```/g, "").trim();
    if (cleaned.toLowerCase().startsWith("json")) {
      cleaned = cleaned.slice(4).trim();
    }
  }
  return cleaned;
}

function validateStructured(payload) {
  if (!payload || typeof payload !== "object") {
    throw new Error("Response is not an object.");
  }
  if (typeof payload.title !== "string" || payload.title.trim() === "") {
    throw new Error("Missing 'title'.");
  }
  if (typeof payload.summary !== "string" || payload.summary.trim() === "") {
    throw new Error("Missing 'summary'.");
  }
  if (!Array.isArray(payload.keywords)) {
    throw new Error("Missing 'keywords'.");
  }
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