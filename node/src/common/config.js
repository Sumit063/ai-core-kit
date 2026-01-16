import fs from "fs";
import path from "path";
import dotenv from "dotenv";

// Load .env from the nearest parent directory.
function findEnvPath(startDir) {
  let current = path.resolve(startDir);
  while (true) {
    const candidate = path.join(current, ".env");
    if (fs.existsSync(candidate)) {
      return candidate;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return null;
    }
    current = parent;
  }
}

const envPath = findEnvPath(process.cwd());
if (envPath) {
  dotenv.config({ path: envPath });
} else {
  dotenv.config();
}

// loadConfig returns validated runtime settings.
export function loadConfig() {
  const timeoutRaw = process.env.REQUEST_TIMEOUT_SECONDS || "30";
  const timeoutSeconds = Number(timeoutRaw);
  if (Number.isNaN(timeoutSeconds)) {
    throw new Error("REQUEST_TIMEOUT_SECONDS must be a number.");
  }

  return {
    openaiApiKey: process.env.OPENAI_API_KEY || "",
    openaiModel: process.env.OPENAI_MODEL || "gpt-4o-mini",
    openaiEmbedModel: process.env.OPENAI_EMBED_MODEL || "text-embedding-3-small",
    timeoutMs: timeoutSeconds * 1000,
  };
}

// requireOpenAI enforces that the OpenAI API key is present.
export function requireOpenAI(settings) {
  if (!settings.openaiApiKey) {
    throw new Error("OPENAI_API_KEY is required for this command.");
  }
}