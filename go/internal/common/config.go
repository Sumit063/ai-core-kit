package common

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// holds runtime configuration from environment variables.
type Settings struct {
	OpenAIAPIKey     string
	OpenAIModel      string
	OpenAIEmbedModel string
	Timeout          time.Duration
}

// reads configuration from environment variables.
func LoadSettings() (Settings, error) {
	timeoutRaw := os.Getenv("REQUEST_TIMEOUT_SECONDS")
	if timeoutRaw == "" {
		timeoutRaw = "30"
	}

	timeoutSec, err := strconv.ParseFloat(timeoutRaw, 64)
	if err != nil {
		return Settings{}, fmt.Errorf("REQUEST_TIMEOUT_SECONDS must be a number: %w", err)
	}

	return Settings{
		OpenAIAPIKey:     os.Getenv("OPENAI_API_KEY"),
		OpenAIModel:      envOrDefault("OPENAI_MODEL", "gpt-4o-mini"),
		OpenAIEmbedModel: envOrDefault("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
		Timeout:          time.Duration(timeoutSec * float64(time.Second)),
	}, nil
}

// validates that an OpenAI API key is present.
func RequireOpenAI(settings Settings) error {
	if settings.OpenAIAPIKey == "" {
		return fmt.Errorf("OPENAI_API_KEY is required for this command")
	}
	return nil
}

func envOrDefault(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}
