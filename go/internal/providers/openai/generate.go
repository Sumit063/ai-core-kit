package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"ai-core-kit-go/internal/common"
	"ai-core-kit-go/internal/providers"
)

const chatCompletionsURL = "https://api.openai.com/v1/chat/completions"

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

// sends a single-turn prompt to OpenAI chat completions.
func Generate(ctx context.Context, prompt string, settings common.Settings) (string, error) {
	messages := []chatMessage{
		{Role: "user", Content: prompt},
	}
	return chatCompletion(ctx, messages, settings, 0.2)
}

func chatCompletion(
	ctx context.Context,
	messages []chatMessage,
	settings common.Settings,
	temperature float64,
) (string, error) {
	if err := common.RequireOpenAI(settings); err != nil {
		return "", err
	}

	headers := map[string]string{
		"Authorization": fmt.Sprintf("Bearer %s", settings.OpenAIAPIKey),
	}
	payload := chatRequest{
		Model:       settings.OpenAIModel,
		Messages:    messages,
		Temperature: temperature,
	}

	status, body, err := common.PostJSON(ctx, chatCompletionsURL, headers, payload, settings.Timeout)
	if err != nil {
		return "", err
	}
	if status >= 300 {
		return "", providers.APIError{StatusCode: status, Body: string(body)}
	}

	var response chatResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if len(response.Choices) == 0 {
		return "", fmt.Errorf("unexpected response from OpenAI")
	}

	return response.Choices[0].Message.Content, nil
}
