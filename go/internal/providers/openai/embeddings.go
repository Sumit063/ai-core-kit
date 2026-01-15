package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"ai-core-kit-go/internal/common"
	"ai-core-kit-go/internal/providers"
)

const embeddingsURL = "https://api.openai.com/v1/embeddings"

type embeddingsRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embeddingsResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

// returns embeddings for a batch of texts.
func EmbedTexts(ctx context.Context, texts []string, settings common.Settings) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if err := common.RequireOpenAI(settings); err != nil {
		return nil, err
	}

	headers := map[string]string{
		"Authorization": fmt.Sprintf("Bearer %s", settings.OpenAIAPIKey),
	}
	payload := embeddingsRequest{
		Model: settings.OpenAIEmbedModel,
		Input: texts,
	}

	status, body, err := common.PostJSON(ctx, embeddingsURL, headers, payload, settings.Timeout)
	if err != nil {
		return nil, err
	}
	if status >= 300 {
		return nil, providers.APIError{StatusCode: status, Body: string(body)}
	}

	var response embeddingsResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	embeddings := make([][]float64, 0, len(response.Data))
	for _, item := range response.Data {
		embeddings = append(embeddings, item.Embedding)
	}
	return embeddings, nil
}
