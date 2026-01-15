package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"ai-core-kit-go/internal/common"
)

const structuredSystemPrompt = "Return ONLY valid JSON with keys: title, summary, keywords. Do not wrap the JSON in markdown or add extra text."

// defines the expected JSON schema for structured output.
type StructuredOutput struct {
	Title    string   `json:"title"`
	Summary  string   `json:"summary"`
	Keywords []string `json:"keywords"`
}

// requests a schema-validated JSON response.
func StructuredJSON(ctx context.Context, prompt string, settings common.Settings) (StructuredOutput, error) {
	baseMessages := []chatMessage{
		{Role: "system", Content: structuredSystemPrompt},
		{Role: "user", Content: prompt},
	}

	messages := baseMessages
	var lastErr error
	for attempt := 0; attempt <= 2; attempt++ {
		content, err := chatCompletion(ctx, messages, settings, 0)
		if err != nil {
			return StructuredOutput{}, err
		}

		cleaned := stripJSONFence(content)
		output, err := decodeStructured(cleaned)
		if err == nil {
			return output, nil
		}

		lastErr = err
		if attempt >= 2 {
			break
		}

		correction := fmt.Sprintf(
			"The previous response was invalid JSON or did not match the schema. Error: %v. Return ONLY valid JSON with keys: title, summary, keywords.",
			err,
		)
		messages = append(baseMessages,
			chatMessage{Role: "assistant", Content: content},
			chatMessage{Role: "user", Content: correction},
		)
	}

	return StructuredOutput{}, fmt.Errorf("structured output validation failed: %w", lastErr)
}

func stripJSONFence(text string) string {
	cleaned := strings.TrimSpace(text)
	if strings.HasPrefix(cleaned, "```") {
		cleaned = strings.Trim(cleaned, "`")
		cleaned = strings.TrimSpace(cleaned)
		if strings.HasPrefix(strings.ToLower(cleaned), "json") {
			cleaned = strings.TrimSpace(cleaned[4:])
		}
	}
	return cleaned
}

func decodeStructured(raw string) (StructuredOutput, error) {
	var output StructuredOutput
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&output); err != nil {
		return StructuredOutput{}, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return StructuredOutput{}, fmt.Errorf("unexpected trailing content")
	}
	if strings.TrimSpace(output.Title) == "" || strings.TrimSpace(output.Summary) == "" {
		return StructuredOutput{}, fmt.Errorf("missing required fields")
	}
	return output, nil
}
