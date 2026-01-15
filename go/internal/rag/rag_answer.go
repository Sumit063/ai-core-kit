package rag

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"ai-core-kit-go/internal/common"
	"ai-core-kit-go/internal/providers/openai"
)

var citationPattern = regexp.MustCompile(`\[((?:[^\[\]]+?):\d+)\]`)

// calls the model with context and extracts citations.
func AnswerWithCitations(
	ctx context.Context,
	query string,
	chunks []ScoredEntry,
	settings common.Settings,
) (string, []string, error) {
	prompt := buildPrompt(query, chunks)
	answer, err := openai.Generate(ctx, prompt, settings)
	if err != nil {
		return "", nil, err
	}

	citations := extractCitations(answer)
	return strings.TrimSpace(answer), citations, nil
}

func buildPrompt(query string, chunks []ScoredEntry) string {
	var contextLines []string
	for _, chunk := range chunks {
		tag := fmt.Sprintf("[%s:%d]", chunk.Source, chunk.ID)
		contextLines = append(contextLines, fmt.Sprintf("%s %s", tag, chunk.Text))
	}
	contextBlock := strings.Join(contextLines, "\n\n")

	return fmt.Sprintf(
		"Answer the question using only the context below. Cite sources in-line as [source:id]. "+
			"If the answer is not in the context, say you do not know.\n\nContext:\n%s\n\nQuestion: %s\nAnswer:",
		contextBlock,
		query,
	)
}

func extractCitations(answer string) []string {
	matches := citationPattern.FindAllStringSubmatch(answer, -1)
	unique := make(map[string]struct{}, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			unique[match[1]] = struct{}{}
		}
	}

	citations := make([]string, 0, len(unique))
	for key := range unique {
		citations = append(citations, key)
	}
	sort.Strings(citations)
	return citations
}
