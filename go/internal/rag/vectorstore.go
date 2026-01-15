package rag

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// represents a stored chunk and its embedding.
type Entry struct {
	ID        int       `json:"id"`
	Source    string    `json:"source"`
	Text      string    `json:"text"`
	Embedding []float64 `json:"embedding"`
}

// an Entry with a similarity score.
type ScoredEntry struct {
	ID     int     `json:"id"`
	Source string  `json:"source"`
	Text   string  `json:"text"`
	Score  float64 `json:"score"`
}

// reads a JSON vector store from disk.
func LoadStore(path string) ([]Entry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read store: %w", err)
	}

	var entries []Entry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("decode store: %w", err)
	}
	return entries, nil
}

// writes entries to a JSON vector store on disk.
func SaveStore(path string, entries []Entry) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create store directory: %w", err)
	}
	payload, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return fmt.Errorf("encode store: %w", err)
	}
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		return fmt.Errorf("write store: %w", err)
	}
	return nil
}

// returns the top-k most similar entries.
func Search(entries []Entry, queryEmbedding []float64, topK int) []ScoredEntry {
	if topK <= 0 {
		topK = 5
	}

	scored := make([]ScoredEntry, 0, len(entries))
	for _, entry := range entries {
		score := cosineSimilarity(queryEmbedding, entry.Embedding)
		scored = append(scored, ScoredEntry{
			ID:     entry.ID,
			Source: entry.Source,
			Text:   entry.Text,
			Score:  score,
		})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	if topK > len(scored) {
		topK = len(scored)
	}
	return scored[:topK]
}

func cosineSimilarity(a []float64, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
