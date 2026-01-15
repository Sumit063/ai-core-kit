package rag

import "strings"

// splits text into overlapping chunks for embedding.
func ChunkText(text string, chunkSize int, overlap int) []string {
	normalized := strings.Join(strings.Fields(text), " ")
	if normalized == "" {
		return nil
	}

	runes := []rune(normalized)
	length := len(runes)
	if chunkSize <= 0 {
		chunkSize = 800
	}
	if overlap < 0 {
		overlap = 0
	}

	var chunks []string
	start := 0
	for start < length {
		end := start + chunkSize
		if end > length {
			end = length
		}
		chunks = append(chunks, string(runes[start:end]))
		if end >= length {
			break
		}
		start = end - overlap
		if start < 0 {
			start = 0
		}
	}

	return chunks
}
