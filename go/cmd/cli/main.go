package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"ai-core-kit-go/internal/common"
	"ai-core-kit-go/internal/providers/openai"
	"ai-core-kit-go/internal/rag"
)

const (
	defaultDocsDir = "./sample_docs"
	defaultStore   = "./data/vectorstore.json"
	defaultTopK    = 5
)

type chunkRecord struct {
	Source string
	Text   string
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	switch cmd {
	case "generate":
		runGenerate(os.Args[2:])
	case "structured":
		runStructured(os.Args[2:])
	case "index":
		runIndex(os.Args[2:])
	case "search":
		runSearch(os.Args[2:])
	case "rag":
		runRag(os.Args[2:])
	case "-h", "--help", "help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func runGenerate(args []string) {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	prompt := fs.String("prompt", "", "Prompt text")
	fs.Parse(args)

	if strings.TrimSpace(*prompt) == "" {
		exitWithError("generate: --prompt is required")
	}

	settings := loadSettings()
	ctx := context.Background()
	output, err := openai.Generate(ctx, *prompt, settings)
	if err != nil {
		exitWithError(err.Error())
	}
	fmt.Println(strings.TrimSpace(output))
}

func runStructured(args []string) {
	fs := flag.NewFlagSet("structured", flag.ExitOnError)
	prompt := fs.String("prompt", "", "Prompt text")
	fs.Parse(args)

	if strings.TrimSpace(*prompt) == "" {
		exitWithError("structured: --prompt is required")
	}

	settings := loadSettings()
	ctx := context.Background()
	output, err := openai.StructuredJSON(ctx, *prompt, settings)
	if err != nil {
		exitWithError(err.Error())
	}

	payload, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		exitWithError(err.Error())
	}
	fmt.Println(string(payload))
}

func runIndex(args []string) {
	fs := flag.NewFlagSet("index", flag.ExitOnError)
	docsDir := fs.String("docs_dir", defaultDocsDir, "Docs directory")
	store := fs.String("store", defaultStore, "Vector store path")
	fs.Parse(args)

	settings := loadSettings()
	ctx := context.Background()

	records, err := collectChunks(*docsDir)
	if err != nil {
		exitWithError(err.Error())
	}
	if len(records) == 0 {
		exitWithError("no text files found to index")
	}

	const batchSize = 32
	var entries []rag.Entry
	nextID := 0
	for i := 0; i < len(records); i += batchSize {
		end := i + batchSize
		if end > len(records) {
			end = len(records)
		}

		batch := records[i:end]
		texts := make([]string, 0, len(batch))
		for _, record := range batch {
			texts = append(texts, record.Text)
		}

		embeddings, err := openai.EmbedTexts(ctx, texts, settings)
		if err != nil {
			exitWithError(err.Error())
		}

		for idx, record := range batch {
			entries = append(entries, rag.Entry{
				ID:        nextID,
				Source:    record.Source,
				Text:      record.Text,
				Embedding: embeddings[idx],
			})
			nextID++
		}
	}

	if err := rag.SaveStore(*store, entries); err != nil {
		exitWithError(err.Error())
	}
	fmt.Printf("Indexed %d chunks -> %s\n", len(entries), *store)
}

func runSearch(args []string) {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	query := fs.String("query", "", "Search query")
	store := fs.String("store", defaultStore, "Vector store path")
	topK := fs.Int("top_k", defaultTopK, "Number of results")
	fs.Parse(args)

	if strings.TrimSpace(*query) == "" {
		exitWithError("search: --query is required")
	}

	settings := loadSettings()
	ctx := context.Background()

	entries, err := rag.LoadStore(*store)
	if err != nil {
		exitWithError(err.Error())
	}

	embeddings, err := openai.EmbedTexts(ctx, []string{*query}, settings)
	if err != nil {
		exitWithError(err.Error())
	}
	if len(embeddings) == 0 {
		exitWithError("failed to embed query")
	}

	results := rag.Search(entries, embeddings[0], *topK)
	payload, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		exitWithError(err.Error())
	}
	fmt.Println(string(payload))
}

func runRag(args []string) {
	fs := flag.NewFlagSet("rag", flag.ExitOnError)
	query := fs.String("query", "", "Question to answer")
	store := fs.String("store", defaultStore, "Vector store path")
	topK := fs.Int("top_k", defaultTopK, "Number of chunks")
	fs.Parse(args)

	if strings.TrimSpace(*query) == "" {
		exitWithError("rag: --query is required")
	}

	settings := loadSettings()
	ctx := context.Background()

	entries, err := rag.LoadStore(*store)
	if err != nil {
		exitWithError(err.Error())
	}

	embeddings, err := openai.EmbedTexts(ctx, []string{*query}, settings)
	if err != nil {
		exitWithError(err.Error())
	}
	if len(embeddings) == 0 {
		exitWithError("failed to embed query")
	}

	results := rag.Search(entries, embeddings[0], *topK)
	answer, citations, err := rag.AnswerWithCitations(ctx, *query, results, settings)
	if err != nil {
		exitWithError(err.Error())
	}

	fmt.Println(answer)
	if len(citations) > 0 {
		payload, err := json.MarshalIndent(citations, "", "  ")
		if err != nil {
			exitWithError(err.Error())
		}
		fmt.Println("\nCitations:")
		fmt.Println(string(payload))
	}
}

func collectChunks(docsDir string) ([]chunkRecord, error) {
	var records []chunkRecord
	err := filepath.WalkDir(docsDir, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".txt" && ext != ".md" && ext != ".markdown" {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", path, err)
		}

		for _, chunk := range rag.ChunkText(string(content), 800, 150) {
			records = append(records, chunkRecord{
				Source: filepath.Base(path),
				Text:   chunk,
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return records, nil
}

func loadSettings() common.Settings {
	settings, err := common.LoadSettings()
	if err != nil {
		exitWithError(err.Error())
	}
	if err := common.RequireOpenAI(settings); err != nil {
		exitWithError(err.Error())
	}
	return settings
}

func printUsage() {
	fmt.Println("Usage: cli <command> [options]")
	fmt.Println("Commands: generate, structured, index, search, rag")
	fmt.Println("Use \"cli <command> -h\" for command-specific help.")
}

func exitWithError(message string) {
	fmt.Fprintf(os.Stderr, "Error: %s\n", message)
	os.Exit(1)
}
