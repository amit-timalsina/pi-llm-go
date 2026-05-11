// Streaming example: prints assistant text to stdout as it streams.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/streaming
//
// Pass --openai to hit an OpenAI-compatible provider instead (requires
// OPENAI_API_KEY).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
	"github.com/amit-timalsina/pi-llm-go/providers/openai"
)

func main() {
	useOpenAI := flag.Bool("openai", false, "use the OpenAI-compatible provider instead of Anthropic")
	prompt := flag.String("prompt", "Explain Go iterators in two sentences.", "user prompt")
	flag.Parse()

	var provider llm.LLM
	var model string
	if *useOpenAI {
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			fmt.Fprintln(os.Stderr, "OPENAI_API_KEY is required")
			os.Exit(2)
		}
		p, err := openai.New(openai.Options{APIKey: key})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		provider = p
		model = openai.GPT5_5
	} else {
		key := os.Getenv("ANTHROPIC_API_KEY")
		if key == "" {
			fmt.Fprintln(os.Stderr, "ANTHROPIC_API_KEY is required")
			os.Exit(2)
		}
		p, err := anthropic.New(anthropic.Options{APIKey: key})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		provider = p
		model = anthropic.ClaudeSonnet4_6
	}

	req := llm.Request{
		Model:     model,
		MaxTokens: 1024,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: *prompt}}},
		},
	}

	for event, err := range provider.Stream(context.Background(), req) {
		if err != nil {
			fmt.Fprintln(os.Stderr, "\nstream error:", err)
			os.Exit(1)
		}
		switch e := event.(type) {
		case llm.EventTextDelta:
			fmt.Print(e.Delta)
		case llm.EventMessageEnd:
			fmt.Printf("\n\n[stop=%s tokens in/out=%d/%d]\n",
				e.StopReason, e.Usage.InputTokens, e.Usage.OutputTokens)
		}
	}
}
