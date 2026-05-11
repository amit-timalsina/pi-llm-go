// openai_responses: stream from the OpenAI Responses API (/v1/responses).
//
// Shows the providers/openai_responses package against either:
//   - OpenAI directly (with OPENAI_API_KEY).
//   - Azure OpenAI / Azure AI Services (with AZURE_OPENAI_KEY and the
//     URL pointing at /openai/v1/responses).
//
// Picks Azure by default since most Responses-API users start there for
// access to GPT-5 family. Override via env:
//
//	# Azure (default)
//	export AZURE_OPENAI_KEY=...
//	go run ./examples/openai_responses
//
//	# OpenAI direct
//	export OPENAI_API_KEY=...
//	go run ./examples/openai_responses -openai
//
// Use -reasoning to request reasoning summary streaming (surfaces as
// ThinkingBlock).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	llm "github.com/amittimalsina/pi-llm-go"
	openai_responses "github.com/amittimalsina/pi-llm-go/providers/openai_responses"
)

const azureDefaultURL = "https://anthropicgenesis.cognitiveservices.azure.com/openai/v1/responses?api-version=preview"

func main() {
	useOpenAI := flag.Bool("openai", false, "use OpenAI directly (default: Azure)")
	withReasoning := flag.Bool("reasoning", false, "request reasoning summary streaming")
	prompt := flag.String("prompt", "If a train leaves Boston at 3pm going 60 mph and another leaves NYC at 4pm going 75 mph toward Boston (215 miles apart), when do they meet?", "user prompt")
	flag.Parse()

	var opts openai_responses.Options
	if *useOpenAI {
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			fmt.Fprintln(os.Stderr, "OPENAI_API_KEY is required when -openai is set")
			os.Exit(2)
		}
		opts = openai_responses.Options{APIKey: key}
	} else {
		key := os.Getenv("AZURE_OPENAI_KEY")
		if key == "" {
			fmt.Fprintln(os.Stderr, "AZURE_OPENAI_KEY is required (or pass -openai to use OpenAI directly)")
			os.Exit(2)
		}
		opts = openai_responses.Options{
			URL:     azureDefaultURL,
			Headers: map[string]string{"api-key": key},
		}
	}
	if *withReasoning {
		opts.ReasoningEffort = openai_responses.ReasoningMedium
		opts.IncludeReasoningSummary = true
	}

	p, err := openai_responses.New(opts)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	req := llm.Request{
		Model: "gpt-5.4-mini",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: *prompt}}},
		},
		MaxTokens: 1024,
	}

	var inThinking bool
	for ev, err := range p.Stream(context.Background(), req) {
		if err != nil {
			fmt.Fprintln(os.Stderr, "\nstream error:", err)
			os.Exit(1)
		}
		switch e := ev.(type) {
		case llm.EventThinkingStart:
			inThinking = true
			fmt.Print("\n\033[2m[reasoning]\033[0m ")
		case llm.EventThinkingDelta:
			if inThinking {
				fmt.Printf("\033[2m%s\033[0m", e.Delta)
			}
		case llm.EventThinkingEnd:
			inThinking = false
			fmt.Println()
		case llm.EventTextStart:
			fmt.Print("\n[answer] ")
		case llm.EventTextDelta:
			fmt.Print(e.Delta)
		case llm.EventMessageEnd:
			fmt.Printf("\n\n[stop=%s in/out/total=%d/%d/%d]\n",
				e.StopReason, e.Usage.InputTokens, e.Usage.OutputTokens, e.Usage.TotalTokens)
		}
	}
}
