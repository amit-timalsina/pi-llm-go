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
// ThinkingBlock). Use -tools to run a multi-turn tool loop, which exercises
// the stateless replay of assistant function_call items.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
	openai_responses "github.com/amit-timalsina/pi-llm-go/providers/openai_responses"
)

const azureDefaultURL = "https://anthropicgenesis.cognitiveservices.azure.com/openai/v1/responses?api-version=preview"

var timeToolSchema = json.RawMessage(`{
    "type": "object",
    "properties": {
        "timezone": {
            "type": "string",
            "description": "IANA timezone name, e.g. 'Asia/Tokyo'. Defaults to UTC."
        }
    },
    "additionalProperties": false
}`)

func currentTime(call llm.ToolCallBlock) (string, bool) {
	var args struct {
		Timezone string `json:"timezone"`
	}
	if err := json.Unmarshal(call.Arguments, &args); err != nil {
		return fmt.Sprintf("invalid arguments: %v", err), true
	}
	loc := time.UTC
	if args.Timezone != "" {
		parsed, err := time.LoadLocation(args.Timezone)
		if err != nil {
			return fmt.Sprintf("unknown timezone %q: %v", args.Timezone, err), true
		}
		loc = parsed
	}
	return time.Now().In(loc).Format(time.RFC3339), false
}

// toolLoop runs the hand-rolled tool loop. The second request replays the
// assistant's function_call alongside the function_call_output, which is what
// the Responses API needs when no previous_response_id is carried.
func toolLoop(ctx context.Context, p *openai_responses.Provider, model string) error {
	tools := []llm.Tool{{
		Name:        "get_current_time",
		Description: "Get the current wall-clock time in an IANA timezone.",
		InputSchema: timeToolSchema,
	}}
	messages := []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{
		llm.TextBlock{Text: "What time is it right now in Tokyo? Use the tool."},
	}}}

	for i := 1; i <= 5; i++ {
		fmt.Printf("\n--- iteration %d ---\n", i)
		msg, err := llm.Complete(ctx, p, llm.Request{
			Model:     model,
			Tools:     tools,
			Messages:  messages,
			MaxTokens: 1024,
		})
		if err != nil {
			return err
		}
		messages = append(messages, *msg)

		var results []llm.Block
		for _, block := range msg.Content {
			switch b := block.(type) {
			case llm.TextBlock:
				fmt.Printf("assistant: %s\n", b.Text)
			case llm.ToolCallBlock:
				fmt.Printf("tool call:  %s(%s) id=%s\n", b.Name, string(b.Arguments), b.ID)
				out, isErr := currentTime(b)
				fmt.Printf("tool result: %s\n", out)
				results = append(results, llm.ToolResultBlock{
					ToolCallID: b.ID,
					Content:    out,
					IsError:    isErr,
				})
			}
		}
		if len(results) == 0 {
			fmt.Println("\n[done]")
			return nil
		}
		messages = append(messages, llm.Message{Role: llm.RoleTool, Content: results})
	}
	return fmt.Errorf("max iterations reached")
}

func main() {
	useOpenAI := flag.Bool("openai", false, "use OpenAI directly (default: Azure)")
	withReasoning := flag.Bool("reasoning", false, "request reasoning summary streaming")
	withTools := flag.Bool("tools", false, "run a multi-turn tool loop instead of a single completion")
	model := flag.String("model", "gpt-5.4-mini", "model id")
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

	if *withTools {
		if err := toolLoop(context.Background(), p, *model); err != nil {
			fmt.Fprintln(os.Stderr, "tool loop error:", err)
			os.Exit(1)
		}
		return
	}

	req := llm.Request{
		Model: *model,
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
			fmt.Printf("\n\n[stop=%s in/out/total=%d/%d/%d reasoning=%d cached=%d]\n",
				e.StopReason, e.Usage.InputTokens, e.Usage.OutputTokens, e.Usage.TotalTokens,
				e.Usage.ReasoningTokens, e.Usage.CacheReadTokens)
		}
	}
}
