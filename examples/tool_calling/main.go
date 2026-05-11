// Tool-calling example: registers a get_current_time tool and runs a
// hand-rolled loop until the model issues no more tool calls. This is the
// minimal tool-calling pattern — pi-agent-go wraps the same loop with
// hooks, steering, and a typed tool registry.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/tool_calling
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// A small, hand-written JSON Schema for the tool's input. pi-agent-go adds a
// Typed[I, O] helper that derives this from a Go struct.
var timeToolSchema = json.RawMessage(`{
    "type": "object",
    "properties": {
        "timezone": {
            "type": "string",
            "description": "IANA timezone name, e.g. 'America/New_York'. Defaults to UTC."
        }
    },
    "additionalProperties": false
}`)

// executeTool returns the text result for a single tool call.
func executeTool(call llm.ToolCallBlock) (result string, isError bool) {
	switch call.Name {
	case "get_current_time":
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
	default:
		return fmt.Sprintf("unknown tool %q", call.Name), true
	}
}

func main() {
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

	tools := []llm.Tool{
		{
			Name:        "get_current_time",
			Description: "Get the current wall-clock time in an IANA timezone.",
			InputSchema: timeToolSchema,
		},
	}

	messages := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Block{
			llm.TextBlock{Text: "What time is it right now in Tokyo and New York? Use the tool."},
		}},
	}

	ctx := context.Background()
	for iter := 1; iter <= 5; iter++ {
		fmt.Printf("\n--- iteration %d ---\n", iter)
		msg, err := llm.Complete(ctx, p, llm.Request{
			Model:     anthropic.ClaudeSonnet4_6,
			Tools:     tools,
			Messages:  messages,
			MaxTokens: 1024,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "complete error:", err)
			os.Exit(1)
		}

		// Print assistant content for visibility.
		for _, block := range msg.Content {
			switch b := block.(type) {
			case llm.TextBlock:
				fmt.Printf("assistant: %s\n", b.Text)
			case llm.ToolCallBlock:
				fmt.Printf("tool call:  %s(%s)\n", b.Name, string(b.Arguments))
			}
		}

		messages = append(messages, *msg)

		// Collect tool results, if any.
		var results []llm.Block
		for _, block := range msg.Content {
			if call, ok := block.(llm.ToolCallBlock); ok {
				out, isErr := executeTool(call)
				fmt.Printf("tool result: %s\n", out)
				results = append(results, llm.ToolResultBlock{
					ToolCallID: call.ID,
					Content:    out,
					IsError:    isErr,
				})
			}
		}

		if len(results) == 0 {
			fmt.Println("\n[done]")
			return
		}

		messages = append(messages, llm.Message{Role: llm.RoleTool, Content: results})
	}
	fmt.Fprintln(os.Stderr, "max iterations reached")
}
