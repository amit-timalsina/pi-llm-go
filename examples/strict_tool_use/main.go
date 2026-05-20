// strict_tool_use: cookbook for grammar-constrained tool-call sampling.
//
// Demonstrates two related features (closes issue #26):
//
//   - llm.Tool.Strict: opts the tool into grammar-constrained sampling.
//     The model's token sampler is constrained to schema-valid tokens,
//     so the emitted input is GUARANTEED to match the InputSchema (no
//     app-side validate-and-retry loop).
//
//   - llm.Request.ToolChoice: controls whether/which tool the model
//     must call this turn. Combined with Strict, the response is
//     guaranteed: tool=<this one> AND args strictly match the schema.
//
// Use case: enum-constrained inputs (e.g. a `name` field that must be
// one of a fixed canonical list). Without strict mode the model often
// hallucinates near-misses; with strict mode it cannot emit an out-of-
// schema token.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/strict_tool_use
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "ANTHROPIC_API_KEY is required")
		os.Exit(2)
	}
	p, err := anthropic.New(anthropic.Options{APIKey: apiKey})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// Schema with an enum constraint on `name`. For strict mode to
	// engage on Anthropic, the schema must:
	//   - have additionalProperties: false at every object level
	//   - list every property in `required`
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"name":  { "type": "string", "enum": [
				"SINTER_MC_SPEED_CCR1 (m/min)",
				"COOLER_PALLET_SPEED (mm/s)",
				"BLOWER_FAN_RPM (rpm)"
			]},
			"value": { "type": "number" }
		},
		"required": ["name", "value"],
		"additionalProperties": false
	}`)

	req := llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Tools: []llm.Tool{{
			Name:        "set_control",
			Description: "Set a process control to a target value.",
			InputSchema: json.RawMessage(schemaBytes),
			Strict:      true, // grammar-constrained — see godoc on llm.Tool.Strict
		}},
		// Force the model to call exactly THIS tool (it can't dodge the
		// constraint by answering in prose). Anthropic wire shape:
		// `tool_choice: {"type":"tool","name":"set_control"}`.
		// Note: Anthropic forbids ToolChoiceTool + ThinkingConfig on
		// the same request — only Auto/None are compatible with thinking.
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceTool, Name: "set_control"},
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			Content: []llm.Block{llm.TextBlock{
				Text: "Set blower fan RPM to 1850. Use the set_control tool.",
			}},
		}},
	}

	msg, err := llm.Complete(ctx(), p, req)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	for _, block := range msg.Content {
		if tc, ok := block.(llm.ToolCallBlock); ok {
			fmt.Printf("tool=%s\nargs=%s\n", tc.Name, string(tc.Arguments))
			var parsed struct {
				Name  string  `json:"name"`
				Value float64 `json:"value"`
			}
			if err := json.Unmarshal(tc.Arguments, &parsed); err != nil {
				fmt.Println("UNEXPECTED: args failed to parse:", err)
				os.Exit(1)
			}
			fmt.Printf("\nparsed: name=%q value=%v\n", parsed.Name, parsed.Value)
			fmt.Println("strict mode guaranteed the name is one of the enum values above.")
		}
	}
}

func ctx() context.Context { return context.Background() }
