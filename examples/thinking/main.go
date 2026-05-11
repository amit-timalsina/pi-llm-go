// thinking: demonstrates Anthropic's extended thinking via pi-llm-go.
//
// Extended thinking exposes the model's reasoning as a separate content
// block stream before the final answer. The model deliberates "out loud"
// inside ThinkingBlocks; the user only sees the final TextBlock unless
// they choose to render the thinking too.
//
// Two things this example shows:
//   1. Enable thinking by setting llm.Request.Thinking on the request.
//   2. Distinguish ThinkingBlock streaming from TextBlock streaming so a
//      UI can render the two differently (dim/collapsed thinking, bold
//      final answer).
//
// Usage:
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/thinking
//	go run ./examples/thinking -prompt "..."  -- custom prompt
//	go run ./examples/thinking -hide-thinking -- suppress thinking output
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

func main() {
	prompt := flag.String("prompt", "If I have 7 apples and I give away 3, then buy 5 more, then eat 2, how many do I have? Think step by step.", "user prompt")
	hide := flag.Bool("hide-thinking", false, "suppress thinking output and show only the final answer")
	budget := flag.Int("budget", 4096, "thinking token budget (minimum 1024)")
	flag.Parse()

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

	// Extended thinking only works on reasoning-capable models. Sonnet 4.6
	// and Haiku 4.5 both support it; Opus 4.7 uses adaptive thinking and
	// ignores the explicit budget (no harm in setting it).
	// Anthropic requires max_tokens > thinking.budget_tokens (the thinking
	// budget is *included* in max_tokens, plus room for the final answer).
	// Allow ~2x the budget for the visible answer.
	req := llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: *budget * 2,
		Thinking:  &llm.ThinkingConfig{BudgetTokens: *budget},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: *prompt}}},
		},
	}

	// State tracking so we can label which block we're streaming.
	var currentBlock string // "thinking" | "text"

	for event, err := range p.Stream(context.Background(), req) {
		if err != nil {
			fmt.Fprintln(os.Stderr, "\nstream error:", err)
			os.Exit(1)
		}
		switch e := event.(type) {
		case llm.EventThinkingStart:
			currentBlock = "thinking"
			if !*hide {
				fmt.Print("\n\033[2m[thinking]\033[0m ") // dim ANSI
			}
		case llm.EventThinkingDelta:
			if !*hide && currentBlock == "thinking" {
				fmt.Printf("\033[2m%s\033[0m", e.Delta) // dim
			}
		case llm.EventThinkingEnd:
			if !*hide {
				fmt.Println()
			}
		case llm.EventTextStart:
			currentBlock = "text"
			fmt.Print("\n[answer] ")
		case llm.EventTextDelta:
			if currentBlock == "text" {
				fmt.Print(e.Delta)
			}
		case llm.EventMessageEnd:
			fmt.Printf("\n\n[stop=%s in/out/total=%d/%d/%d]\n",
				e.StopReason, e.Usage.InputTokens, e.Usage.OutputTokens, e.Usage.TotalTokens)
		}
	}
}
