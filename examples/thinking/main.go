// thinking: demonstrates Anthropic's extended thinking via pi-llm-go.
//
// Extended thinking exposes the model's reasoning as a separate content
// block stream before the final answer. The model deliberates "out loud"
// inside ThinkingBlocks; the user only sees the final TextBlock unless
// they choose to render the thinking too.
//
// Two things this example shows:
//  1. Enable thinking by setting llm.Request.Thinking on the request.
//  2. Distinguish ThinkingBlock streaming from TextBlock streaming so a
//     UI can render the two differently (dim/collapsed thinking, bold
//     final answer).
//
// Usage:
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/thinking
//	go run ./examples/thinking -prompt "..."  -- custom prompt
//	go run ./examples/thinking -hide-thinking -- suppress thinking output
//	go run ./examples/thinking -effort high   -- adaptive thinking (Opus 4.7+)
//	go run ./examples/thinking -effort high -display summarized -model claude-opus-4-8
//	  -- adaptive + summarized thinking text (without -display, Opus 4.7+
//	     streams signature-only thinking blocks with no text)
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
	budget := flag.Int("budget", 4096, "manual thinking token budget (minimum 1024); ignored when -effort is set")
	effort := flag.String("effort", "", "adaptive thinking effort (low|medium|high); Opus 4.7+ — overrides -budget")
	display := flag.String("display", "", "adaptive thinking display (summarized|omitted); needs -effort")
	model := flag.String("model", string(anthropic.ClaudeSonnet4_6), "model id")
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

	// Two thinking shapes: adaptive (Opus 4.7+ — set -effort; the model
	// decides how much to think within the effort bucket) and manual
	// (older models — a fixed -budget). On adaptive, -display summarized
	// opts back into streamed thinking TEXT; without it Opus 4.7+ streams
	// signature-only blocks and EventThinkingDelta carries nothing.
	//
	// Anthropic requires max_tokens > thinking.budget_tokens in manual
	// mode (the budget is *included* in max_tokens, plus room for the
	// final answer). Allow ~2x the budget for the visible answer.
	thinking := &llm.ThinkingConfig{}
	maxTokens := *budget * 2
	if *effort != "" {
		thinking.Effort = llm.Effort(*effort)
		thinking.Display = llm.ThinkingDisplay(*display)
		maxTokens = 8192
	} else {
		thinking.BudgetTokens = *budget
	}
	req := llm.Request{
		Model:     *model,
		MaxTokens: maxTokens,
		Thinking:  thinking,
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
