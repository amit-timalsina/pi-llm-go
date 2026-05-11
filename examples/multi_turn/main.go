// multi_turn: build up a conversation across several Complete() calls.
//
// pi-llm-go is stateless — it doesn't manage transcripts for you. To carry
// context across turns, append each assistant response and the next user
// prompt to a llm.Message slice and send the growing slice in each
// Request. This is exactly what pi-agent-go does internally, but here we
// show the pattern by hand for users who want bare control.
//
// Three chained turns: each question references the prior answer.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/multi_turn
package main

import (
	"context"
	"fmt"
	"os"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

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

	transcript := []llm.Message{}
	system := "You are a precise math assistant. Reply with just the numeric answer plus a one-line explanation. No formatting."

	turns := []string{
		"What is 17 * 23?",
		"Now multiply that by 2.",
		"What's the prime factorization of that final number?",
	}

	ctx := context.Background()
	for i, prompt := range turns {
		fmt.Printf("\n--- turn %d ---\n", i+1)
		fmt.Printf("user: %s\n", prompt)

		// Append user message to the transcript.
		transcript = append(transcript, llm.Message{
			Role:    llm.RoleUser,
			Content: []llm.Block{llm.TextBlock{Text: prompt}},
		})

		msg, err := llm.Complete(ctx, p, llm.Request{
			Model:     anthropic.ClaudeHaiku4_5,
			System:    system,
			Messages:  transcript,
			MaxTokens: 256,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "complete error:", err)
			os.Exit(1)
		}

		// Print the assistant's text content.
		fmt.Print("assistant: ")
		for _, block := range msg.Content {
			if tb, ok := block.(llm.TextBlock); ok {
				fmt.Print(tb.Text)
			}
		}
		fmt.Printf("\n[stop=%s usage in/out=%d/%d]\n",
			msg.StopReason, msg.Usage.InputTokens, msg.Usage.OutputTokens)

		// Append the full assistant message so future turns see this answer
		// as context.
		transcript = append(transcript, *msg)
	}

	fmt.Printf("\nFinal transcript: %d messages (each Complete() saw the growing history)\n", len(transcript))
}
