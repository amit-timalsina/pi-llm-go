package llm_test

// These Example functions render on pkg.go.dev next to the documented
// types. They're compiled by `go test` but never executed (no
// // Output: lines), so they don't need API keys to verify — they're
// here so coding agents and humans land on a runnable, copy-pasteable
// snippet for every public-API entry point.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// Example demonstrates the smallest useful pi-llm-go program: a one-shot
// Anthropic Claude completion. Replace anthropic.New / ClaudeSonnet4_6
// with any other provider to switch backends.
func Example() {
	p, err := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
	if err != nil {
		panic(err)
	}

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "Reply with one word: ready"}}},
		},
	})
	if err != nil {
		panic(err)
	}

	for _, block := range msg.Content {
		if tb, ok := block.(llm.TextBlock); ok {
			fmt.Println(tb.Text)
		}
	}
}

// ExampleComplete shows a one-shot synchronous completion. Use this
// when you don't care about per-token streaming and just need the
// final assistant Message.
func ExampleComplete() {
	p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hello"}}},
		},
	})
	if err != nil {
		return
	}
	fmt.Println(msg.Usage.TotalTokens, "tokens")
}

// ExampleLLM_Stream shows iterator-based streaming. Range over the
// returned iter.Seq2 and type-switch on the event for the granularity
// you need — token-level text via EventTextDelta, tool-call assembly
// via EventToolCall*, final usage via EventMessageEnd.
func ExampleLLM_Stream() {
	p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})

	req := llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "Count to three."}}},
		},
	}
	for event, err := range p.Stream(context.Background(), req) {
		if err != nil {
			return
		}
		switch e := event.(type) {
		case llm.EventTextDelta:
			fmt.Print(e.Delta)
		case llm.EventMessageEnd:
			fmt.Printf("\n[in=%d out=%d]\n", e.Usage.InputTokens, e.Usage.OutputTokens)
		}
	}
}

// ExampleRequest_toolCalling shows declaring a tool, receiving the
// model's ToolCallBlock, and shipping a ToolResultBlock back on the
// next turn. For a built-in loop that does this for you, see
// https://github.com/amit-timalsina/pi-agent-go.
func ExampleRequest_toolCalling() {
	p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})

	schema := json.RawMessage(`{
		"type":"object",
		"properties":{"city":{"type":"string"}},
		"required":["city"]
	}`)
	req := llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Tools:     []llm.Tool{{Name: "get_weather", Description: "Get weather for a city.", InputSchema: schema}},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "What's the weather in Tokyo?"}}},
		},
	}

	msg, err := llm.Complete(context.Background(), p, req)
	if err != nil {
		return
	}
	for _, block := range msg.Content {
		if tc, ok := block.(llm.ToolCallBlock); ok {
			fmt.Printf("model wants to call %s with %s\n", tc.Name, tc.Arguments)
			// Execute the tool, then send ToolResultBlock on next turn:
			req.Messages = append(req.Messages, *msg, llm.Message{
				Role: llm.RoleUser,
				Content: []llm.Block{llm.ToolResultBlock{
					ToolCallID: tc.ID,
					Content:    `{"temp": 22, "unit": "C"}`,
				}},
			})
		}
	}
}

// ExampleThinkingConfig shows the two Anthropic extended-thinking
// shapes pi-llm-go supports. Pick the shape that matches the model:
//
//   - Opus 4.7 / Opus 4.6 / Sonnet 4.6: set Effort.
//   - Opus 4.5 / Sonnet 4.5 / Sonnet 3.7: set BudgetTokens.
//
// When both fields are set, Effort wins (adaptive shape emitted) —
// useful when a single config object flows through multiple models
// during a migration.
func ExampleThinkingConfig() {
	p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})

	// Adaptive thinking (Opus 4.7+): model decides depth within Effort.
	adaptiveReq := llm.Request{
		Model:     anthropic.ClaudeOpus4_7,
		MaxTokens: 4096,
		Thinking:  &llm.ThinkingConfig{Effort: llm.EffortMedium},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "Reason carefully about X."}}},
		},
	}
	_, _ = llm.Complete(context.Background(), p, adaptiveReq)

	// Manual thinking (Opus 4.5 and older): caller pins the token cap.
	manualReq := llm.Request{
		Model:     "claude-opus-4-5",
		MaxTokens: 4096,
		Thinking:  &llm.ThinkingConfig{BudgetTokens: 2048},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "Reason carefully about X."}}},
		},
	}
	_, _ = llm.Complete(context.Background(), p, manualReq)
}

// ExampleCacheRetention shows enabling Anthropic prompt caching at the
// 1-hour TTL tier. Cache breakpoints are auto-placed at the end of
// System / last Tool / last user message block. Cache hits surface via
// Usage.CacheReadTokens; the per-TTL breakdown (CacheWrite5mTokens /
// CacheWrite1hTokens) lets callers detect silent 5min fallback when a
// model doesn't honor the extended TTL.
func ExampleCacheRetention() {
	p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})

	req := llm.Request{
		Model:          anthropic.ClaudeSonnet4_6,
		MaxTokens:      1024,
		System:         "Long static system prompt that's identical across iterations.",
		CacheRetention: llm.CacheRetentionLong, // 1h TTL; CacheRetentionShort for 5m
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hello"}}},
		},
	}
	msg, _ := llm.Complete(context.Background(), p, req)

	if req.CacheRetention == llm.CacheRetentionLong &&
		msg.Usage.CacheWrite5mTokens > 0 && msg.Usage.CacheWrite1hTokens == 0 {
		fmt.Println("WARNING: model fell back to 5min cache — 1h tier not honored")
	}
}

// ExampleComputeCost shows projecting dollar cost from a Usage record.
// Pricing for Claude 4 / GPT-5 / Gemini 2.5+3.1 families ships in the
// seed table; register your own via llm.RegisterPricing for other
// models or for batch / regional pricing tiers.
func ExampleComputeCost() {
	usage := llm.Usage{
		InputTokens:        50_000,
		OutputTokens:       2_000,
		CacheReadTokens:    100_000,
		CacheWrite1hTokens: 30_000,
	}
	cost, err := llm.ComputeCost(usage, anthropic.ClaudeSonnet4_6)
	if err != nil {
		return
	}
	fmt.Printf("$%.4f total (in=$%.4f out=$%.4f cache_read=$%.4f cache_1h=$%.4f)\n",
		cost.Total(), cost.Input, cost.Output, cost.CacheRead, cost.CacheWrite1h)
}

// ExampleTokenCounter shows pre-flight input-token counting against a
// provider's dedicated count endpoint. Anthropic and Gemini implement
// llm.TokenCounter; OpenAI does not (no server-side count endpoint —
// run tiktoken yourself if you need a count for an OpenAI-hosted
// model).
func ExampleTokenCounter() {
	p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})

	if counter, ok := llm.LLM(p).(llm.TokenCounter); ok {
		n, err := counter.CountTokens(context.Background(), llm.Request{
			Model: anthropic.ClaudeSonnet4_6,
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hello"}}},
			},
		})
		if err == nil {
			fmt.Printf("would consume %d input tokens\n", n)
		}
	}
}

// ExampleRetryPolicy wires retry middleware into a provider. Retries
// retriable errors (429 / 529 / 5xx / transient network) with
// exponential backoff + full jitter; honors server-supplied
// Retry-After hints up to MaxDelay. Only the initial HTTP attempt is
// retried — mid-stream connection breaks are NOT replayed.
func ExampleRetryPolicy() {
	p, _ := anthropic.New(anthropic.Options{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
		Retry: &llm.RetryPolicy{
			MaxAttempts: 4,
			BaseDelay:   time.Second,
			MaxDelay:    30 * time.Second,
		},
	})
	_ = p

	// Or use the sane defaults:
	defaults := llm.DefaultRetryPolicy()
	p2, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY"), Retry: &defaults})
	_ = p2
}

// ExampleAPIError shows error-classification with errors.Is for typed
// retry / escalation policies. Sugar helpers (IsRateLimited,
// IsOverloaded, IsServerError, IsContextLength, IsPolicyViolation,
// IsRetriable) avoid string-matching on err.Error().
func ExampleAPIError() {
	p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
	req := llm.Request{Model: anthropic.ClaudeSonnet4_6, MaxTokens: 1024}

	for ev, err := range p.Stream(context.Background(), req) {
		_ = ev
		if err == nil {
			continue
		}

		var apiErr *llm.APIError
		if errors.As(err, &apiErr) {
			switch {
			case llm.IsRateLimited(err):
				fmt.Printf("rate-limited; retry after %s\n", apiErr.RetryAfter)
			case llm.IsOverloaded(err):
				fmt.Println("overloaded; consider provider fallback")
			case llm.IsServerError(err):
				fmt.Println("5xx; retry with backoff")
			case llm.IsContextLength(err):
				fmt.Println("prompt too long; truncate or route to longer-context model")
			case llm.IsPolicyViolation(err):
				fmt.Println("rejected by safety policy; do not retry")
			case errors.Is(err, llm.ErrAuth):
				fmt.Println("auth failed; check API key")
			}
		}
		return
	}
}

// ExampleRunWithRetry shows the exported retry primitive used by every
// provider internally. Useful for building cross-provider fallback or
// circuit-breaker logic on top.
func ExampleRunWithRetry() {
	policy := llm.RetryPolicy{MaxAttempts: 3, BaseDelay: time.Second, MaxDelay: 10 * time.Second}

	result, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		// Your operation here. Return a retriable error
		// (errors.Is(err, llm.ErrRateLimit) etc.) to trigger retry;
		// return any other error to abort immediately.
		return "ok", nil
	})
	_ = result
	_ = err
}
