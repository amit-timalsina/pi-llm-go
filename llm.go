// Package llm is a minimal, provider-agnostic Go interface for streaming
// LLM completions with tool calling. Built-in providers cover the Anthropic
// Messages API and OpenAI-compatible Chat Completions APIs (OpenAI, Groq,
// Together, vLLM, OpenRouter, Ollama, and similar).
//
// Two patterns of use:
//
//   - Streaming events: range over LLM.Stream to react to deltas in real time.
//   - One-shot completion: call Complete to drain the stream and receive the
//     final assistant message.
//
// Cancellation propagates through context.Context. Provider errors surface
// through the iterator's error return as *APIError wrapping a sentinel
// (ErrAuth, ErrRateLimit, ErrInvalidRequest, ErrProvider).
//
// pi-llm-go does not execute tools. It declares tool schemas on requests and
// surfaces ToolCallBlocks on responses. The companion pi-agent-go module
// adds the execution loop.
package llm

import (
	"context"
	"errors"
	"iter"
)

// LLM is the provider-agnostic streaming interface. Anthropic and OpenAI
// providers implement this; third-party providers may do the same to plug
// into Complete and Accumulate.
//
// Implementations must:
//   - Honor cancellation of ctx by terminating the underlying HTTP request
//     and yielding (nil, ctx.Err()) from the iterator.
//   - Surface HTTP errors as *APIError values via the iterator's error half.
//   - Emit events in the order documented on StreamEvent.
type LLM interface {
	Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error]
}

// Request is the common payload for a completion. Provider-specific
// tunables that have no portable meaning live on the provider's own Options
// struct (passed to its constructor), not here.
//
// Temperature is a pointer so the zero value can be distinguished from
// "unset"; callers that want temperature=0 must set *Temperature to 0.
type Request struct {
	Model       string
	System      string
	Messages    []Message
	Tools       []Tool
	Temperature *float64
	MaxTokens   int
	Thinking    *ThinkingConfig
	StopReasons []string

	// CacheRetention controls Anthropic prompt-cache breakpoint placement.
	// When unset or "none", no cache markers are emitted. When "short" or
	// "long", the Anthropic provider auto-places ephemeral cache_control
	// markers at the static prefix boundary: the last block of the System
	// prompt, the final Tool in Tools, and the last text block of the most
	// recent user message. "long" additionally selects the 1h TTL and
	// auto-attaches the extended-cache-ttl-2025-04-11 beta header.
	//
	// Ignored by OpenAI providers (their cache is automatic and opaque).
	CacheRetention CacheRetention
}

// ThinkingConfig enables extended thinking on supported models. Honored by
// the Anthropic provider. Ignored by the OpenAI-compatible provider in v1.
type ThinkingConfig struct {
	// BudgetTokens is the maximum number of thinking tokens the model may
	// emit before producing the final response. Required when ThinkingConfig
	// is non-nil. Provider minimums apply (Anthropic: 1024).
	//
	// IMPORTANT: Anthropic requires Request.MaxTokens > BudgetTokens because
	// thinking tokens are counted against max_tokens. A common safe choice
	// is MaxTokens == BudgetTokens * 2, giving roughly equal budget to the
	// reasoning trace and the visible answer.
	BudgetTokens int
}

// Complete drains a streaming completion and returns the final assistant
// message. It is equivalent to iterating Stream and folding each event
// into a Message via Accumulate.
//
// Returns the partial message and a wrapped error if the stream terminates
// early; the partial may be useful for debugging or replay.
func Complete(ctx context.Context, l LLM, req Request) (*Message, error) {
	var final *Message
	for msg, err := range Accumulate(l.Stream(ctx, req)) {
		if err != nil {
			return final, err
		}
		final = msg
	}
	if final == nil {
		return nil, errors.New("llm: stream produced no message")
	}
	return final, nil
}
