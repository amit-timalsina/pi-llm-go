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

// ThinkingConfig enables extended thinking on supported models.
//
// Per-provider behavior of the two fields:
//
//   - **Anthropic**: dispatches on the field that's set. Effort emits
//     the adaptive shape (Opus 4.6+, required on 4.7+). BudgetTokens
//     emits the manual shape (Opus 4.5- / Sonnet 3.7, deprecated on
//     4.6 family).
//   - **Gemini**: honors BudgetTokens only (mapped to thinkingBudget;
//     -1=dynamic, 0=disabled). Effort is currently ignored on Gemini
//     because Gemini's wire shape is an integer budget rather than an
//     enum; revisit if Gemini ships an effort-style knob.
//   - **OpenAI Chat Completions**: ignores the entire field;
//     reasoning-effort dialects vary across compatible hosts and
//     don't map portably.
//   - **OpenAI Responses**: ignores this field; the provider routes
//     reasoning-effort via its own Options.ReasoningEffort enum.
//
// Anthropic has two on-wire shapes for extended thinking, gated by
// model:
//
//   - **Adaptive** (Opus 4.7+, recommended on Opus/Sonnet 4.6+) —
//     `thinking.type = "adaptive"` plus a TOP-LEVEL `output_config.effort`
//     enum. The model decides how many thinking tokens to spend within
//     the effort bucket. REQUIRED on Opus 4.7+; manual mode returns 400.
//   - **Manual** (Opus 4.5- and Sonnet 3.7, deprecated on 4.6 family) —
//     `thinking.type = "enabled"` plus `thinking.budget_tokens`. The
//     caller pins the exact token cap.
//
// Set the field that matches the model's expectation. When both Effort
// and BudgetTokens are set on the same request, the provider emits the
// adaptive shape (Effort wins) — adaptive is the future, manual is
// deprecated.
//
// See [closes #20] for the live failure that motivated the dispatch.
//
// [closes #20]: https://github.com/amit-timalsina/pi-llm-go/issues/20
type ThinkingConfig struct {
	// Effort controls thinking depth on adaptive-thinking models (Opus
	// 4.6+, REQUIRED on 4.7+). Set to one of the Effort* constants.
	// Empty string means "don't request adaptive thinking"; provider
	// will fall back to BudgetTokens (manual mode) if non-zero.
	//
	// On Anthropic the wire shape is `{ "thinking": { "type":
	// "adaptive" }, "output_config": { "effort": <Effort> } }` (note
	// output_config is a TOP-LEVEL request field, not nested under
	// thinking).
	Effort Effort

	// BudgetTokens is the manual-mode thinking-token cap. Required on
	// Opus 4.5 and older / Sonnet 3.7. Deprecated on Opus/Sonnet 4.6.
	// Rejected on Opus 4.7+ — those models REQUIRE Effort and return
	// a 400 if budget_tokens appears on the wire.
	//
	// Anthropic requires Request.MaxTokens > BudgetTokens (thinking
	// tokens count against max_tokens). A common safe choice is
	// MaxTokens == BudgetTokens * 2.
	//
	// Provider minimum: Anthropic refuses BudgetTokens < 1024.
	//
	// When both Effort and BudgetTokens are set, Effort wins (adaptive
	// shape emitted, BudgetTokens ignored). This lets callers pre-set
	// both during the migration without per-call branching.
	BudgetTokens int
}

// Effort is the adaptive-thinking depth enum. Honored by Anthropic on
// Opus 4.6+ via the top-level `output_config.effort` field. Empty
// string is the zero value and means "don't request adaptive thinking";
// the provider falls back to BudgetTokens (manual mode) if non-zero.
//
// The OpenAI Responses provider has its own per-provider
// ReasoningEffort enum on its Options; the two are intentionally
// separate because the values are not 1:1.
type Effort string

// Effort levels recognized by Anthropic's adaptive thinking. Match the
// wire enum exactly — do NOT alias to numbers or "minimal" / "xhigh"
// without confirming the model accepts them.
const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
)

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
