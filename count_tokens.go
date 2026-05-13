package llm

import "context"

// TokenCounter is the optional capability of counting input tokens for a
// Request without spending an inference call. Implemented by providers
// whose API exposes a dedicated count-tokens endpoint (Anthropic Messages
// and Gemini both do; both providers document the call as free today —
// confirm against current provider docs if billing matters to you).
//
// Use via type assertion against an LLM value:
//
//	if c, ok := p.(llm.TokenCounter); ok {
//	    n, err := c.CountTokens(ctx, req)
//	    if err != nil { ... }
//	    fmt.Printf("would consume %d input tokens\n", n)
//	}
//
// The OpenAI Chat Completions and Responses providers do NOT implement
// this — OpenAI's tokenization is local-only (tiktoken), which pi-llm-go
// does not bundle. Callers needing pre-flight counts for an OpenAI-hosted
// model should run tiktoken themselves; for OpenAI-compatible self-hosted
// endpoints (vLLM, Ollama), counts are tokenizer-specific and not
// portably reachable.
//
// CountTokens does NOT consume the cache or interact with the inference
// path; calling it does not warm a cache breakpoint. Anthropic's
// count_tokens endpoint accepts the same body shape as /v1/messages
// (system, messages, tools, thinking) but ignores cache_control markers
// and max_tokens. Gemini's countTokens accepts the same contents array
// as generateContent.
type TokenCounter interface {
	// CountTokens returns the number of input tokens the request would
	// consume if streamed. The returned error follows the same wrapping
	// discipline as Stream — HTTP errors surface as *APIError wrapping
	// a sentinel; callers can branch via errors.Is.
	CountTokens(ctx context.Context, req Request) (int, error)
}
