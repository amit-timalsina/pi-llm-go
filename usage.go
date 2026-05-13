package llm

// Usage records token accounting returned by a provider for a single
// completion request. Cache fields are zero when the provider doesn't
// bill or report cache reads/writes separately.
//
// Cache-write TTL breakdown:
//
//   - CacheWriteTokens is the TOTAL of cache_creation_input_tokens (all
//     TTL tiers summed). Populated by every provider that emits a cache
//     creation count.
//   - CacheWrite5mTokens and CacheWrite1hTokens are the Anthropic-specific
//     breakdown from the `cache_creation.ephemeral_*_input_tokens`
//     response fields. Other providers (OpenAI, Gemini) leave these at
//     zero because their cache surface is opaque or single-TTL.
//
// The TTL breakdown is the structured signal callers use to detect
// "I requested CacheRetention=long but the model silently fell back
// to 5min" (closes issue #12). When CacheRetention=long is requested
// AND CacheWrite5mTokens > 0 AND CacheWrite1hTokens == 0 on the
// response, the 1h hold did NOT take effect — caller-side cost
// budgeting that assumed a 1h-cached prefix needs to adjust.
//
// Backward compat: CacheWriteTokens semantics unchanged — still the
// total, regardless of TTL. CacheWrite5mTokens + CacheWrite1hTokens
// equals CacheWriteTokens for Anthropic responses; the sum may be
// LESS than CacheWriteTokens if Anthropic ships a new TTL tier we
// haven't surfaced yet.
type Usage struct {
	InputTokens  int
	OutputTokens int

	// CacheReadTokens is the total tokens served from a cache hit on
	// this request. Anthropic populates this; OpenAI's cache is
	// opaque (no telemetry); Gemini does not yet surface it.
	CacheReadTokens int

	// CacheWriteTokens is the total tokens written to a cache on
	// this request (all TTL tiers summed for Anthropic).
	CacheWriteTokens int

	// CacheWrite5mTokens is the count of tokens cached at the default
	// ~5-minute TTL on this request. Anthropic-specific. Zero on
	// other providers and zero when no cache write happened or all
	// cached tokens went to a longer TTL.
	CacheWrite5mTokens int

	// CacheWrite1hTokens is the count of tokens cached at the
	// extended 1-hour TTL on this request. Anthropic-specific (gated
	// on the extended-cache-ttl-2025-04-11 beta header which the
	// provider auto-attaches when CacheRetention=long). Zero on
	// other providers and zero when the 1h tier was not honored —
	// the latter is the structured signal Noumenal issue #12 needs
	// to detect silent 5min fallback on unsupported models.
	CacheWrite1hTokens int

	TotalTokens int
}
