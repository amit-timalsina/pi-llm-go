package llm

// seedPricing is a small hand-curated pricing table for canonical models
// across the three providers pi-llm-go ships. Rates are dollars per
// million tokens.
//
// Verified against provider docs on 2026-05-13:
//
//   - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
//   - OpenAI:    https://platform.openai.com/docs/pricing
//   - Gemini:    https://ai.google.dev/gemini-api/docs/pricing
//
// This table is INTENTIONALLY SMALL. The maintenance cost of an
// exhaustive table that tracks every variant (long-context tiers, batch
// discounts, regional multipliers, deprecated models) is real, and
// production callers should override via RegisterPricing rather than
// trust the built-in numbers across upgrades.
//
// For Anthropic models, the three cache rates today follow uniform
// multipliers on the base input rate (5m=1.25×, 1h=2×, read=0.1×). They
// are stored explicitly rather than computed so a future per-model
// policy change (e.g. a "cache-discount premium model" tier) doesn't
// silently rewrite cost projections.
//
// For OpenAI and Gemini, the CacheRead field captures the published
// "cached input" / "context caching" rate. CacheWrite5m and CacheWrite1h
// stay at zero because:
//
//   - OpenAI bills cache writes implicitly in the base input rate (the
//     caller-visible category is only the discounted "cached input"
//     read rate).
//   - Gemini's context caching is single-TTL and the write is not
//     separately metered against an explicit "cache write" line.
//
// Long-context tiers (Gemini Pro models charge 2× over 200k input
// tokens) and batch discounts (Anthropic + OpenAI both offer 50%) are
// out of scope for the seed table — callers needing them should
// register custom Pricing values for their billing scenario.
var seedPricing = map[string]Pricing{
	// Anthropic Claude 4 family.
	"claude-opus-4-7":   {Input: 5.00, Output: 25.00, CacheRead: 0.50, CacheWrite5m: 6.25, CacheWrite1h: 10.00},
	"claude-sonnet-4-6": {Input: 3.00, Output: 15.00, CacheRead: 0.30, CacheWrite5m: 3.75, CacheWrite1h: 6.00},
	"claude-haiku-4-5":  {Input: 1.00, Output: 5.00, CacheRead: 0.10, CacheWrite5m: 1.25, CacheWrite1h: 2.00},

	// OpenAI GPT-5 family. Tokenizers and rates verified against the
	// developers.openai.com pricing endpoint; older gpt-4 family is
	// omitted to keep the seed minimal — register via RegisterPricing
	// if needed.
	"gpt-5.5":      {Input: 5.00, Output: 30.00, CacheRead: 0.50},
	"gpt-5.4":      {Input: 2.50, Output: 15.00, CacheRead: 0.25},
	"gpt-5.4-mini": {Input: 0.75, Output: 4.50, CacheRead: 0.075},

	// Gemini canonical models. Pro tier rates are the ≤200k-input price;
	// long-context (>200k) doubles per Google's published policy and
	// must be handled by a caller-side RegisterPricing override if the
	// caller routinely sends large prompts.
	"gemini-3.1-pro":   {Input: 2.00, Output: 12.00, CacheRead: 0.20},
	"gemini-2.5-pro":   {Input: 1.25, Output: 10.00, CacheRead: 0.125},
	"gemini-2.5-flash": {Input: 0.30, Output: 2.50, CacheRead: 0.03},
}
