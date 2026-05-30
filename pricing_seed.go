package llm

// seedPricing is a small hand-curated pricing table for canonical models
// across the three providers pi-llm-go ships. Rates are dollars per
// million tokens.
//
// Last verified against provider docs on 2026-05-30 (this PR
// re-verified the Anthropic + Gemini tables in full while adding
// Opus 4.6 / Opus 4.5 / Sonnet 4.5 / Robotics ER 1.6 Preview;
// OpenAI not re-checked this round). Per-entry verification history
// lives in git blame.
//
//   - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
//   - OpenAI:    https://developers.openai.com/api/docs/pricing
//   - Gemini:    https://ai.google.dev/gemini-api/docs/pricing
//
// Pricing surprise log (so a re-verification spike doesn't repeat my
// double-takes):
//
//   - Claude Opus 4.5+ dropped from $15/$75 to $5/$25. Opus 4.1 and
//     earlier remain at the legacy $15/$75 tier and are not in this
//     seed (call RegisterPricing if you target them).
//   - GPT-5.5 lands at $5/$30 with cached at $0.50 — a premium step up
//     from the previous gpt-5 family ($1.25/$10). The discounted
//     "cached input" rate is exactly 0.1× input, consistent with
//     OpenAI's published cache-read policy.
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
	//
	// Opus 4.5 / 4.6 / 4.7 all ship at the SAME prices per Anthropic's
	// pricing table (verified 2026-05-30) — the rate cut from the
	// $15/$75 Opus 4 / 4.1 tier to $5/$25 applied at Opus 4.5 and
	// has held through 4.7. Sonnet 4.5 and 4.6 likewise share rates.
	// Listed explicitly so noumenal's Actioning Agent (pins Opus 4.6
	// because Opus 4.7 deprecates `temperature` and the AA needs
	// temperature=0 for reproducibility — closes #32) gets a direct
	// hit instead of ErrUnknownModel.
	"claude-opus-4-7":   {Input: 5.00, Output: 25.00, CacheRead: 0.50, CacheWrite5m: 6.25, CacheWrite1h: 10.00},
	"claude-opus-4-6":   {Input: 5.00, Output: 25.00, CacheRead: 0.50, CacheWrite5m: 6.25, CacheWrite1h: 10.00},
	"claude-opus-4-5":   {Input: 5.00, Output: 25.00, CacheRead: 0.50, CacheWrite5m: 6.25, CacheWrite1h: 10.00},
	"claude-sonnet-4-6": {Input: 3.00, Output: 15.00, CacheRead: 0.30, CacheWrite5m: 3.75, CacheWrite1h: 6.00},
	"claude-sonnet-4-5": {Input: 3.00, Output: 15.00, CacheRead: 0.30, CacheWrite5m: 3.75, CacheWrite1h: 6.00},
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
	//
	// Robotics ER 1.6 Preview lists $1 text/image/video input and $2
	// audio on Google AI Studio pricing (verified 2026-05-30). The
	// seed entry uses the text/image/video rate — consistent with our
	// gemini-2.5-flash entry which also uses the standard
	// text/image/video rate over its audio rate. Audio-heavy callers
	// should `RegisterPricing` with their billing-mix-appropriate
	// blend. Closes #32 (noumenal DSA VLM default per ADR-024).
	"gemini-3.1-pro":                 {Input: 2.00, Output: 12.00, CacheRead: 0.20},
	"gemini-2.5-pro":                 {Input: 1.25, Output: 10.00, CacheRead: 0.125},
	"gemini-2.5-flash":               {Input: 0.30, Output: 2.50, CacheRead: 0.03},
	"gemini-robotics-er-1.6-preview": {Input: 1.00, Output: 5.00},
}
