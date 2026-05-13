package llm

// CacheRetention controls Anthropic prompt-cache breakpoint placement.
// It is a single-knob abstraction: callers pick a retention tier and the
// provider decides where to place the cache_control markers on the
// underlying wire format.
//
// The marker tells Anthropic "everything from the start of the request up
// to and including this block is cacheable; on a subsequent request with
// byte-identical content up to this marker, return a cache hit and bill
// cache-read rates instead of full input rates."
//
// Behavior by value:
//
//   - CacheRetentionNone (zero value, "") — no markers emitted.
//   - CacheRetentionShort — ephemeral markers with the default ~5 minute
//     lifetime, placed at: (a) the System prompt's trailing block, (b) the
//     final Tool in Request.Tools, (c) the last block (any type) of the
//     most recent user-role message. The last-block placement is
//     type-agnostic so that subsequent calls in a tool loop reuse the
//     cached tool_result round-trip instead of re-billing it.
//   - CacheRetentionLong — same placement as Short with TTL "1h" and the
//     "extended-cache-ttl-2025-04-11" beta header auto-attached to the
//     outgoing HTTP request.
//
// OpenAI's Chat Completions and Responses providers silently ignore
// CacheRetention — OpenAI caches automatically with no caller-side
// breakpoint API.
//
// 1h-TTL availability and fallback behavior:
//
// All currently-shipped Claude 4 family models (Opus 4.7, Sonnet 4.6,
// Haiku 4.5) support 1h cache TTL. Older Claude 3.x models accept the
// beta header but may silently downgrade the hold to the 5-minute
// default — Anthropic does NOT error on an unsupported model. The
// silent fallback is a real cost-budgeting hazard for callers that
// assumed a 1h-cached prefix would survive across long iterations
// (Noumenal issue #12).
//
// Detect silent fallback via the Usage breakdown: when
// CacheRetention=long was requested, inspect the response message's
// Usage.CacheWrite5mTokens vs CacheWrite1hTokens. If the 5m count is
// non-zero and the 1h count is zero, the model honored the request
// at 5min, not 1h — adjust cost projections accordingly. The two
// fields are populated from Anthropic's cache_creation response
// breakdown; other providers leave them at 0.
//
// The heuristic (5m>0 && 1h==0) assumes UNIFORM TTL placement
// across all cache_control markers in the request, which
// CacheRetention=long guarantees today (every marker carries
// ttl:"1h"). If a future per-block-TTL feature lets callers mix
// 5min and 1h breakpoints in one request, this diagnostic would
// produce false positives and would need a different signal —
// likely a per-block annotation rather than aggregate counts.
//
// Note that as of March 2026, Anthropic regressed the DEFAULT
// ephemeral TTL from 60min to 5min; the 1h tier is now opt-in via
// CacheRetentionLong + the extended-cache-ttl-2025-04-11 beta header
// (which the Anthropic provider auto-attaches when CacheRetention=long).
//
// The caller owns prompt determinism: cached sections must be byte-stable
// across iterations for the cache to hit. Any change (timestamps, map
// iteration order, reordered items) invalidates the cache from that point
// forward in the request.
//
// See https://docs.claude.com/en/docs/build-with-claude/prompt-caching for
// the full discipline.
type CacheRetention string

const (
	// CacheRetentionNone disables prompt caching for this request. This is
	// the zero value of CacheRetention; an unset field and an explicit
	// CacheRetentionNone are byte-identical and produce no cache_control
	// markers.
	CacheRetentionNone CacheRetention = ""

	// CacheRetentionShort places ephemeral cache breakpoints with the
	// default ~5 minute lifetime. The right default for iterative agent
	// loops where the prefix is reused within a single session.
	CacheRetentionShort CacheRetention = "short"

	// CacheRetentionLong places ephemeral cache breakpoints with the 1-hour
	// TTL and auto-attaches the "extended-cache-ttl-2025-04-11" beta header.
	// For long-lived static prefixes (large system prompts, big tool sets)
	// that survive across many sessions.
	CacheRetentionLong CacheRetention = "long"
)
