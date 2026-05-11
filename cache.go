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
//   - "" (zero) or CacheRetentionNone — no markers emitted.
//   - CacheRetentionShort — ephemeral markers with the default ~5 minute
//     lifetime, placed at: (a) the System prompt's trailing block, (b) the
//     final Tool in Request.Tools, (c) the last text block of the most
//     recent user message.
//   - CacheRetentionLong — same placement as Short with TTL "1h" and the
//     "extended-cache-ttl-2025-04-11" beta header auto-attached to the
//     outgoing HTTP request.
//
// OpenAI's Chat Completions and Responses providers silently ignore
// CacheRetention — OpenAI caches automatically with no caller-side
// breakpoint API.
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
	// CacheRetentionNone disables prompt caching for this request. Zero
	// value of CacheRetention; the provider emits no cache_control markers.
	CacheRetentionNone CacheRetention = "none"

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
