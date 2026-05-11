package llm

// CacheControl marks a cache breakpoint in a request, enabling provider-side
// prompt caching. The marker tells the provider "everything from the start
// of the request up to and including this block is cacheable; on a
// subsequent request with byte-identical content up to this marker, return
// a cache hit and bill cache-read rates instead of full input rates."
//
// CacheControl is honored by the Anthropic Messages provider; OpenAI's
// Chat Completions and Responses providers silently drop the marker (OpenAI
// does automatic caching with no caller-side breakpoint API). The library
// does not enforce Anthropic's per-request 4-breakpoint limit — if exceeded,
// the API will reject the request with a clear error.
//
// Place a CacheControl on:
//   - any Block in a Message's Content slice — caches up to that block.
//   - any Tool in Request.Tools — caches up to that tool definition.
//   - Request.SystemCacheControl — caches the System prompt as a whole.
//   - Request.ToolsCacheControl — caches the full Tools section (shortcut for
//     placing CacheControl on the last Tool in the slice).
//
// The caller owns prompt determinism: cached sections must be byte-stable
// across iterations for the cache to hit. Any change (timestamps, map
// iteration order, reordered items) invalidates the cache from that point
// forward in the request.
//
// See https://docs.claude.com/en/docs/build-with-claude/prompt-caching for
// the full discipline.
type CacheControl struct {
	// Type is the cache strategy. Anthropic's only supported value today is
	// "ephemeral"; future modes may add more. Treated as a tagged enum-via-
	// string so the API can grow without a Go-side breaking change.
	Type string

	// TTL extends the default ~5 minute ephemeral lifetime. Set to "1h" to
	// request 1-hour cache retention; the Anthropic provider then auto-
	// applies the "extended-cache-ttl-2025-04-11" beta header on the
	// outgoing HTTP request. Leave empty for the default 5-minute lifetime.
	//
	// "1h" is the only non-default value Anthropic supports at the time of
	// writing; the field is a string for the same forward-compatibility
	// reason as Type.
	TTL string
}

// Ephemeral returns a CacheControl with Type set to "ephemeral" and the
// default 5-minute TTL. The most common breakpoint shape.
func Ephemeral() *CacheControl {
	return &CacheControl{Type: "ephemeral"}
}

// EphemeralLong returns a CacheControl with Type "ephemeral" and TTL "1h",
// for the extended 1-hour retention that the Anthropic provider unlocks
// via beta header.
func EphemeralLong() *CacheControl {
	return &CacheControl{Type: "ephemeral", TTL: "1h"}
}
