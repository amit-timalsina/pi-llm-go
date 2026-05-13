package llm

import (
	"errors"
	"fmt"
	"sync"
)

// Pricing holds the per-token cost rates for a model. All rates are in
// dollars per million tokens (the industry-standard quoting unit).
//
// Rate semantics by provider:
//
//   - Anthropic publishes input + output rates plus three cache rates
//     (5m write, 1h write, read). The cache rates are deterministic
//     multipliers on the base input rate today (1.25× / 2× / 0.1×) but
//     the multiplier policy could change, so the seed table stores them
//     explicitly.
//   - OpenAI publishes input + output + a single "cached input" rate;
//     CacheRead is the appropriate field. OpenAI has no caller-visible
//     cache-write category — cache reads are billed at the discounted
//     rate; writes implicit in standard input.
//   - Gemini publishes input + output + a single context-caching rate.
//     CacheRead is the appropriate field; CacheWrite5m and CacheWrite1h
//     stay 0 (Gemini's cache is single-TTL and not separately metered
//     for the write).
//
// Pricing is forward-flexible: a provider that ships a new cache tier
// can populate the unused field without breaking existing callers.
type Pricing struct {
	// Input is the per-million-token cost for non-cached input tokens.
	Input float64

	// Output is the per-million-token cost for output tokens. Includes
	// thinking tokens on Anthropic (extended thinking is billed as
	// output).
	Output float64

	// CacheRead is the per-million-token cost for tokens served from
	// a cache hit. On Anthropic this is 0.1× Input by policy; on
	// OpenAI it's roughly 0.1× Input (the "cached input" rate); on
	// Gemini it's the published "context caching" rate.
	CacheRead float64

	// CacheWrite5m is the per-million-token cost for tokens cached at
	// the default ~5 minute TTL. Anthropic-specific (1.25× Input today).
	// Other providers leave this at 0.
	CacheWrite5m float64

	// CacheWrite1h is the per-million-token cost for tokens cached at
	// the extended 1-hour TTL. Anthropic-specific (2× Input today,
	// gated on the extended-cache-ttl-2025-04-11 beta header which the
	// provider auto-attaches when CacheRetention=long). Other providers
	// leave this at 0.
	CacheWrite1h float64
}

// Cost is the dollar breakdown of a single completion's Usage. Sum via
// Total() for a single number.
//
// Categories track Usage fields directly so the diagnostic from
// Usage.CacheWrite5mTokens / CacheWrite1hTokens (silent 5min fallback
// detection) carries into the cost projection.
type Cost struct {
	Input        float64
	Output       float64
	CacheRead    float64
	CacheWrite5m float64
	CacheWrite1h float64
}

// Total returns the sum of all cost categories.
func (c Cost) Total() float64 {
	return c.Input + c.Output + c.CacheRead + c.CacheWrite5m + c.CacheWrite1h
}

// ErrUnknownModel is returned by ComputeCost when the model is not in
// the built-in pricing table and has no caller-registered pricing.
// Callers can register pricing via RegisterPricing.
var ErrUnknownModel = errors.New("llm: unknown model pricing")

// ComputeCost applies the registered pricing for model to usage and
// returns the dollar breakdown. Returns ErrUnknownModel wrapped with
// the model ID when no pricing is registered.
//
// The Anthropic-specific TTL breakdown on Usage.CacheWrite5mTokens /
// CacheWrite1hTokens is honored: each tier prices against its own rate
// so silent 5min fallback (Issue #12 diagnostic) is reflected in the
// cost projection automatically.
//
// When Usage.CacheWriteTokens > 0 but both CacheWrite5mTokens and
// CacheWrite1hTokens are 0 (OpenAI / Gemini, or older Anthropic SDK
// versions that didn't surface the breakdown), the writeable tokens
// fall through to the 5m rate as a best-effort. On non-Anthropic
// providers this rate is 0, so the cost is 0 regardless — matching
// the wire reality (OpenAI's cache is automatic; Gemini doesn't
// separately meter the write).
func ComputeCost(usage Usage, model string) (Cost, error) {
	p, ok := PricingFor(model)
	if !ok {
		return Cost{}, fmt.Errorf("%w: %q", ErrUnknownModel, model)
	}
	return ApplyPricing(usage, p), nil
}

// ApplyPricing is the pure-arithmetic form of ComputeCost: takes a
// caller-supplied Pricing rather than a model lookup. Useful when the
// caller maintains their own pricing source (e.g. a database, models.dev
// poll) outside the built-in table.
func ApplyPricing(usage Usage, p Pricing) Cost {
	c := Cost{
		Input:     perMillion(usage.InputTokens, p.Input),
		Output:    perMillion(usage.OutputTokens, p.Output),
		CacheRead: perMillion(usage.CacheReadTokens, p.CacheRead),
	}

	// Prefer the TTL-tagged breakdown when present. Falls back to a
	// total-only attribution against the 5m rate (the default tier) when
	// the breakdown is missing — see ComputeCost godoc for the rationale.
	switch {
	case usage.CacheWrite5mTokens > 0 || usage.CacheWrite1hTokens > 0:
		c.CacheWrite5m = perMillion(usage.CacheWrite5mTokens, p.CacheWrite5m)
		c.CacheWrite1h = perMillion(usage.CacheWrite1hTokens, p.CacheWrite1h)
	case usage.CacheWriteTokens > 0:
		c.CacheWrite5m = perMillion(usage.CacheWriteTokens, p.CacheWrite5m)
	}

	return c
}

func perMillion(tokens int, ratePerMillion float64) float64 {
	return float64(tokens) * ratePerMillion / 1_000_000
}

// PricingFor returns the registered pricing for a model ID and true,
// or zero Pricing and false if the model is unknown.
//
// Lookup order:
//  1. Pricing registered via RegisterPricing (caller-overrideable).
//  2. The built-in seed table (a small set of canonical models, last
//     verified against the provider docs on the build date — see
//     pricing_seed.go for the verification date).
//
// Pricing changes. Production callers SHOULD verify rates against the
// provider's current pricing page and call RegisterPricing for any
// model they care about, rather than trusting the built-in table
// across upgrades.
func PricingFor(model string) (Pricing, bool) {
	pricingMu.RLock()
	p, ok := pricingOverrides[model]
	pricingMu.RUnlock()
	if ok {
		return p, true
	}
	p, ok = seedPricing[model]
	return p, ok
}

// RegisterPricing registers (or overrides) pricing for a model ID. Safe
// for concurrent use. Registered entries take precedence over the
// built-in seed table.
//
// Production callers maintaining their own pricing source should call
// RegisterPricing once at startup for every model they bill against,
// rather than relying on the built-in table.
func RegisterPricing(model string, p Pricing) {
	pricingMu.Lock()
	defer pricingMu.Unlock()
	if pricingOverrides == nil {
		pricingOverrides = make(map[string]Pricing)
	}
	pricingOverrides[model] = p
}

var (
	pricingMu        sync.RWMutex
	pricingOverrides map[string]Pricing
)
