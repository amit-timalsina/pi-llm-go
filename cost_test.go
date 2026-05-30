package llm_test

import (
	"errors"
	"math"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestApplyPricing_BasicInputOutput(t *testing.T) {
	t.Parallel()

	usage := llm.Usage{InputTokens: 1_000_000, OutputTokens: 500_000}
	p := llm.Pricing{Input: 3.00, Output: 15.00}

	got := llm.ApplyPricing(usage, p)
	if !approxEqual(got.Input, 3.00) {
		t.Errorf("Input: got %v, want 3.00", got.Input)
	}
	if !approxEqual(got.Output, 7.50) {
		t.Errorf("Output: got %v, want 7.50", got.Output)
	}
	if !approxEqual(got.Total(), 10.50) {
		t.Errorf("Total: got %v, want 10.50", got.Total())
	}
}

func TestApplyPricing_CacheTTLBreakdownPreferred(t *testing.T) {
	t.Parallel()

	// When TTL breakdown fields are populated, they price independently.
	// CacheWriteTokens is the wire aggregate, NOT double-counted.
	usage := llm.Usage{
		InputTokens:        100_000,
		OutputTokens:       50_000,
		CacheReadTokens:    1_000_000,
		CacheWriteTokens:   1_000_000,
		CacheWrite5mTokens: 400_000,
		CacheWrite1hTokens: 600_000,
	}
	p := llm.Pricing{
		Input:        3.00,
		Output:       15.00,
		CacheRead:    0.30,
		CacheWrite5m: 3.75,
		CacheWrite1h: 6.00,
	}

	got := llm.ApplyPricing(usage, p)
	wantWrite5m := 0.4 * 3.75
	wantWrite1h := 0.6 * 6.00
	if !approxEqual(got.CacheWrite5m, wantWrite5m) {
		t.Errorf("CacheWrite5m: got %v, want %v", got.CacheWrite5m, wantWrite5m)
	}
	if !approxEqual(got.CacheWrite1h, wantWrite1h) {
		t.Errorf("CacheWrite1h: got %v, want %v", got.CacheWrite1h, wantWrite1h)
	}
	// Total must NOT include CacheWriteTokens; breakdown wins.
	wantTotal := (0.1 * 3.00) + (0.05 * 15.00) + (1.0 * 0.30) + wantWrite5m + wantWrite1h
	if !approxEqual(got.Total(), wantTotal) {
		t.Errorf("Total: got %v, want %v (breakdown should not double-count CacheWriteTokens)", got.Total(), wantTotal)
	}
}

func TestApplyPricing_FallbackToAggregateWhenBreakdownMissing(t *testing.T) {
	t.Parallel()

	// Older Anthropic SDK versions (or providers that don't surface TTL)
	// populate CacheWriteTokens with no breakdown. Fall back to the 5m
	// rate as a best-effort.
	usage := llm.Usage{
		InputTokens:      100_000,
		CacheWriteTokens: 800_000,
		// CacheWrite5mTokens + CacheWrite1hTokens both zero
	}
	p := llm.Pricing{Input: 3.00, CacheWrite5m: 3.75, CacheWrite1h: 6.00}

	got := llm.ApplyPricing(usage, p)
	want5m := 0.8 * 3.75
	if !approxEqual(got.CacheWrite5m, want5m) {
		t.Errorf("CacheWrite5m fallback: got %v, want %v", got.CacheWrite5m, want5m)
	}
	if got.CacheWrite1h != 0 {
		t.Errorf("CacheWrite1h: got %v, want 0 (no breakdown means 1h tier unknown)", got.CacheWrite1h)
	}
}

func TestApplyPricing_NoCacheWriteWhenAllZero(t *testing.T) {
	t.Parallel()

	usage := llm.Usage{InputTokens: 100_000, OutputTokens: 50_000}
	p := llm.Pricing{Input: 3.00, Output: 15.00, CacheWrite5m: 3.75}

	got := llm.ApplyPricing(usage, p)
	if got.CacheWrite5m != 0 || got.CacheWrite1h != 0 {
		t.Errorf("expected zero cache-write cost when no cache-write tokens, got 5m=%v 1h=%v",
			got.CacheWrite5m, got.CacheWrite1h)
	}
}

func TestComputeCost_BuiltInSonnetTable(t *testing.T) {
	t.Parallel()

	usage := llm.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000}
	cost, err := llm.ComputeCost(usage, "claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("ComputeCost(sonnet-4-6): %v", err)
	}
	if !approxEqual(cost.Input, 3.00) || !approxEqual(cost.Output, 15.00) {
		t.Errorf("sonnet-4-6: got input=%v output=%v, want 3.00 / 15.00", cost.Input, cost.Output)
	}
}

// TestComputeCost_OpusFortySix verifies the v0.11.2 addition (closes
// #32). Opus 4.6 ships at the same rate as Opus 4.7; noumenal's AA
// pins this model. Regression guard.
func TestComputeCost_OpusFortySix(t *testing.T) {
	t.Parallel()

	usage := llm.Usage{
		InputTokens:        1_000_000,
		OutputTokens:       100_000,
		CacheReadTokens:    1_000_000,
		CacheWrite5mTokens: 100_000,
		CacheWrite1hTokens: 100_000,
	}
	cost, err := llm.ComputeCost(usage, "claude-opus-4-6")
	if err != nil {
		t.Fatalf("ComputeCost(opus-4-6): %v", err)
	}
	if !approxEqual(cost.Input, 5.00) {
		t.Errorf("input: got %v, want 5.00", cost.Input)
	}
	if !approxEqual(cost.Output, 2.50) {
		t.Errorf("output: got %v, want 2.50", cost.Output)
	}
	if !approxEqual(cost.CacheRead, 0.50) {
		t.Errorf("cache_read: got %v, want 0.50", cost.CacheRead)
	}
	if !approxEqual(cost.CacheWrite5m, 0.625) {
		t.Errorf("cache_write_5m: got %v, want 0.625", cost.CacheWrite5m)
	}
	if !approxEqual(cost.CacheWrite1h, 1.00) {
		t.Errorf("cache_write_1h: got %v, want 1.00", cost.CacheWrite1h)
	}
}

// TestComputeCost_GeminiRoboticsER verifies the v0.11.2 addition (closes
// #32). noumenal's DSA VLM pins this model.
func TestComputeCost_GeminiRoboticsER(t *testing.T) {
	t.Parallel()

	usage := llm.Usage{InputTokens: 1_000_000, OutputTokens: 100_000}
	cost, err := llm.ComputeCost(usage, "gemini-robotics-er-1.6-preview")
	if err != nil {
		t.Fatalf("ComputeCost(gemini-robotics-er-1.6-preview): %v", err)
	}
	if !approxEqual(cost.Input, 1.00) {
		t.Errorf("input: got %v, want 1.00 (text/image/video standard tier)", cost.Input)
	}
	if !approxEqual(cost.Output, 0.50) {
		t.Errorf("output: got %v, want 0.50", cost.Output)
	}
	// Robotics ER has no listed context-caching rate today; cache
	// fields should stay at zero.
	if cost.CacheRead != 0 || cost.CacheWrite5m != 0 || cost.CacheWrite1h != 0 {
		t.Errorf("expected zero cache costs for Robotics ER (no published rate); got read=%v 5m=%v 1h=%v",
			cost.CacheRead, cost.CacheWrite5m, cost.CacheWrite1h)
	}
}

// TestPricingFor_FormerlyUnknownModelsAreNowPriced is the
// backward-compat regression guard for #32: each of the newly-seeded
// model IDs previously returned ok=false from PricingFor (and
// ErrUnknownModel from ComputeCost). Adding seed entries silently
// flips this — callers using errors.Is(err, ErrUnknownModel) to
// graceful-degrade will now start receiving cost data. This test
// pins that transition so a future seed-cleanup that drops an entry
// fails CI.
func TestPricingFor_FormerlyUnknownModelsAreNowPriced(t *testing.T) {
	t.Parallel()

	newlySeeded := []string{
		"claude-opus-4-6",
		"claude-opus-4-5",
		"claude-sonnet-4-5",
		"gemini-robotics-er-1.6-preview",
	}
	for _, m := range newlySeeded {
		if _, ok := llm.PricingFor(m); !ok {
			t.Errorf("PricingFor(%q) = ok:false — seed entry missing; v0.11.2 regression", m)
		}
	}
}

func TestComputeCost_UnknownModelWrapsSentinel(t *testing.T) {
	t.Parallel()

	_, err := llm.ComputeCost(llm.Usage{InputTokens: 100}, "some-fake-model-name")
	if err == nil {
		t.Fatal("expected ErrUnknownModel, got nil")
	}
	if !errors.Is(err, llm.ErrUnknownModel) {
		t.Errorf("err does not wrap ErrUnknownModel: %v", err)
	}
}

func TestRegisterPricing_OverridesSeed(t *testing.T) {
	// Not parallel: writes to the global override map.

	const id = "test-model-override"
	original, ok := llm.PricingFor(id)
	if ok {
		t.Fatalf("test model %q unexpectedly in seed (got %+v)", id, original)
	}
	custom := llm.Pricing{Input: 99.99, Output: 88.88}
	llm.RegisterPricing(id, custom)

	got, ok := llm.PricingFor(id)
	if !ok {
		t.Fatal("registered model not found")
	}
	if got != custom {
		t.Errorf("got %+v, want %+v", got, custom)
	}
}

func TestRegisterPricing_OverridesShadowSeedEntries(t *testing.T) {
	// Override a seed entry; ensure the override wins.

	seedBefore, ok := llm.PricingFor("claude-haiku-4-5")
	if !ok {
		t.Fatal("seed haiku-4-5 missing")
	}
	override := llm.Pricing{Input: 999, Output: 999}
	llm.RegisterPricing("claude-haiku-4-5", override)
	t.Cleanup(func() { llm.RegisterPricing("claude-haiku-4-5", seedBefore) })

	got, ok := llm.PricingFor("claude-haiku-4-5")
	if !ok || got != override {
		t.Errorf("override did not win: got %+v, want %+v", got, override)
	}
}
