package llm_test

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// captureHandler is a slog.Handler that records every accepted record
// into an in-memory slice. Used to assert the structured field set
// emitted by RunWithRetry on each retry attempt.
type captureHandler struct {
	mu       sync.Mutex
	records  []slog.Record
	minLevel slog.Level
}

func (h *captureHandler) Enabled(_ context.Context, l slog.Level) bool { return l >= h.minLevel }

func (h *captureHandler) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, r.Clone())
	return nil
}

func (h *captureHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *captureHandler) WithGroup(string) slog.Handler      { return h }

// recordOf returns the first record whose Message equals msg, or nil.
func (h *captureHandler) recordOf(msg string) *slog.Record {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := range h.records {
		if h.records[i].Message == msg {
			r := h.records[i]
			return &r
		}
	}
	return nil
}

// attrMap flattens a record's attributes into a name→value map for assertions.
func attrMap(r *slog.Record) map[string]slog.Value {
	out := map[string]slog.Value{}
	r.Attrs(func(a slog.Attr) bool {
		out[a.Key] = a.Value
		return true
	})
	return out
}

// installCaptureHandler replaces slog.Default for the duration of the
// test and restores the prior default on cleanup. Captures at
// LevelDebug so it surfaces RunWithRetry's DEBUG emission. Tests
// using this MUST NOT run in parallel — slog.Default is process-
// global state.
func installCaptureHandler(t *testing.T) *captureHandler {
	t.Helper()
	prior := slog.Default()
	h := &captureHandler{minLevel: slog.LevelDebug}
	slog.SetDefault(slog.New(h))
	t.Cleanup(func() { slog.SetDefault(prior) })
	return h
}

// TestRetry_SlogPerAttemptRecord asserts that `"llm.retry.attempt"` is
// emitted for each retriable failure with the documented field set.
// Closes pi-llm-go issue #29.
func TestRetry_SlogPerAttemptRecord(t *testing.T) {
	h := installCaptureHandler(t)

	policy := llm.RetryPolicy{
		MaxAttempts: 3,
		BaseDelay:   time.Millisecond,
		MaxDelay:    5 * time.Millisecond,
	}

	var attempts int
	_, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		attempts++
		// Always retriable; surface RetryAfter on the first failure to
		// verify the optional retry_after_ms field reaches the record.
		if attempts == 1 {
			return "", &llm.APIError{
				Provider:   "x",
				Status:     429,
				Inner:      llm.ErrRateLimit,
				RetryAfter: 50 * time.Millisecond,
			}
		}
		if attempts == 2 {
			return "ok", nil
		}
		t.Fatalf("unexpected third attempt")
		return "", nil
	})
	if err != nil {
		t.Fatalf("RunWithRetry: %v", err)
	}

	r := h.recordOf("llm.retry.attempt")
	if r == nil {
		t.Fatal("expected an llm.retry.attempt record; got none")
	}
	attrs := attrMap(r)

	if got := attrs["attempt"].Int64(); got != 1 {
		t.Errorf("attempt: got %d, want 1 (1-indexed)", got)
	}
	if got := attrs["max_attempts"].Int64(); got != 3 {
		t.Errorf("max_attempts: got %d, want 3", got)
	}
	if got := attrs["cause"].String(); got != "rate_limit" {
		t.Errorf("cause: got %q, want rate_limit", got)
	}
	if got := attrs["retry_after_ms"].Int64(); got != 50 {
		t.Errorf("retry_after_ms: got %d, want 50 (from APIError.RetryAfter)", got)
	}
	if got := attrs["error"].String(); !strings.Contains(got, "rate limited") {
		t.Errorf("error: got %q, want substring 'rate limited'", got)
	}
	// delay_ms is jittered and clamped — assert it's in (0, MaxDelay]
	// rather than expecting an exact value. The RetryAfter (50ms) is
	// above MaxDelay (5ms) so the policy clamps to MaxDelay.
	if got := attrs["delay_ms"].Int64(); got <= 0 || got > 5 {
		t.Errorf("delay_ms: got %d, want (0,5] ms (RetryAfter clamped to MaxDelay)", got)
	}
}

// TestRetry_SlogClassifiesCause verifies the cause string mapping
// across the four retriable categories.
func TestRetry_SlogClassifiesCause(t *testing.T) {
	cases := []struct {
		name    string
		failure error
		want    string
	}{
		{"rate_limit", &llm.APIError{Provider: "x", Status: 429, Inner: llm.ErrRateLimit}, "rate_limit"},
		{"overloaded", &llm.APIError{Provider: "x", Status: 529, Inner: llm.ErrOverloaded}, "overloaded"},
		{"server_error", &llm.APIError{Provider: "x", Status: 503, Inner: llm.ErrServerError}, "server_error"},
		// net.Error path — DNS / connect refused / TLS handshake all
		// satisfy net.Error via *net.OpError. Closes the documented
		// enum value gap.
		{"net_error", &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}, "net_error"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := installCaptureHandler(t)
			policy := llm.RetryPolicy{MaxAttempts: 2, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond}
			var attempts int
			_, _ = llm.RunWithRetry(context.Background(), policy, func() (string, error) {
				attempts++
				if attempts == 1 {
					return "", tc.failure
				}
				return "ok", nil
			})
			r := h.recordOf("llm.retry.attempt")
			if r == nil {
				t.Fatalf("expected llm.retry.attempt record")
			}
			if got := attrMap(r)["cause"].String(); got != tc.want {
				t.Errorf("cause: got %q, want %q", got, tc.want)
			}
		})
	}
}

// TestRetry_SlogExhaustionRecord asserts the "llm.retry.exhausted"
// record fires once when all attempts are spent on retriable errors.
func TestRetry_SlogExhaustionRecord(t *testing.T) {
	h := installCaptureHandler(t)
	policy := llm.RetryPolicy{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond}

	_, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		return "", &llm.APIError{Provider: "x", Status: 529, Inner: llm.ErrOverloaded}
	})
	if err == nil {
		t.Fatal("expected error after exhausting attempts")
	}

	r := h.recordOf("llm.retry.exhausted")
	if r == nil {
		t.Fatal("expected llm.retry.exhausted record on retry-budget exhaustion")
	}
	attrs := attrMap(r)
	if got := attrs["max_attempts"].Int64(); got != 3 {
		t.Errorf("max_attempts: got %d, want 3", got)
	}
	if got := attrs["last_cause"].String(); got != "overloaded" {
		t.Errorf("last_cause: got %q, want overloaded", got)
	}

	// And exactly (MaxAttempts-1) attempt records — last attempt has
	// no following delay because budget is spent.
	h.mu.Lock()
	var attemptRecords int
	for _, r := range h.records {
		if r.Message == "llm.retry.attempt" {
			attemptRecords++
		}
	}
	h.mu.Unlock()
	if attemptRecords != 2 {
		t.Errorf("attempt records: got %d, want 2 (MaxAttempts-1)", attemptRecords)
	}
}

// TestRetry_SlogSilentOnFirstSuccess verifies the no-noise contract:
// a successful first call emits ZERO records.
func TestRetry_SlogSilentOnFirstSuccess(t *testing.T) {
	h := installCaptureHandler(t)
	policy := llm.DefaultRetryPolicy()

	_, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(h.records) != 0 {
		t.Errorf("expected 0 records on first-attempt success; got %d", len(h.records))
	}
}

// TestRetry_SlogSilentOnNonRetriableError verifies that a
// non-retriable error returns immediately without logging — the
// caller sees the error directly; the telemetry surface needs no
// narration.
func TestRetry_SlogSilentOnNonRetriableError(t *testing.T) {
	h := installCaptureHandler(t)
	policy := llm.RetryPolicy{MaxAttempts: 5, BaseDelay: time.Millisecond}

	_, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		return "", &llm.APIError{Provider: "x", Status: 401, Inner: llm.ErrAuth}
	})
	if err == nil || !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("expected ErrAuth, got %v", err)
	}
	if len(h.records) != 0 {
		t.Errorf("expected 0 records on non-retriable error; got %d", len(h.records))
	}
}

// TestRetry_SlogEmitOrderingCtxCancelMidSleep locks the ordering
// invariant: the per-attempt slog record is emitted BEFORE the
// backoff sleep. If the consumer cancels mid-sleep, the attempt
// record is on the books (so ops can see "we were retrying when the
// cancel arrived") while the exhaustion record is NOT emitted (we
// didn't surrender, the caller cancelled).
//
// A future refactor that inverts this ordering (sleep then emit)
// would silently hide all retries that get cancelled — this test
// freezes the current contract.
func TestRetry_SlogEmitOrderingCtxCancelMidSleep(t *testing.T) {
	h := installCaptureHandler(t)
	policy := llm.RetryPolicy{
		MaxAttempts: 5,
		BaseDelay:   100 * time.Millisecond, // long enough to cancel during
		MaxDelay:    200 * time.Millisecond,
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	var attempts int
	_, err := llm.RunWithRetry(ctx, policy, func() (string, error) {
		attempts++
		return "", &llm.APIError{Provider: "x", Status: 529, Inner: llm.ErrOverloaded}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	if h.recordOf("llm.retry.attempt") == nil {
		t.Error("expected an llm.retry.attempt record before the cancel landed")
	}
	if h.recordOf("llm.retry.exhausted") != nil {
		t.Error("did not expect an exhausted record on ctx-cancel (we did not surrender)")
	}
}

// TestRetry_SlogTruncatesLongErrorMessage verifies the err field cap.
// Provider error bodies can be multi-kilobyte; the truncated form
// stays scannable for log aggregators.
func TestRetry_SlogTruncatesLongErrorMessage(t *testing.T) {
	h := installCaptureHandler(t)
	policy := llm.RetryPolicy{MaxAttempts: 2, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond}

	bigBody := strings.Repeat("x", 4000)
	_, _ = llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		return "", &llm.APIError{
			Provider: "x",
			Status:   529,
			Body:     []byte(bigBody),
			Inner:    llm.ErrOverloaded,
		}
	})
	r := h.recordOf("llm.retry.attempt")
	if r == nil {
		t.Fatal("expected llm.retry.attempt record")
	}
	got := attrMap(r)["error"].String()
	// Cap is 256 + the "...(truncated)" sentinel ≈ 271 chars max.
	if len(got) > 300 {
		t.Errorf("err truncation failed: got %d chars, want <=300", len(got))
	}
	if !strings.HasSuffix(got, "(truncated)") {
		t.Errorf("err should end with '(truncated)' marker; got tail %q", got[len(got)-20:])
	}
}
