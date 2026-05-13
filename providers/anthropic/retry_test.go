package anthropic_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// TestStream_RetryRecoversFrom429 verifies that Options.Retry causes
// Stream to retry a 429 and ultimately succeed when the server returns
// a 200 SSE on the second attempt.
func TestStream_RetryRecoversFrom429(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		n := attempts.Add(1)
		if n == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"type":"rate_limit_error"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, textOnlyPayload)
	}))
	defer srv.Close()

	p, err := anthropic.New(anthropic.Options{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Retry:   &llm.RetryPolicy{MaxAttempts: 3, BaseDelay: 5 * time.Millisecond, MaxDelay: 20 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var events int
	for ev, err := range p.Stream(context.Background(), llm.Request{
		Model:    anthropic.ClaudeSonnet4_6,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		_ = ev
		events++
	}
	if events == 0 {
		t.Error("expected events after retry success; got 0")
	}
	if attempts.Load() != 2 {
		t.Errorf("attempts: got %d, want 2 (one 429, then 200)", attempts.Load())
	}
}

// TestStream_NoRetryWhenPolicyDisabled verifies that without Retry the
// first 429 is surfaced immediately.
func TestStream_NoRetryWhenPolicyDisabled(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer srv.Close()

	p, err := anthropic.New(anthropic.Options{APIKey: "test-key", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var sawErr error
	for _, err := range p.Stream(context.Background(), llm.Request{
		Model:    anthropic.ClaudeSonnet4_6,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}) {
		if err != nil {
			sawErr = err
			break
		}
	}
	if !errors.Is(sawErr, llm.ErrRateLimit) {
		t.Errorf("expected ErrRateLimit, got %v", sawErr)
	}
	if attempts.Load() != 1 {
		t.Errorf("attempts: got %d, want 1 (no retry when policy disabled)", attempts.Load())
	}
}

// TestStream_RetryDoesNotRetry400 verifies that a 400 (non-retriable)
// is surfaced immediately even with retry configured.
func TestStream_RetryDoesNotRetry400(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"prompt is too long"}}`)
	}))
	defer srv.Close()

	p, err := anthropic.New(anthropic.Options{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Retry:   &llm.RetryPolicy{MaxAttempts: 5, BaseDelay: time.Millisecond},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var sawErr error
	for _, err := range p.Stream(context.Background(), llm.Request{
		Model:    anthropic.ClaudeSonnet4_6,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "x"}}}},
	}) {
		if err != nil {
			sawErr = err
			break
		}
	}
	if !errors.Is(sawErr, llm.ErrContextLength) {
		t.Errorf("expected ErrContextLength (400 + 'prompt is too long' body), got %v", sawErr)
	}
	if attempts.Load() != 1 {
		t.Errorf("attempts: got %d, want 1 (400 should not retry)", attempts.Load())
	}
}

// TestStream_RetryRecoversFromNetworkLevelFailure verifies that a
// transport-level error (server hijacks the connection and closes it
// before any response headers) is retried — the IsRetriable net.Error
// branch is exercised end-to-end through RunWithRetry, not just the
// classifier unit test.
func TestStream_RetryRecoversFromNetworkLevelFailure(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		n := attempts.Add(1)
		if n == 1 {
			// Hijack the connection and close it without writing any
			// response. From the client's perspective this surfaces
			// as a net-level error from http.Client.Do, which
			// IsRetriable should classify retriable.
			hj, ok := w.(http.Hijacker)
			if !ok {
				t.Fatalf("response writer does not support hijacking")
			}
			conn, _, err := hj.Hijack()
			if err != nil {
				t.Fatalf("hijack: %v", err)
			}
			_ = conn.Close()
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, textOnlyPayload)
	}))
	defer srv.Close()

	p, err := anthropic.New(anthropic.Options{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Retry:   &llm.RetryPolicy{MaxAttempts: 3, BaseDelay: 5 * time.Millisecond, MaxDelay: 20 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var events int
	for ev, err := range p.Stream(context.Background(), llm.Request{
		Model:    anthropic.ClaudeSonnet4_6,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}) {
		if err != nil {
			t.Fatalf("stream error after retry: %v", err)
		}
		_ = ev
		events++
	}
	if events == 0 {
		t.Error("expected events after network-failure retry; got 0")
	}
	if attempts.Load() != 2 {
		t.Errorf("attempts: got %d, want 2 (one closed, then 200)", attempts.Load())
	}
}

// TestStream_400ClassifiedAsContextLength verifies that the SentinelFor
// upgrade reaches Stream's caller without retry.
func TestStream_400ClassifiedAsPolicyViolation(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"content policy violation"}}`)
	}))
	defer srv.Close()

	p, err := anthropic.New(anthropic.Options{APIKey: "test-key", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var sawErr error
	for _, err := range p.Stream(context.Background(), llm.Request{
		Model:    anthropic.ClaudeSonnet4_6,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "x"}}}},
	}) {
		if err != nil {
			sawErr = err
			break
		}
	}
	if !errors.Is(sawErr, llm.ErrPolicyViolation) {
		t.Errorf("expected ErrPolicyViolation, got %v", sawErr)
	}
}
