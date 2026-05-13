package llm_test

import (
	"context"
	"errors"
	"net"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
)

func TestIsRetriable_RetriableSet(t *testing.T) {
	t.Parallel()

	retriable := []error{
		llm.ErrRateLimit,
		llm.ErrOverloaded,
		llm.ErrServerError,
		&llm.APIError{Provider: "x", Status: 429, Inner: llm.ErrRateLimit},
		&llm.APIError{Provider: "x", Status: 529, Inner: llm.ErrOverloaded},
		&llm.APIError{Provider: "x", Status: 503, Inner: llm.ErrServerError},
		&net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")},
	}
	for _, e := range retriable {
		if !llm.IsRetriable(e) {
			t.Errorf("expected retriable: %v", e)
		}
	}
}

func TestIsRetriable_NonRetriableSet(t *testing.T) {
	t.Parallel()

	nonRetriable := []error{
		nil,
		errors.New("random error"),
		llm.ErrAuth,
		llm.ErrInvalidRequest,
		llm.ErrContextLength,
		llm.ErrPolicyViolation,
		context.Canceled,
		context.DeadlineExceeded,
		&llm.APIError{Provider: "x", Status: 401, Inner: llm.ErrAuth},
		&llm.APIError{Provider: "x", Status: 400, Inner: llm.ErrInvalidRequest},
	}
	for _, e := range nonRetriable {
		if llm.IsRetriable(e) {
			t.Errorf("expected NOT retriable: %v", e)
		}
	}
}

// TestRunWithRetry_SucceedsAfterTransientFailures verifies the loop
// retries on retriable errors and returns the eventual success.
func TestRunWithRetry_SucceedsAfterTransientFailures(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	policy := llm.RetryPolicy{MaxAttempts: 4, BaseDelay: time.Millisecond, MaxDelay: 10 * time.Millisecond}

	got, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		n := attempts.Add(1)
		if n < 3 {
			return "", &llm.APIError{Provider: "x", Status: 429, Inner: llm.ErrRateLimit}
		}
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("unexpected error after retries: %v", err)
	}
	if got != "ok" {
		t.Errorf("got %q, want ok", got)
	}
	if attempts.Load() != 3 {
		t.Errorf("attempts: got %d, want 3", attempts.Load())
	}
}

func TestRunWithRetry_ExhaustsAttemptsAndReturnsLastError(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	policy := llm.RetryPolicy{MaxAttempts: 3, BaseDelay: time.Millisecond, MaxDelay: 10 * time.Millisecond}

	_, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		attempts.Add(1)
		return "", &llm.APIError{Provider: "x", Status: 529, Inner: llm.ErrOverloaded}
	})
	if err == nil {
		t.Fatal("expected error after exhausting attempts")
	}
	if !errors.Is(err, llm.ErrOverloaded) {
		t.Errorf("expected ErrOverloaded, got %v", err)
	}
	if attempts.Load() != 3 {
		t.Errorf("attempts: got %d, want 3", attempts.Load())
	}
}

func TestRunWithRetry_DoesNotRetryNonRetriableError(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	policy := llm.RetryPolicy{MaxAttempts: 5, BaseDelay: time.Millisecond}

	_, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		attempts.Add(1)
		return "", &llm.APIError{Provider: "x", Status: 401, Inner: llm.ErrAuth}
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llm.ErrAuth) {
		t.Errorf("expected ErrAuth, got %v", err)
	}
	if attempts.Load() != 1 {
		t.Errorf("attempts: got %d, want 1 (no retry on auth error)", attempts.Load())
	}
}

func TestRunWithRetry_ZeroPolicyDisablesRetry(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32

	_, err := llm.RunWithRetry(context.Background(), llm.RetryPolicy{}, func() (string, error) {
		attempts.Add(1)
		return "", llm.ErrRateLimit
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if attempts.Load() != 1 {
		t.Errorf("attempts: got %d, want 1 (zero policy = no retry)", attempts.Load())
	}
}

// TestRunWithRetry_ContextCancellationStops verifies that a canceled
// ctx during the backoff sleep aborts retry promptly with ctx.Err().
func TestRunWithRetry_ContextCancellationStops(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	policy := llm.RetryPolicy{MaxAttempts: 10, BaseDelay: time.Hour, MaxDelay: time.Hour}

	var attempts atomic.Int32
	// Cancel after first attempt so the loop is stuck in the long sleep.
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	_, err := llm.RunWithRetry(ctx, policy, func() (string, error) {
		attempts.Add(1)
		return "", llm.ErrRateLimit
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	if attempts.Load() != 1 {
		t.Errorf("attempts: got %d, want 1 (cancellation should preempt retry)", attempts.Load())
	}
}

// TestRetryPolicy_HonorsServerRetryAfter verifies that an APIError
// carrying RetryAfter dominates the exponential backoff.
func TestRetryPolicy_HonorsServerRetryAfter(t *testing.T) {
	t.Parallel()

	policy := llm.RetryPolicy{MaxAttempts: 2, BaseDelay: time.Hour, MaxDelay: 100 * time.Millisecond}

	var attempts atomic.Int32
	start := time.Now()
	_, err := llm.RunWithRetry(context.Background(), policy, func() (string, error) {
		n := attempts.Add(1)
		if n == 1 {
			// RetryAfter=50ms; should win over the 1h BaseDelay (clamped to 100ms MaxDelay).
			return "", &llm.APIError{
				Provider:   "x",
				Status:     429,
				Inner:      llm.ErrRateLimit,
				RetryAfter: 50 * time.Millisecond,
			}
		}
		return "ok", nil
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("retry took %v; expected ~50ms (RetryAfter hint should dominate BaseDelay)", elapsed)
	}
}

func TestRetryPolicy_DefaultRetryPolicySane(t *testing.T) {
	t.Parallel()

	p := llm.DefaultRetryPolicy()
	if p.MaxAttempts < 2 {
		t.Errorf("MaxAttempts: got %d, want >= 2 (default should be retry-enabled)", p.MaxAttempts)
	}
	if p.BaseDelay <= 0 || p.MaxDelay <= 0 || p.MaxDelay < p.BaseDelay {
		t.Errorf("delays: got base=%v max=%v; want positive and MaxDelay >= BaseDelay", p.BaseDelay, p.MaxDelay)
	}
}
