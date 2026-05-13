package llm

import (
	"context"
	"errors"
	"math/rand/v2"
	"net"
	"time"
)

// RetryPolicy configures provider-side retry on retriable errors.
// Zero-value `RetryPolicy{}` (MaxAttempts == 0) disables retry — the
// provider runs each Stream / CountTokens call exactly once and surfaces
// the first error. Set `MaxAttempts >= 2` to opt in.
//
// Wire via the provider's Options.Retry field:
//
//	p, _ := anthropic.New(anthropic.Options{
//	    APIKey: key,
//	    Retry:  &llm.RetryPolicy{MaxAttempts: 4, BaseDelay: time.Second, MaxDelay: 30 * time.Second},
//	})
//
// Retry SCOPE: only the initial HTTP attempt is retriable. Once the
// streaming response begins yielding events (200 OK + first SSE frame
// parsed) the run is committed — a mid-stream connection break
// terminates the iterator with an error rather than retrying, because
// resuming would replay events the consumer already saw. Callers that
// need at-least-once streaming should wrap pi-llm-go in their own
// idempotent replay layer.
//
// Retriable categories: ErrRateLimit (429), ErrOverloaded (529),
// ErrServerError (other 5xx), and transient network errors (DNS,
// connection refused, TLS handshake timeout, etc.). ErrAuth,
// ErrInvalidRequest, and context.Canceled / context.DeadlineExceeded
// are NOT retried.
type RetryPolicy struct {
	// MaxAttempts caps total tries INCLUDING the first attempt. So
	// MaxAttempts=4 means up to 3 retries after the initial failure.
	// Zero or negative disables retry.
	MaxAttempts int

	// BaseDelay is the initial backoff before the first retry.
	// Subsequent attempts double the delay (exponential) up to MaxDelay.
	// Zero defaults to 1 second.
	BaseDelay time.Duration

	// MaxDelay caps each backoff. Server-supplied Retry-After hints
	// from APIError.RetryAfter are honored UP TO this cap (an
	// upstream "retry in 1 hour" hint becomes "wait MaxDelay seconds"
	// to bound the worst case). Zero defaults to 30 seconds.
	MaxDelay time.Duration
}

// DefaultRetryPolicy returns a sane starting point for production use:
// 4 attempts (3 retries), 1s base, 30s cap. Modify as needed before
// passing to the provider's Options.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxAttempts: 4,
		BaseDelay:   time.Second,
		MaxDelay:    30 * time.Second,
	}
}

// IsRetriable reports whether err is one of the retriable categories.
// Used internally by RunWithRetry; exported so callers can build their
// own retry layer on top.
//
// Returns true for:
//   - ErrRateLimit (429)
//   - ErrOverloaded (529)
//   - ErrServerError (other 5xx)
//   - net.Error and *net.OpError (DNS, connect refused, TLS handshake)
//
// Returns false for:
//   - nil
//   - context.Canceled / context.DeadlineExceeded
//   - ErrAuth, ErrInvalidRequest, ErrContextLength, ErrPolicyViolation
//   - any other error (including unwrapped *APIError with non-retriable Inner)
func IsRetriable(err error) bool {
	if err == nil {
		return false
	}
	// Explicit context-cancellation precedence: even if a wrapped
	// net.Error happens to be present, the caller's intent (cancel)
	// trumps the underlying failure mode.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if errors.Is(err, ErrRateLimit) || errors.Is(err, ErrOverloaded) || errors.Is(err, ErrServerError) {
		return true
	}
	// Network-level failures from http.Client.Do — connection refused,
	// DNS failure, TLS handshake timeout, etc. *url.Error wraps these
	// and satisfies net.Error via the embedded net.OpError.
	var netErr net.Error
	return errors.As(err, &netErr)
}

// nextDelay computes the sleep duration before the (attempt+1)-th try
// given the policy and the most recent error. attempt is zero-indexed:
// attempt=0 is the delay before the first RETRY (i.e. after the first
// failure).
//
// Algorithm:
//  1. If err carries a non-zero APIError.RetryAfter, use it (clamped
//     to MaxDelay). Server hints win — they often carry rate-limit
//     reset times the model side knows better than us.
//  2. Otherwise, exponential backoff: BaseDelay * 2^attempt, clamped
//     to MaxDelay.
//  3. Apply FULL jitter: return a random duration in [0, backoff).
//     Full jitter (vs equal jitter) gives better thundering-herd
//     properties — see Marc Brooker's AWS Architecture blog.
func (p RetryPolicy) nextDelay(attempt int, err error) time.Duration {
	base := p.BaseDelay
	if base <= 0 {
		base = time.Second
	}
	maxD := p.MaxDelay
	if maxD <= 0 {
		maxD = 30 * time.Second
	}

	// Server-hinted Retry-After wins (capped at MaxDelay).
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.RetryAfter > 0 {
		d := apiErr.RetryAfter
		if d > maxD {
			d = maxD
		}
		return d
	}

	// Exponential backoff. Guard against overflow on large attempt
	// counts: time.Duration is int64, so 2^attempt past ~62 overflows.
	// Clamp to MaxDelay on overflow.
	var backoff time.Duration
	if attempt >= 62 {
		backoff = maxD
	} else {
		backoff = base << attempt
		if backoff <= 0 || backoff > maxD {
			backoff = maxD
		}
	}

	// Full jitter. rand/v2 N is safe-for-concurrent-use and auto-seeded.
	if backoff <= 0 {
		return 0
	}
	return rand.N(backoff)
}

// RunWithRetry calls do() up to policy.MaxAttempts times, sleeping
// policy.nextDelay between attempts when do() returns a retriable
// error. Returns the first successful result, the first
// non-retriable error, or the last result if all attempts fail.
//
// Honors ctx: sleeping is interruptible via ctx.Done(), and a
// canceled ctx short-circuits before the next attempt.
//
// do MUST construct any single-use resources (request body readers,
// fresh http.Request values) on each call — RunWithRetry invokes do
// from scratch each attempt.
//
// Used internally by provider implementations to wrap their HTTP
// attempt loops; exported so callers building higher-level retry
// (e.g. cross-provider fallback) can compose on top.
func RunWithRetry[T any](ctx context.Context, policy RetryPolicy, do func() (T, error)) (T, error) {
	var (
		zero   T
		result T
		err    error
	)
	attempts := policy.MaxAttempts
	if attempts <= 0 {
		attempts = 1
	}
	for attempt := 0; attempt < attempts; attempt++ {
		// Caller cancellation pre-check — if the ctx was canceled while
		// we were sleeping in the previous iteration, abort here.
		if cerr := ctx.Err(); cerr != nil {
			return zero, cerr
		}

		result, err = do()
		if err == nil {
			return result, nil
		}
		if attempt+1 >= attempts {
			break
		}
		if !IsRetriable(err) {
			return zero, err
		}
		delay := policy.nextDelay(attempt, err)
		if delay > 0 {
			timer := time.NewTimer(delay)
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				return zero, ctx.Err()
			}
		}
	}
	return zero, err
}
