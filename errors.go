package llm

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Sentinel errors. Wrap via APIError or return directly. Use errors.Is to
// branch on them in caller retry / fallback logic.
//
// Hierarchy:
//
//	ErrProvider           // generic "something provider-side broke"
//	├─ ErrServerError     // HTTP 5xx (excluding 529)
//	└─ ErrOverloaded      // HTTP 529 (Anthropic infra overload)
//
// ErrServerError and ErrOverloaded each wrap ErrProvider via
// fmt.Errorf("%w"), so existing callers using errors.Is(err, ErrProvider)
// continue to match 5xx + 529 responses (backward compatible). The two
// child sentinels add specificity for consumers that need distinct
// retry / escalation policies per the category guidance in issue #11.
var (
	ErrAuth           = errors.New("llm: authentication failed")
	ErrRateLimit      = errors.New("llm: rate limited")
	ErrInvalidRequest = errors.New("llm: invalid request")
	ErrProvider       = errors.New("llm: provider error")
	// ErrServerError signals a generic 5xx response (excluding 529).
	// Recommended consumer policy: retry with backoff; if sustained
	// past a threshold, surface for engineer escalation.
	ErrServerError = fmt.Errorf("%w: server error (5xx)", ErrProvider)
	// ErrOverloaded signals an Anthropic-style 529 "overloaded"
	// response. Recommended consumer policy: short backoff (~60s)
	// then retry; consider provider fallback if sustained.
	ErrOverloaded = fmt.Errorf("%w: overloaded (529)", ErrProvider)
)

// APIError wraps a non-2xx HTTP response from a provider. The Inner field
// is one of the sentinel errors above so that errors.Is works through the
// wrapping. Status and Body let callers inspect the raw failure (e.g. to
// parse a structured provider error payload). RetryAfter, when non-zero,
// is the parsed value of the response's Retry-After or retry-after-ms
// header — populated by providers for 429 / 529 responses to support
// caller-side rate-limit / overload backoff.
type APIError struct {
	Provider string
	Status   int
	Body     []byte
	Inner    error
	// RetryAfter, when > 0, is the server's hint for how long the
	// caller should wait before retrying. Parsed from the response's
	// Retry-After header (RFC 7231 seconds-or-HTTP-date) or, for
	// OpenAI-family providers, the `retry-after-ms` header. Zero
	// means "no header present" — callers fall back to their own
	// backoff schedule.
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	if e == nil {
		return ""
	}
	retryHint := ""
	if e.RetryAfter > 0 {
		retryHint = fmt.Sprintf(" retry_after=%s", e.RetryAfter)
	}
	if len(e.Body) > 0 {
		return fmt.Sprintf("%s: provider=%s status=%d%s body=%s", e.Inner, e.Provider, e.Status, retryHint, string(e.Body))
	}
	return fmt.Sprintf("%s: provider=%s status=%d%s", e.Inner, e.Provider, e.Status, retryHint)
}

func (e *APIError) Unwrap() error { return e.Inner }

// SentinelForStatus maps an HTTP status code to the matching sentinel
// error. Used by provider implementations when constructing APIError.
//
// Status to sentinel:
//
//	401, 403       → ErrAuth
//	429            → ErrRateLimit
//	other 4xx      → ErrInvalidRequest
//	529            → ErrOverloaded (Anthropic-style)
//	other 5xx      → ErrServerError
//	otherwise      → ErrProvider
//
// ErrServerError and ErrOverloaded both wrap ErrProvider, so legacy
// errors.Is(err, ErrProvider) keeps working for 5xx / 529 responses.
func SentinelForStatus(status int) error {
	switch {
	case status == 401, status == 403:
		return ErrAuth
	case status == 429:
		return ErrRateLimit
	case status == 529:
		return ErrOverloaded
	case status >= 500:
		return ErrServerError
	case status >= 400:
		return ErrInvalidRequest
	default:
		return ErrProvider
	}
}

// IsRateLimited reports whether err (or anything in its Unwrap chain)
// is a 429 rate-limit error. Equivalent to errors.Is(err, ErrRateLimit)
// — sugar for the common caller-side branch.
func IsRateLimited(err error) bool { return errors.Is(err, ErrRateLimit) }

// IsOverloaded reports whether err is an Anthropic-style 529
// overloaded response.
func IsOverloaded(err error) bool { return errors.Is(err, ErrOverloaded) }

// IsServerError reports whether err is a generic 5xx response
// (excluding 529; check IsOverloaded for that case).
func IsServerError(err error) bool { return errors.Is(err, ErrServerError) }

// ParseRetryAfter extracts a wait hint from a provider response's
// Retry-After / retry-after-ms headers. Returns 0 if no header is
// present or none of them parse.
//
// Supported forms (checked in order):
//
//   - retry-after-ms: integer milliseconds (OpenAI convention).
//   - Retry-After: integer seconds (RFC 7231 delta-seconds).
//   - Retry-After: HTTP-date (RFC 7231; computed as now() delta).
//
// Negative deltas (HTTP-date already in the past) are clamped to 0.
// Callers should still apply their own minimum bound — a 0-second
// server hint usually means "as soon as you can" but pummeling the
// API immediately is rarely productive.
func ParseRetryAfter(headers http.Header) time.Duration {
	if headers == nil {
		return 0
	}
	if ms := strings.TrimSpace(headers.Get("retry-after-ms")); ms != "" {
		if n, err := strconv.Atoi(ms); err == nil && n > 0 {
			return time.Duration(n) * time.Millisecond
		}
	}
	if ra := strings.TrimSpace(headers.Get("Retry-After")); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil {
			if secs < 0 {
				return 0
			}
			return time.Duration(secs) * time.Second
		}
		if t, err := http.ParseTime(ra); err == nil {
			d := time.Until(t)
			if d < 0 {
				return 0
			}
			return d
		}
	}
	return 0
}
