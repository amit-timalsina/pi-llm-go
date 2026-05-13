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
//	ErrInvalidRequest     // HTTP 4xx (other than 401/403/429)
//	├─ ErrContextLength   // prompt/output exceeds context window
//	└─ ErrPolicyViolation // input flagged by provider safety policy
//
// Child sentinels wrap their parents via fmt.Errorf("%w"), so existing
// callers using errors.Is(err, ErrInvalidRequest) or
// errors.Is(err, ErrProvider) continue to match the full subtree
// (backward compatible). The child sentinels add specificity for
// consumers that need distinct retry / escalation policies.
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
	// ErrContextLength signals that the prompt + tools + thinking
	// budget exceeded the model's context window (or that max_tokens
	// requested more output than the model can produce in the
	// remaining window). Returns as a 400 in practice. Recommended
	// consumer policy: do NOT retry as-is; truncate, summarize, or
	// route to a longer-context model.
	ErrContextLength = fmt.Errorf("%w: context length exceeded", ErrInvalidRequest)
	// ErrPolicyViolation signals that the request was rejected by the
	// provider's content / safety policy. Returns as a 400 in
	// practice. Recommended consumer policy: do NOT retry; surface
	// to the user or route through a moderation step.
	ErrPolicyViolation = fmt.Errorf("%w: content policy violation", ErrInvalidRequest)
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

// ClassifyInvalidRequest inspects an error response body for known
// substrings that identify the request as ErrContextLength or
// ErrPolicyViolation, returning the more-specific sentinel when one
// applies and ErrInvalidRequest otherwise.
//
// Used by providers when constructing APIError on a 4xx response — the
// status code maps to ErrInvalidRequest, but the body usually contains
// a free-form message describing the specific cause. Substring matching
// is a pragmatic last-resort: provider response schemas don't carry a
// canonical machine-readable category for "context too long" vs "policy
// violation" vs generic 4xx, so we pattern-match the message text.
//
// Pattern coverage (lowercased substring matches on body bytes — see
// contextLengthPatterns / policyViolationPatterns for the live list):
//
//   - Context length: phrases like "context length", "context window",
//     "prompt is too long", "too many tokens",
//     "maximum allowed number of output tokens" (Anthropic),
//     "maximum context length is" (OpenAI).
//   - Policy: "content policy" / "content_policy", "policy violation",
//     "violates", "safety", "moderation", "blocked".
//
// Patterns are intentionally SPECIFIC — e.g. the request field name
// "max_tokens" on its own is NOT a context-length signal because a
// generic 400 like "max_tokens must be a positive integer" would
// otherwise be misclassified. The patterns target phrases that only
// appear in genuine context-length / policy responses.
//
// Provider-specific schemas (Anthropic's {"error":{"type":"...","message":"..."}},
// OpenAI's {"error":{"code":"context_length_exceeded","message":"..."}},
// Gemini's Google-style error envelope) all surface human-readable
// messages that hit these patterns. Callers needing structured
// per-provider decoding should branch on apiErr.Body themselves rather
// than rely on this classifier.
func ClassifyInvalidRequest(body []byte) error {
	lower := strings.ToLower(string(body))
	for _, p := range contextLengthPatterns {
		if strings.Contains(lower, p) {
			return ErrContextLength
		}
	}
	for _, p := range policyViolationPatterns {
		if strings.Contains(lower, p) {
			return ErrPolicyViolation
		}
	}
	return ErrInvalidRequest
}

var (
	// contextLengthPatterns are phrases that ONLY appear in genuine
	// context-length errors. The bare token name "max_tokens" was
	// dropped after a cold-context review pointed out it would
	// misclassify "max_tokens must be positive" and similar generic
	// validation errors. The provider-specific phrasings below carry
	// enough context to be unambiguous.
	contextLengthPatterns = []string{
		"context length",
		"context_length",
		"context window",
		"prompt is too long",
		"too many tokens",
		"maximum context",
		// Anthropic-specific: "max_tokens: N > M, which is the maximum
		// allowed number of output tokens for <model>". The phrase
		// "maximum allowed number of output tokens" is unique to this
		// failure mode and avoids the bare-max_tokens false positive.
		"maximum allowed number of output tokens",
		// OpenAI-specific: "This model's maximum context length is
		// 8192 tokens, however you requested ..."
		"maximum context length is",
	}
	policyViolationPatterns = []string{
		"content policy",
		"content_policy",
		"policy violation",
		"violates",
		"safety",
		"moderation",
		"blocked",
	}
)

// SentinelFor returns the most specific sentinel for an HTTP error
// response, combining status-code mapping with body-pattern inspection
// for 400 responses. Equivalent to SentinelForStatus(status) for non-400
// responses; for 400s, upgrades to ErrContextLength / ErrPolicyViolation
// when the body matches a known pattern (see ClassifyInvalidRequest).
//
// Prefer this over SentinelForStatus when constructing APIError values —
// the finer sentinels are part of pi-llm-go's public error surface.
func SentinelFor(status int, body []byte) error {
	s := SentinelForStatus(status)
	if errors.Is(s, ErrInvalidRequest) && len(body) > 0 {
		return ClassifyInvalidRequest(body)
	}
	return s
}

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

// IsContextLength reports whether err is a context-length-exceeded
// error. Sugar for errors.Is(err, ErrContextLength).
func IsContextLength(err error) bool { return errors.Is(err, ErrContextLength) }

// IsPolicyViolation reports whether err is a content-policy rejection.
// Sugar for errors.Is(err, ErrPolicyViolation).
func IsPolicyViolation(err error) bool { return errors.Is(err, ErrPolicyViolation) }

// ParseRetryAfter extracts a wait hint from a provider response's
// Retry-After / retry-after-ms headers. Returns 0 if no header is
// present or none of them parse.
//
// Header precedence (`retry-after-ms` wins when both are present
// because it carries sub-second precision):
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
