package llm

import (
	"errors"
	"fmt"
)

// Sentinel errors. Wrap via APIError or return directly. Use errors.Is to
// branch on them in caller retry / fallback logic.
var (
	ErrAuth           = errors.New("llm: authentication failed")
	ErrRateLimit      = errors.New("llm: rate limited")
	ErrInvalidRequest = errors.New("llm: invalid request")
	ErrProvider       = errors.New("llm: provider error")
)

// APIError wraps a non-2xx HTTP response from a provider. The Inner field
// is one of the sentinel errors above so that errors.Is works through the
// wrapping. Status and Body let callers inspect the raw failure (e.g. to
// parse a structured provider error payload).
type APIError struct {
	Provider string
	Status   int
	Body     []byte
	Inner    error
}

func (e *APIError) Error() string {
	if e == nil {
		return ""
	}
	if len(e.Body) > 0 {
		return fmt.Sprintf("%s: provider=%s status=%d body=%s", e.Inner, e.Provider, e.Status, string(e.Body))
	}
	return fmt.Sprintf("%s: provider=%s status=%d", e.Inner, e.Provider, e.Status)
}

func (e *APIError) Unwrap() error { return e.Inner }

// SentinelForStatus maps an HTTP status code to the matching sentinel
// error. Used by provider implementations when constructing APIError.
func SentinelForStatus(status int) error {
	switch {
	case status == 401, status == 403:
		return ErrAuth
	case status == 429:
		return ErrRateLimit
	case status >= 400 && status < 500:
		return ErrInvalidRequest
	default:
		return ErrProvider
	}
}
