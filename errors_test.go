package llm_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// TestSentinelForStatus pins the status → sentinel mapping. Includes
// the new ErrServerError (5xx-excluding-529) and ErrOverloaded (529)
// categories from issue #11.
func TestSentinelForStatus(t *testing.T) {
	cases := []struct {
		status int
		want   error
	}{
		{401, llm.ErrAuth},
		{403, llm.ErrAuth},
		{429, llm.ErrRateLimit},
		{400, llm.ErrInvalidRequest},
		{404, llm.ErrInvalidRequest},
		{500, llm.ErrServerError},
		{502, llm.ErrServerError},
		{503, llm.ErrServerError},
		{529, llm.ErrOverloaded},
	}
	for _, c := range cases {
		if got := llm.SentinelForStatus(c.status); got != c.want {
			t.Errorf("status=%d: got %v, want %v", c.status, got, c.want)
		}
	}
}

// TestServerErrorWrapsErrProvider verifies the documented hierarchy:
// ErrServerError and ErrOverloaded both wrap ErrProvider via
// fmt.Errorf("%w"). Backward-compat: existing callers using
// errors.Is(err, ErrProvider) keep matching 5xx + 529 responses.
func TestServerErrorWrapsErrProvider(t *testing.T) {
	if !errors.Is(llm.ErrServerError, llm.ErrProvider) {
		t.Error("ErrServerError must wrap ErrProvider (backward compat)")
	}
	if !errors.Is(llm.ErrOverloaded, llm.ErrProvider) {
		t.Error("ErrOverloaded must wrap ErrProvider (backward compat)")
	}
	// But the two children are parallel — not in each other's chain.
	if errors.Is(llm.ErrOverloaded, llm.ErrServerError) {
		t.Error("ErrOverloaded should NOT match ErrServerError (parallel children, not nested)")
	}
	if errors.Is(llm.ErrServerError, llm.ErrOverloaded) {
		t.Error("ErrServerError should NOT match ErrOverloaded (parallel children, not nested)")
	}
}

func TestAPIErrorUnwrap(t *testing.T) {
	apiErr := &llm.APIError{
		Provider: "anthropic",
		Status:   429,
		Body:     []byte(`{"error":"too many"}`),
		Inner:    llm.ErrRateLimit,
	}
	if !errors.Is(apiErr, llm.ErrRateLimit) {
		t.Errorf("errors.Is should match wrapped sentinel")
	}
	var extracted *llm.APIError
	if !errors.As(apiErr, &extracted) {
		t.Errorf("errors.As should extract APIError")
	}
	if extracted.Status != 429 {
		t.Errorf("Status not preserved: %d", extracted.Status)
	}
}

// TestAPIErrorErrorStringIncludesRetryAfter verifies RetryAfter is
// surfaced in Error() output for debuggability.
func TestAPIErrorErrorStringIncludesRetryAfter(t *testing.T) {
	apiErr := &llm.APIError{
		Provider:   "anthropic",
		Status:     429,
		Inner:      llm.ErrRateLimit,
		RetryAfter: 5 * time.Second,
	}
	s := apiErr.Error()
	if !strings.Contains(s, "retry_after=5s") {
		t.Errorf("Error() should include retry_after=5s; got %q", s)
	}
}

// TestIsHelpersBranchOnCategory verifies the IsRateLimited /
// IsOverloaded / IsServerError sugar helpers route to the right
// sentinel.
func TestIsHelpersBranchOnCategory(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want struct{ rate, over, srv bool }
	}{
		{
			"rate-limited",
			&llm.APIError{Status: 429, Inner: llm.ErrRateLimit},
			struct{ rate, over, srv bool }{rate: true},
		},
		{
			"overloaded",
			&llm.APIError{Status: 529, Inner: llm.ErrOverloaded},
			struct{ rate, over, srv bool }{over: true},
		},
		{
			"server-error",
			&llm.APIError{Status: 503, Inner: llm.ErrServerError},
			struct{ rate, over, srv bool }{srv: true},
		},
		{
			"auth",
			&llm.APIError{Status: 401, Inner: llm.ErrAuth},
			struct{ rate, over, srv bool }{},
		},
		{
			"nil",
			nil,
			struct{ rate, over, srv bool }{},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := llm.IsRateLimited(c.err); got != c.want.rate {
				t.Errorf("IsRateLimited=%v, want %v", got, c.want.rate)
			}
			if got := llm.IsOverloaded(c.err); got != c.want.over {
				t.Errorf("IsOverloaded=%v, want %v", got, c.want.over)
			}
			if got := llm.IsServerError(c.err); got != c.want.srv {
				t.Errorf("IsServerError=%v, want %v", got, c.want.srv)
			}
		})
	}
}

// TestParseRetryAfter pins the supported header forms.
func TestParseRetryAfter(t *testing.T) {
	// retry-after-ms wins when both are present (OpenAI-style, more
	// precise).
	hdr := http.Header{}
	hdr.Set("retry-after-ms", "1500")
	hdr.Set("Retry-After", "10")
	if got := llm.ParseRetryAfter(hdr); got != 1500*time.Millisecond {
		t.Errorf("retry-after-ms should win: got %v, want 1.5s", got)
	}

	// Retry-After as delta-seconds.
	hdr = http.Header{}
	hdr.Set("Retry-After", "30")
	if got := llm.ParseRetryAfter(hdr); got != 30*time.Second {
		t.Errorf("Retry-After delta-seconds: got %v, want 30s", got)
	}

	// Retry-After as HTTP-date (future).
	future := time.Now().Add(2 * time.Minute).UTC().Format(http.TimeFormat)
	hdr = http.Header{}
	hdr.Set("Retry-After", future)
	got := llm.ParseRetryAfter(hdr)
	if got <= 0 || got > 2*time.Minute+5*time.Second {
		t.Errorf("Retry-After HTTP-date future: got %v, want ~2min", got)
	}

	// Retry-After as HTTP-date in the past clamps to 0.
	past := time.Now().Add(-1 * time.Hour).UTC().Format(http.TimeFormat)
	hdr = http.Header{}
	hdr.Set("Retry-After", past)
	if got := llm.ParseRetryAfter(hdr); got != 0 {
		t.Errorf("past HTTP-date should clamp to 0; got %v", got)
	}

	// No headers.
	if got := llm.ParseRetryAfter(nil); got != 0 {
		t.Errorf("nil headers: got %v, want 0", got)
	}
	if got := llm.ParseRetryAfter(http.Header{}); got != 0 {
		t.Errorf("empty headers: got %v, want 0", got)
	}

	// Garbage values fall through to 0.
	hdr = http.Header{}
	hdr.Set("Retry-After", "not-a-number")
	hdr.Set("retry-after-ms", "also-bad")
	if got := llm.ParseRetryAfter(hdr); got != 0 {
		t.Errorf("garbage headers: got %v, want 0", got)
	}
}
