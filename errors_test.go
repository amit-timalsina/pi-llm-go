package llm_test

import (
	"errors"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

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
		{500, llm.ErrProvider},
		{503, llm.ErrProvider},
	}
	for _, c := range cases {
		if got := llm.SentinelForStatus(c.status); got != c.want {
			t.Errorf("status=%d: got %v, want %v", c.status, got, c.want)
		}
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
