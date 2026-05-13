package llm_test

import (
	"errors"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

func TestClassifyInvalidRequest_ContextLengthPatterns(t *testing.T) {
	t.Parallel()

	cases := []string{
		// Anthropic
		`{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 200000 tokens > 199999 maximum"}}`,
		// OpenAI
		`{"error":{"message":"This model's maximum context length is 8192 tokens, however you requested 9001.","code":"context_length_exceeded"}}`,
		// Gemini-style
		`{"error":{"code":400,"message":"The request payload exceeds the context window","status":"INVALID_ARGUMENT"}}`,
		// generic "too many tokens"
		`{"error":"too many tokens for this model"}`,
	}
	for _, body := range cases {
		got := llm.ClassifyInvalidRequest([]byte(body))
		if !errors.Is(got, llm.ErrContextLength) {
			t.Errorf("body %q: got %v, want ErrContextLength", body, got)
		}
		if !errors.Is(got, llm.ErrInvalidRequest) {
			t.Errorf("body %q: ErrContextLength should wrap ErrInvalidRequest", body)
		}
	}
}

func TestClassifyInvalidRequest_PolicyViolationPatterns(t *testing.T) {
	t.Parallel()

	cases := []string{
		`{"error":{"message":"Your request was rejected as a content policy violation","code":"content_policy_violation"}}`,
		`{"type":"error","error":{"type":"invalid_request_error","message":"input violates our usage policies"}}`,
		`{"error":{"code":400,"message":"Content blocked by safety filters","status":"INVALID_ARGUMENT"}}`,
	}
	for _, body := range cases {
		got := llm.ClassifyInvalidRequest([]byte(body))
		if !errors.Is(got, llm.ErrPolicyViolation) {
			t.Errorf("body %q: got %v, want ErrPolicyViolation", body, got)
		}
		if !errors.Is(got, llm.ErrInvalidRequest) {
			t.Errorf("body %q: ErrPolicyViolation should wrap ErrInvalidRequest", body)
		}
	}
}

func TestClassifyInvalidRequest_FallsBackToInvalidRequest(t *testing.T) {
	t.Parallel()

	cases := []string{
		``,
		`{"error":{"message":"missing field 'model'"}}`,
		`{"error":{"message":"bad parameter"}}`,
	}
	for _, body := range cases {
		got := llm.ClassifyInvalidRequest([]byte(body))
		if errors.Is(got, llm.ErrContextLength) || errors.Is(got, llm.ErrPolicyViolation) {
			t.Errorf("body %q: classifier promoted to a specific sentinel when it shouldn't have: %v", body, got)
		}
		if !errors.Is(got, llm.ErrInvalidRequest) {
			t.Errorf("body %q: should fall back to ErrInvalidRequest, got %v", body, got)
		}
	}
}

func TestSentinelFor_UpgradesOnlyOn400(t *testing.T) {
	t.Parallel()

	contextBody := []byte(`{"error":{"message":"context length exceeded"}}`)

	// 400 with context-length body upgrades.
	got := llm.SentinelFor(400, contextBody)
	if !errors.Is(got, llm.ErrContextLength) {
		t.Errorf("400 + context body: got %v, want ErrContextLength", got)
	}

	// 401 stays ErrAuth even if body contains the pattern.
	got = llm.SentinelFor(401, contextBody)
	if !errors.Is(got, llm.ErrAuth) {
		t.Errorf("401: got %v, want ErrAuth", got)
	}

	// 429 stays ErrRateLimit.
	got = llm.SentinelFor(429, contextBody)
	if !errors.Is(got, llm.ErrRateLimit) {
		t.Errorf("429: got %v, want ErrRateLimit", got)
	}

	// 500 stays ErrServerError.
	got = llm.SentinelFor(500, contextBody)
	if !errors.Is(got, llm.ErrServerError) {
		t.Errorf("500: got %v, want ErrServerError", got)
	}

	// 400 with no body stays generic.
	got = llm.SentinelFor(400, nil)
	if errors.Is(got, llm.ErrContextLength) || errors.Is(got, llm.ErrPolicyViolation) {
		t.Errorf("400 + nil body: got %v, want generic ErrInvalidRequest", got)
	}
	if !errors.Is(got, llm.ErrInvalidRequest) {
		t.Errorf("400 + nil body: should be ErrInvalidRequest, got %v", got)
	}
}

func TestHelpers_IsContextLength_IsPolicyViolation(t *testing.T) {
	t.Parallel()

	wrapped := &llm.APIError{Provider: "x", Status: 400, Inner: llm.ErrContextLength}
	if !llm.IsContextLength(wrapped) {
		t.Error("IsContextLength: expected true on APIError wrapping ErrContextLength")
	}
	if llm.IsPolicyViolation(wrapped) {
		t.Error("IsPolicyViolation: expected false on context-length error")
	}

	policy := &llm.APIError{Provider: "x", Status: 400, Inner: llm.ErrPolicyViolation}
	if !llm.IsPolicyViolation(policy) {
		t.Error("IsPolicyViolation: expected true on APIError wrapping ErrPolicyViolation")
	}
	if llm.IsContextLength(policy) {
		t.Error("IsContextLength: expected false on policy-violation error")
	}
}
