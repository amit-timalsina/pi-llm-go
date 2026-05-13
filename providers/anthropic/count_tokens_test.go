package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// fakeCountServer is the count_tokens counterpart of fakeServer:
// returns the canned input_tokens JSON or an error status.
type fakeCountServer struct {
	inputTokens int
	statusCode  int
	statusBody  string

	lastPath    string
	lastBody    json.RawMessage
	lastHeaders http.Header
}

func (f *fakeCountServer) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		f.lastPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		f.lastBody = json.RawMessage(body)
		f.lastHeaders = r.Header.Clone()

		if f.statusCode != 0 && f.statusCode != http.StatusOK {
			w.WriteHeader(f.statusCode)
			_, _ = io.WriteString(w, f.statusBody)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"input_tokens": f.inputTokens,
		})
	}
}

func TestCountTokens_HitsCountEndpointAndReturnsCount(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{inputTokens: 137}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	got, err := p.CountTokens(context.Background(), llm.Request{
		Model:  anthropic.ClaudeSonnet4_6,
		System: "you are concise",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if got != 137 {
		t.Errorf("input_tokens: got %d, want 137", got)
	}
	if srv.lastPath != "/v1/messages/count_tokens" {
		t.Errorf("path: got %q, want /v1/messages/count_tokens", srv.lastPath)
	}

	// Body must NOT contain max_tokens or stream (count endpoint rejects them).
	body := string(srv.lastBody)
	if strings.Contains(body, "max_tokens") {
		t.Errorf("body should not contain max_tokens, got: %s", body)
	}
	if strings.Contains(body, `"stream"`) {
		t.Errorf("body should not contain stream field, got: %s", body)
	}
	if !strings.Contains(body, `"model":"claude-sonnet-4-6"`) {
		t.Errorf("body missing model field: %s", body)
	}
}

func TestCountTokens_RejectsEmptyModel(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	_, err := p.CountTokens(context.Background(), llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error for empty model, got nil")
	}
	if !strings.Contains(err.Error(), "model is required") {
		t.Errorf("error message: got %q, want substring %q", err.Error(), "model is required")
	}
}

func TestCountTokens_HTTPErrorWrapsSentinel(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{
		statusCode: http.StatusUnauthorized,
		statusBody: `{"error":{"message":"invalid x-api-key"}}`,
	}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	_, err := p.CountTokens(context.Background(), llm.Request{
		Model: anthropic.ClaudeSonnet4_6,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error on 401, got nil")
	}
	if !errors.Is(err, llm.ErrAuth) {
		t.Errorf("err does not match ErrAuth: %v", err)
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Errorf("err does not unwrap to *APIError: %v", err)
	} else if apiErr.Status != http.StatusUnauthorized {
		t.Errorf("status: got %d, want 401", apiErr.Status)
	}
}

// TestCountTokens_AsTokenCounterInterface ensures the provider satisfies
// llm.TokenCounter — the documented integration point for callers.
func TestCountTokens_AsTokenCounterInterface(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{inputTokens: 42}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	var c llm.TokenCounter = p // compile-time check on the interface satisfaction

	n, err := c.CountTokens(context.Background(), llm.Request{
		Model: anthropic.ClaudeSonnet4_6,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "x"}}},
		},
	})
	if err != nil {
		t.Fatalf("CountTokens via interface: %v", err)
	}
	if n != 42 {
		t.Errorf("got %d, want 42", n)
	}
}
