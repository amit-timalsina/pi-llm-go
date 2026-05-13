package gemini_test

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
	"github.com/amit-timalsina/pi-llm-go/providers/gemini"
)

// fakeCountServer fakes Gemini's :countTokens endpoint: returns a
// {totalTokens: N} JSON body or an error status.
type fakeCountServer struct {
	totalTokens int
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
			"totalTokens": f.totalTokens,
		})
	}
}

func TestCountTokens_HitsCountEndpointAndReturnsTotal(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{totalTokens: 91}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	got, err := p.CountTokens(context.Background(), llm.Request{
		Model:  gemini.Gemini2_5Flash,
		System: "system prompt",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if got != 91 {
		t.Errorf("totalTokens: got %d, want 91", got)
	}
	wantPath := "/v1beta/models/" + gemini.Gemini2_5Flash + ":countTokens"
	if srv.lastPath != wantPath {
		t.Errorf("path: got %q, want %q", srv.lastPath, wantPath)
	}

	// Body MUST be wrapped in generateContentRequest — the live v1beta
	// endpoint rejects systemInstruction at the top level.
	body := string(srv.lastBody)
	if !strings.Contains(body, `"generateContentRequest":{`) {
		t.Errorf("body missing generateContentRequest wrapper: %s", body)
	}
	if !strings.Contains(body, `"model":"models/`+gemini.Gemini2_5Flash+`"`) {
		t.Errorf("body inner model field missing or wrong: %s", body)
	}
	if !strings.Contains(body, `"contents":[`) {
		t.Errorf("body missing contents array: %s", body)
	}
	if !strings.Contains(body, `"systemInstruction":{`) {
		t.Errorf("body missing systemInstruction: %s", body)
	}
}

// TestCountTokens_OmitsGenerationConfigOnIdleRequest ensures the
// wrapped count_tokens body still omits the generationConfig field
// when no tunables are set. Otherwise a wire-bloat regression could
// silently land where countTokens posts an empty
// `"generationConfig":{}`.
func TestCountTokens_OmitsGenerationConfigOnIdleRequest(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{totalTokens: 10}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	_, err := p.CountTokens(context.Background(), llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hello"}}},
		},
	})
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	body := string(srv.lastBody)
	if strings.Contains(body, `"generationConfig"`) {
		t.Errorf("body should not contain generationConfig for an idle request, got: %s", body)
	}
}

func TestCountTokens_RejectsEmptyModel(t *testing.T) {
	t.Parallel()

	p, err := gemini.New(gemini.Options{APIKey: "test", BaseURL: "https://example.invalid"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CountTokens(context.Background(), llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "x"}}},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatalf("expected 'model is required' error, got %v", err)
	}
}

func TestCountTokens_HTTPErrorWrapsSentinel(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{
		statusCode: http.StatusTooManyRequests,
		statusBody: `{"error":{"message":"quota exceeded"}}`,
	}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	_, err := p.CountTokens(context.Background(), llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err == nil {
		t.Fatal("expected error on 429, got nil")
	}
	if !errors.Is(err, llm.ErrRateLimit) {
		t.Errorf("err does not match ErrRateLimit: %v", err)
	}
}

func TestCountTokens_AsTokenCounterInterface(t *testing.T) {
	t.Parallel()

	srv := &fakeCountServer{totalTokens: 5}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()
	p := newProvider(t, ts)

	var c llm.TokenCounter = p

	n, err := c.CountTokens(context.Background(), llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "x"}}},
		},
	})
	if err != nil {
		t.Fatalf("CountTokens via interface: %v", err)
	}
	if n != 5 {
		t.Errorf("got %d, want 5", n)
	}
}
