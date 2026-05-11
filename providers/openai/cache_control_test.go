package openai_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/openai"
)

// pi-llm-go's CacheRetention is Anthropic-specific. OpenAI Chat Completions
// does automatic caching with no caller-side breakpoint API; the provider
// must silently drop CacheRetention rather than error.
//
// Reuses textOnlyPayload + fakeServer from openai_test.go.

func TestCacheRetention_SilentlyDropped(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:          openai.GPT5_5,
		System:         "stable system",
		CacheRetention: llm.CacheRetentionLong, // ignored
		Tools: []llm.Tool{
			{Name: "x", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "stable preamble"},
				llm.TextBlock{Text: "dynamic"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// The wire body must contain ZERO cache_control fields anywhere. We
	// search the raw JSON for the string — cheaper and more thorough than
	// a deep-walk through map[string]any.
	if strings.Contains(string(fs.lastBody), "cache_control") {
		t.Errorf("OpenAI wire body unexpectedly contains 'cache_control'; full body:\n%s", string(fs.lastBody))
	}
}
