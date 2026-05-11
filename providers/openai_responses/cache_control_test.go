package openai_responses_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	openai_responses "github.com/amit-timalsina/pi-llm-go/providers/openai_responses"
)

// Same silent-drop guarantee as providers/openai: pi-llm-go's CacheRetention
// is Anthropic-only; the Responses API also has no caller-side breakpoint
// surface, so we must drop the knob silently rather than error.
//
// Reuses textOnlyPayload + fakeServer from responses_test.go.

func TestCacheRetention_SilentlyDropped(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai_responses.New(openai_responses.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:          "gpt-5.5",
		System:         "stable instructions",
		CacheRetention: llm.CacheRetentionLong,
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

	if strings.Contains(string(fs.lastBody), "cache_control") {
		t.Errorf("Responses wire body unexpectedly contains 'cache_control'; full body:\n%s", string(fs.lastBody))
	}
}
