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

// pi-llm-go's CacheControl is an Anthropic-specific feature. OpenAI Chat
// Completions does automatic caching with no caller-side breakpoint API;
// the provider must silently drop CacheControl fields rather than error.
//
// Reuses textOnlyPayload + fakeServer from openai_test.go.

func TestCacheControl_SilentlyDropped(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:              openai.GPT5_5,
		System:             "stable system",
		SystemCacheControl: llm.Ephemeral(), // ignored
		ToolsCacheControl:  llm.EphemeralLong(), // ignored
		Tools: []llm.Tool{
			{Name: "x", InputSchema: json.RawMessage(`{"type":"object"}`),
				CacheControl: llm.Ephemeral()}, // ignored
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "stable preamble", CacheControl: llm.Ephemeral()},
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
