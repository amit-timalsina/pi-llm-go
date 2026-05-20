package openai_responses_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// TestTool_StrictEmittedAtTopLevel verifies the FLATTER Responses-API
// shape: `strict` lives at the TOP of the function tool (peer of
// name/description/parameters), NOT nested under a `function` wrapper
// the way Chat Completions does it. This is the highest-risk delta vs
// Chat Completions and the load-bearing test for the openai_responses
// dispatch.
func TestTool_StrictEmittedAtTopLevel(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     "gpt-5.4-mini",
		MaxTokens: 256,
		Tools: []llm.Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
			Strict:      true,
		}},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	tool := tools[0].(map[string]any)
	if got := tool["strict"]; got != true {
		t.Errorf("tool.strict at TOP level: got %v, want true (Responses API uses flatter shape; no `function` wrapper)", got)
	}
	if _, nested := tool["function"]; nested {
		t.Errorf("tool has unexpected `function` wrapper (that's the Chat Completions shape): %v", tool)
	}
}

// TestTool_StrictOmittedWhenFalse checks the no-leak default.
func TestTool_StrictOmittedWhenFalse(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     "gpt-5.4-mini",
		MaxTokens: 256,
		Tools:     []llm.Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	// Structured-check rather than string-match — schema text could in
	// principle contain "strict" too. Decode and look at the actual
	// field on the tool object.
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tools, _ := body["tools"].([]any)
	if len(tools) > 0 {
		tool := tools[0].(map[string]any)
		if _, present := tool["strict"]; present {
			t.Errorf("strict leaked into body when Tool.Strict was false: %v", tool)
		}
	}
}

// TestToolChoice_NamedToolUsesFlatterShape is THE critical test for
// the Responses-API delta. Chat Completions uses the nested
// `{"type":"function","function":{"name":"..."}}` shape; Responses
// uses the flatter `{"type":"function","name":"..."}` directly.
// A future Chat-Completions-style copy-paste would silently emit the
// wrong shape here — this test catches that.
func TestToolChoice_NamedToolUsesFlatterShape(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      "gpt-5.4-mini",
		MaxTokens:  256,
		Tools:      []llm.Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceTool, Name: "get_weather"},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tc, ok := body["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice is not an object: %v", body["tool_choice"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_choice.type: got %v, want function", tc["type"])
	}
	if tc["name"] != "get_weather" {
		t.Errorf("tool_choice.name at TOP: got %v, want get_weather (Responses uses FLATTER shape than Chat Completions)", tc["name"])
	}
	if _, nested := tc["function"]; nested {
		t.Errorf("tool_choice has unexpected `function` wrapper (that's Chat Completions, NOT Responses): %v", tc)
	}
}

// TestToolChoice_AnyBecomesRequired verifies the OpenAI keyword remap
// (same as Chat Completions): pi-llm-go's neutral ToolChoiceAny serializes
// as OpenAI's "required" (not "any").
func TestToolChoice_AnyBecomesRequired(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      "gpt-5.4-mini",
		MaxTokens:  256,
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceAny},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	if body["tool_choice"] != "required" {
		t.Errorf("tool_choice: got %v, want \"required\" (OpenAI rename of llm.ToolChoiceAny)", body["tool_choice"])
	}
}

// TestToolChoice_NilOmitsField.
func TestToolChoice_NilOmitsField(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     "gpt-5.4-mini",
		MaxTokens: 256,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if strings.Contains(string(fs.lastBody), `"tool_choice"`) {
		t.Errorf("tool_choice leaked into body when nil: %s", fs.lastBody)
	}
}
