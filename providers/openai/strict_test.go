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

// TestTool_StrictEmittedInFunctionObject verifies that Tool.Strict=true
// serializes as `function.strict = true` per OpenAI's wire contract
// (nested under `function`, not at the top of the tool entry). Closes #26.
func TestTool_StrictEmittedInFunctionObject(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     openai.GPT5_5,
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
	fn, _ := tool["function"].(map[string]any)
	if fn == nil {
		t.Fatalf("tool missing function object: %v", tool)
	}
	if got := fn["strict"]; got != true {
		t.Errorf("function.strict: got %v, want true (must be nested under function on OpenAI wire)", got)
	}
}

// TestTool_StrictFalseOmittedFromWire verifies the no-leak default.
func TestTool_StrictFalseOmittedFromWire(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     openai.GPT5_5,
		MaxTokens: 256,
		Tools:     []llm.Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if strings.Contains(string(fs.lastBody), `"strict"`) {
		t.Errorf("strict field leaked into body when not set: %s", fs.lastBody)
	}
}

// TestToolChoice_AnyBecomesRequired verifies the OpenAI keyword
// remapping: pi-llm-go's neutral llm.ToolChoiceAny serializes as
// OpenAI's "required" (not "any" like Anthropic). This is the load-
// bearing cross-provider divergence.
func TestToolChoice_AnyBecomesRequired(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      openai.GPT5_5,
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

// TestToolChoice_AutoBareString.
func TestToolChoice_AutoBareString(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      openai.GPT5_5,
		MaxTokens:  256,
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceAuto},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	if body["tool_choice"] != "auto" {
		t.Errorf("tool_choice: got %v, want \"auto\"", body["tool_choice"])
	}
}

// TestToolChoice_ToolNestedFunctionObject verifies the named-tool case
// uses the Chat-Completions object shape {"type":"function","function":{"name":"..."}}.
func TestToolChoice_ToolNestedFunctionObject(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      openai.GPT5_5,
		MaxTokens:  256,
		Tools:      []llm.Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceTool, Name: "get_weather"},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "Weather in Tokyo?"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tc, _ := body["tool_choice"].(map[string]any)
	if tc["type"] != "function" {
		t.Errorf("tool_choice.type: got %v, want function", tc["type"])
	}
	fn, _ := tc["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("tool_choice.function.name: got %v, want get_weather (Chat Completions nests under function)", fn["name"])
	}
}

// TestToolChoice_NilOmits.
func TestToolChoice_NilOmits(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     openai.GPT5_5,
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
