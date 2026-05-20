package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// TestTool_StrictEmittedOnWire verifies Tool.Strict=true serializes
// as the top-level `strict` field on the tool definition (peer of
// name/description/input_schema) per Anthropic's wire contract.
// Closes #26 — strict tool use.
func TestTool_StrictEmittedOnWire(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`)
	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Tools:     []llm.Tool{{Name: "get_weather", Description: "Get weather", InputSchema: schema, Strict: true}},
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "Weather in Tokyo?"}}}},
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
		t.Errorf("tool.strict: got %v, want true", got)
	}
}

// TestTool_StrictFalseOmittedFromWire verifies the default-non-strict
// case: when Strict=false, the `strict` field is OMITTED from the wire
// body entirely (not sent as `"strict":false`). Some Anthropic-compat
// hosts may not yet support the field; sending only when needed
// minimizes the cross-host surface.
func TestTool_StrictFalseOmittedFromWire(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
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

// TestToolChoice_AutoEmitsAutoType verifies the explicit Auto case
// (vs nil = no field).
func TestToolChoice_AutoEmitsAutoType(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      anthropic.ClaudeSonnet4_6,
		MaxTokens:  1024,
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceAuto},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tc, ok := body["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "auto" {
		t.Errorf("tool_choice: got %v, want {type:auto}", body["tool_choice"])
	}
}

// TestToolChoice_AnyEmitsAnyType verifies that ToolChoiceAny becomes
// the Anthropic-flavored "any" keyword (NOT OpenAI's "required").
func TestToolChoice_AnyEmitsAnyType(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      anthropic.ClaudeSonnet4_6,
		MaxTokens:  1024,
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceAny},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tc, ok := body["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "any" {
		t.Errorf("tool_choice: got %v, want {type:any}", body["tool_choice"])
	}
}

// TestToolChoice_ToolForcesNamedTool verifies the named-tool case:
// type=tool + name=<Name>.
func TestToolChoice_ToolForcesNamedTool(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      anthropic.ClaudeSonnet4_6,
		MaxTokens:  1024,
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
	if tc["type"] != "tool" || tc["name"] != "get_weather" {
		t.Errorf("tool_choice: got %v, want {type:tool, name:get_weather}", body["tool_choice"])
	}
}

// TestToolChoice_NoneEmitsNoneType.
func TestToolChoice_NoneEmitsNoneType(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      anthropic.ClaudeSonnet4_6,
		MaxTokens:  1024,
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceNone},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tc, _ := body["tool_choice"].(map[string]any)
	if tc["type"] != "none" {
		t.Errorf("tool_choice: got %v, want {type:none}", body["tool_choice"])
	}
}

// TestToolChoice_NilOmitsField verifies the no-tool-choice default:
// when Request.ToolChoice is nil, the `tool_choice` field must be
// absent from the wire body.
func TestToolChoice_NilOmitsField(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if strings.Contains(string(fs.lastBody), `"tool_choice"`) {
		t.Errorf("tool_choice leaked into body when nil: %s", fs.lastBody)
	}
}

// TestToolChoice_ToolMissingNameIsBuildError verifies validation: a
// caller setting Type=Tool without Name should get an error before
// the request leaves the client (Anthropic 400s otherwise).
func TestToolChoice_ToolMissingNameIsBuildError(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	var sawErr error
	for _, err := range p.Stream(context.Background(), llm.Request{
		Model:      anthropic.ClaudeSonnet4_6,
		MaxTokens:  1024,
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceTool /* Name missing */},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}) {
		if err != nil {
			sawErr = err
			break
		}
	}
	if sawErr == nil || !strings.Contains(sawErr.Error(), "Name") {
		t.Errorf("expected build error mentioning Name, got %v", sawErr)
	}
}
