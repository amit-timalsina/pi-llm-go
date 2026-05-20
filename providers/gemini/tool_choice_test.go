package gemini_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/gemini"
)

// TestToolChoice_AnyMapsToANYMode verifies that pi-llm-go's
// llm.ToolChoiceAny becomes Gemini's `mode: "ANY"`. Closes #26 for
// Gemini wire shape.
func TestToolChoice_AnyMapsToANYMode(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      gemini.Gemini2_5Flash,
		MaxTokens:  256,
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceAny},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	tc, ok := body["toolConfig"].(map[string]any)
	if !ok {
		t.Fatalf("missing toolConfig: %v", body)
	}
	fcc, _ := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("functionCallingConfig.mode: got %v, want ANY", fcc["mode"])
	}
}

// TestToolChoice_ToolBecomesANYPlusAllowedFunctionNames verifies the
// named-tool case maps to ANY + allowedFunctionNames=[Name]. Gemini
// has no dedicated "force this exact tool" mode; ANY + a 1-element
// allowlist is the closest semantic.
func TestToolChoice_ToolBecomesANYPlusAllowedFunctionNames(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:      gemini.Gemini2_5Flash,
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
	tc, _ := body["toolConfig"].(map[string]any)
	fcc, _ := tc["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Errorf("mode: got %v, want ANY", fcc["mode"])
	}
	allowed, _ := fcc["allowedFunctionNames"].([]any)
	if len(allowed) != 1 || allowed[0] != "get_weather" {
		t.Errorf("allowedFunctionNames: got %v, want [get_weather]", allowed)
	}
}

// TestTool_StrictIgnoredOnGemini verifies that Tool.Strict=true does
// NOT leak any new field into the Gemini wire body. Gemini has no
// per-tool strict mode; the field is silently ignored (documented on
// llm.Tool.Strict's godoc).
func TestTool_StrictIgnoredOnGemini(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     gemini.Gemini2_5Flash,
		MaxTokens: 256,
		Tools:     []llm.Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`), Strict: true}},
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	// Structured check.
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tools, _ := body["tools"].([]any)
	for _, t0 := range tools {
		decls, _ := t0.(map[string]any)["functionDeclarations"].([]any)
		for _, d := range decls {
			if _, present := d.(map[string]any)["strict"]; present {
				t.Errorf("strict leaked into Gemini function declaration: %v", d)
			}
		}
	}
}
