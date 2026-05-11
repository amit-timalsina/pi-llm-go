package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// Reuses textOnlyPayload and fakeServer from anthropic_test.go.

func TestCacheControl_BlockLevelOnWire(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "stable preamble", CacheControl: llm.Ephemeral()},
				llm.TextBlock{Text: "dynamic suffix"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	msgs := body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)

	if len(content) != 2 {
		t.Fatalf("content blocks=%d, want 2", len(content))
	}
	first := content[0].(map[string]any)
	if first["cache_control"] == nil {
		t.Errorf("first block missing cache_control: %+v", first)
	} else {
		cc := first["cache_control"].(map[string]any)
		if cc["type"] != "ephemeral" {
			t.Errorf("first block cache_control.type=%v, want ephemeral", cc["type"])
		}
	}
	second := content[1].(map[string]any)
	if _, has := second["cache_control"]; has {
		t.Errorf("second block should NOT have cache_control: %+v", second)
	}
}

func TestCacheControl_TTLAutoAppliesBetaHeader(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "long-lived prefix", CacheControl: llm.EphemeralLong()},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	betas := fs.lastHeaders.Values("Anthropic-Beta")
	found := false
	for _, b := range betas {
		if b == "extended-cache-ttl-2025-04-11" {
			found = true
		}
	}
	if !found {
		t.Errorf("missing extended-cache-ttl-2025-04-11 beta header; got: %v", betas)
	}

	// And the wire-level TTL field should be "1h".
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	cc := content[0].(map[string]any)["cache_control"].(map[string]any)
	if cc["ttl"] != "1h" {
		t.Errorf("cache_control.ttl=%v, want 1h", cc["ttl"])
	}
}

func TestCacheControl_NoBetaWhenNoLongTTL(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "default 5min TTL", CacheControl: llm.Ephemeral()},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	for _, b := range fs.lastHeaders.Values("Anthropic-Beta") {
		if b == "extended-cache-ttl-2025-04-11" {
			t.Errorf("extended-cache-ttl beta header should NOT be sent for default TTL")
		}
	}
}

func TestCacheControl_SystemPromptStructuredWhenCached(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:              anthropic.ClaudeSonnet4_6,
		MaxTokens:          64,
		System:             "You are a stable assistant.",
		SystemCacheControl: llm.Ephemeral(),
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)

	// When SystemCacheControl is set, "system" must be an array of one block
	// with cache_control attached — NOT a plain string.
	sysVal := body["system"]
	switch s := sysVal.(type) {
	case []any:
		if len(s) != 1 {
			t.Fatalf("system array length=%d, want 1", len(s))
		}
		blk := s[0].(map[string]any)
		if blk["type"] != "text" || blk["text"] != "You are a stable assistant." {
			t.Errorf("system block shape wrong: %+v", blk)
		}
		if blk["cache_control"] == nil {
			t.Errorf("system block missing cache_control: %+v", blk)
		}
	default:
		t.Fatalf("system=%T, want []any (structured) when SystemCacheControl is set", sysVal)
	}
}

func TestCacheControl_SystemPromptPlainWhenNotCached(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		System:    "plain system",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	if s, ok := body["system"].(string); !ok || s != "plain system" {
		t.Errorf("system=%v (%T), want plain string", body["system"], body["system"])
	}
}

func TestCacheControl_PerToolBreakpoint(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		Tools: []llm.Tool{
			{
				Name:         "first",
				Description:  "stable",
				InputSchema:  json.RawMessage(`{"type":"object"}`),
				CacheControl: llm.Ephemeral(),
			},
			{Name: "second", Description: "in flux", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tools := body["tools"].([]any)
	if tools[0].(map[string]any)["cache_control"] == nil {
		t.Errorf("first tool missing cache_control")
	}
	if _, has := tools[1].(map[string]any)["cache_control"]; has {
		t.Errorf("second tool should NOT have cache_control")
	}
}

func TestCacheControl_ToolsShortcutMarksLastTool(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:             anthropic.ClaudeSonnet4_6,
		MaxTokens:         64,
		ToolsCacheControl: llm.Ephemeral(),
		Tools: []llm.Tool{
			{Name: "a", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Name: "b", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Name: "c", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tools := body["tools"].([]any)
	if len(tools) != 3 {
		t.Fatalf("tools len=%d", len(tools))
	}
	if _, has := tools[0].(map[string]any)["cache_control"]; has {
		t.Errorf("tools[0] should NOT have cache_control (shortcut marks only the last)")
	}
	if _, has := tools[1].(map[string]any)["cache_control"]; has {
		t.Errorf("tools[1] should NOT have cache_control (shortcut marks only the last)")
	}
	if tools[2].(map[string]any)["cache_control"] == nil {
		t.Errorf("tools[2] (last) missing cache_control from shortcut")
	}
}

func TestCacheControl_PerToolWinsOverShortcut(t *testing.T) {
	// When both Tool.CacheControl AND Request.ToolsCacheControl could apply
	// to the same (last) tool, the per-tool field already-set wins — the
	// shortcut does not overwrite.
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:             anthropic.ClaudeSonnet4_6,
		MaxTokens:         64,
		ToolsCacheControl: &llm.CacheControl{Type: "ephemeral", TTL: "1h"}, // shortcut wants 1h
		Tools: []llm.Tool{
			{Name: "a", InputSchema: json.RawMessage(`{"type":"object"}`),
				CacheControl: &llm.CacheControl{Type: "ephemeral"}}, // per-tool wants default
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tools := body["tools"].([]any)
	cc := tools[0].(map[string]any)["cache_control"].(map[string]any)
	if _, hasTTL := cc["ttl"]; hasTTL {
		t.Errorf("per-tool CacheControl (no TTL) should win over shortcut (1h); got cc=%v", cc)
	}
}
