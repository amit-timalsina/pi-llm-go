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

// Reuses textOnlyPayload and fakeServer from anthropic_test.go.

// TestCacheRetention_NonePlacesNoMarkers verifies the zero-value contract:
// without an explicit CacheRetention, the wire format carries no
// cache_control fields and no extended-TTL beta header.
func TestCacheRetention_NonePlacesNoMarkers(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		System:    "stable system",
		Tools: []llm.Tool{
			{Name: "a", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	// System should be plain string.
	if _, ok := body["system"].(string); !ok {
		t.Errorf("system=%T, want plain string when CacheRetention unset", body["system"])
	}
	// No cache_control on any tool.
	for _, raw := range body["tools"].([]any) {
		if _, has := raw.(map[string]any)["cache_control"]; has {
			t.Errorf("tool should have no cache_control: %+v", raw)
		}
	}
	// No cache_control on user message blocks.
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	for _, raw := range content {
		if _, has := raw.(map[string]any)["cache_control"]; has {
			t.Errorf("user block should have no cache_control: %+v", raw)
		}
	}
	for _, b := range fs.lastHeaders.Values("Anthropic-Beta") {
		if b == "extended-cache-ttl-2025-04-11" {
			t.Errorf("extended-cache-ttl beta header should NOT be sent when CacheRetention unset")
		}
	}
}

// TestCacheRetention_ShortPlacesMarkersAtPrefix verifies that "short"
// retention auto-attaches ephemeral markers at the three prefix-boundary
// points (system, last tool, last user text block) — with no TTL field
// and no beta header.
func TestCacheRetention_ShortPlacesMarkersAtPrefix(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:          anthropic.ClaudeSonnet4_6,
		MaxTokens:      64,
		CacheRetention: llm.CacheRetentionShort,
		System:         "stable system",
		Tools: []llm.Tool{
			{Name: "a", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Name: "b", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "stable preamble"},
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

	// System: structured single-block array with cache_control.
	sysArr, ok := body["system"].([]any)
	if !ok || len(sysArr) != 1 {
		t.Fatalf("system=%v, want 1-element array under short retention", body["system"])
	}
	sysBlk := sysArr[0].(map[string]any)
	sysCC, ok := sysBlk["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("system block missing cache_control: %+v", sysBlk)
	}
	if sysCC["type"] != "ephemeral" {
		t.Errorf("system cache_control.type=%v, want ephemeral", sysCC["type"])
	}
	if _, hasTTL := sysCC["ttl"]; hasTTL {
		t.Errorf("system cache_control.ttl should be absent for short retention; got %v", sysCC["ttl"])
	}

	// Tools: only the last tool carries cache_control.
	tools := body["tools"].([]any)
	if _, has := tools[0].(map[string]any)["cache_control"]; has {
		t.Errorf("tools[0] should NOT have cache_control")
	}
	if tools[len(tools)-1].(map[string]any)["cache_control"] == nil {
		t.Errorf("last tool missing cache_control")
	}

	// User message: only the LAST text block carries cache_control.
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if _, has := content[0].(map[string]any)["cache_control"]; has {
		t.Errorf("first user text block should NOT have cache_control")
	}
	if content[len(content)-1].(map[string]any)["cache_control"] == nil {
		t.Errorf("last user text block missing cache_control")
	}

	// No extended-TTL beta header for short.
	for _, b := range fs.lastHeaders.Values("Anthropic-Beta") {
		if b == "extended-cache-ttl-2025-04-11" {
			t.Errorf("extended-cache-ttl beta header should NOT be sent for short retention")
		}
	}
}

// TestCacheRetention_LongAddsTTLAndBetaHeader verifies "long" retention
// emits TTL "1h" and auto-attaches the extended-cache-ttl beta header.
func TestCacheRetention_LongAddsTTLAndBetaHeader(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:          anthropic.ClaudeSonnet4_6,
		MaxTokens:      64,
		CacheRetention: llm.CacheRetentionLong,
		System:         "long-lived prefix",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
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

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	sysBlk := body["system"].([]any)[0].(map[string]any)
	cc := sysBlk["cache_control"].(map[string]any)
	if cc["ttl"] != "1h" {
		t.Errorf("system cache_control.ttl=%v, want 1h", cc["ttl"])
	}
}

// TestCacheRetention_UserMarkerLandsOnTrailingToolResult verifies the
// placement algorithm marks the LAST block of the most recent user-role
// message, even when that block is a tool_result (the trailing block
// after a tool round-trip). This matches Mario Zechner's pi-ai design
// and keeps the full tool round-trip inside the cached prefix on the
// next call.
func TestCacheRetention_UserMarkerLandsOnTrailingToolResult(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:          anthropic.ClaudeSonnet4_6,
		MaxTokens:      64,
		CacheRetention: llm.CacheRetentionShort,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "earlier question"}}},
			{Role: llm.RoleAssistant, Content: []llm.Block{
				llm.ToolCallBlock{ID: "t1", Name: "echo", Arguments: json.RawMessage(`{}`)},
			}},
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: "t1", Content: "ok"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	msgs := body["messages"].([]any)

	// Last wire message is the user-role tool_result; the marker lands
	// on the tool_result block (its only block, also the trailing block).
	last := msgs[len(msgs)-1].(map[string]any)
	if last["role"] != "user" {
		t.Fatalf("expected last wire message to be user-role tool_result; got role=%v", last["role"])
	}
	lastBlocks := last["content"].([]any)
	trailing := lastBlocks[len(lastBlocks)-1].(map[string]any)
	if trailing["type"] != "tool_result" {
		t.Fatalf("expected trailing block to be tool_result; got type=%v", trailing["type"])
	}
	cc, ok := trailing["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("tool_result block missing cache_control: %+v", trailing)
	}
	if cc["type"] != "ephemeral" {
		t.Errorf("tool_result cache_control.type=%v, want ephemeral", cc["type"])
	}

	// Earlier user-role text block must NOT carry it (only the trailing
	// block of the most recent user-role message gets marked).
	first := msgs[0].(map[string]any)
	if first["role"] != "user" {
		t.Fatalf("expected first wire message to be user; got %v", first["role"])
	}
	firstContent := first["content"].([]any)
	if _, has := firstContent[0].(map[string]any)["cache_control"]; has {
		t.Errorf("earlier user text block should NOT have cache_control")
	}
}

// TestCacheRetention_ExplicitNoneEquivalentToZero verifies that an
// explicit CacheRetentionNone produces byte-identical wire output to
// the zero value — i.e. callers can compare via == llm.CacheRetentionNone
// regardless of whether the field was left unset.
func TestCacheRetention_ExplicitNoneEquivalentToZero(t *testing.T) {
	if llm.CacheRetentionNone != "" {
		t.Fatalf("CacheRetentionNone must be the zero value of CacheRetention; got %q", string(llm.CacheRetentionNone))
	}

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:          anthropic.ClaudeSonnet4_6,
		MaxTokens:      64,
		CacheRetention: llm.CacheRetentionNone, // explicit
		System:         "stable system",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if strings.Contains(string(fs.lastBody), "cache_control") {
		t.Errorf("explicit CacheRetentionNone unexpectedly emitted cache_control: %s", string(fs.lastBody))
	}
}
