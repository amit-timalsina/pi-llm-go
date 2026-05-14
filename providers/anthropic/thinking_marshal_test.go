package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// TestConvertOutgoingBlock_ThinkingEmptyContentKeepsThinkingField pins
// the wire contract: a ThinkingBlock with empty Thinking text but a
// signed continuation token must still serialize the `thinking` field.
// Anthropic's content-block validator returns 400 with path
// `messages.N.content.M.thinking.thinking: Field required` when the
// field is absent (reported 2026-05-14 on a multi-iteration Opus
// 4.7 agent run with adaptive thinking enabled).
func TestConvertOutgoingBlock_ThinkingEmptyContentKeepsThinkingField(t *testing.T) {
	b, err := convertOutgoingBlock(llm.ThinkingBlock{Thinking: "", Signature: "sig-continuation-token"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	encoded, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(encoded)
	if !strings.Contains(got, `"thinking":""`) {
		t.Errorf("expected `\"thinking\":\"\"` to appear on a thinking-type block even with empty content; got %s", got)
	}
	if !strings.Contains(got, `"signature":"sig-continuation-token"`) {
		t.Errorf("signature dropped: %s", got)
	}
	if !strings.Contains(got, `"type":"thinking"`) {
		t.Errorf("type discriminator dropped: %s", got)
	}
}

// TestConvertOutgoingBlock_ThinkingPopulatedRoundTrips is the parallel
// happy-path case: non-empty thinking text + signature serialize fully.
func TestConvertOutgoingBlock_ThinkingPopulatedRoundTrips(t *testing.T) {
	b, err := convertOutgoingBlock(llm.ThinkingBlock{Thinking: "the model considered...", Signature: "sig-123"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	encoded, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(encoded)
	for _, want := range []string{
		`"type":"thinking"`,
		`"thinking":"the model considered..."`,
		`"signature":"sig-123"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in serialized output: %s", want, got)
		}
	}
}

// TestConvertOutgoingBlock_TextBlockUnaffectedByThinkingMarshalCarveOut
// pins the regression boundary: the MarshalJSON carve-out for the
// "thinking" type must NOT bleed `"thinking":""` into non-thinking
// blocks. Anthropic's text-block validator rejects unknown fields.
func TestConvertOutgoingBlock_TextBlockUnaffectedByThinkingMarshalCarveOut(t *testing.T) {
	b, err := convertOutgoingBlock(llm.TextBlock{Text: "hello"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	encoded, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(encoded)
	if strings.Contains(got, `"thinking"`) {
		t.Errorf("text block leaked a `thinking` field: %s", got)
	}
	if strings.Contains(got, `"signature"`) {
		t.Errorf("text block leaked a `signature` field: %s", got)
	}
	if !strings.Contains(got, `"text":"hello"`) {
		t.Errorf("text field missing: %s", got)
	}
}

// TestConvertOutgoingBlock_ToolUseUnaffectedByThinkingMarshalCarveOut
// is the same regression boundary for tool_use blocks — they must
// continue to serialize id + name + input only.
func TestConvertOutgoingBlock_ToolUseUnaffectedByThinkingMarshalCarveOut(t *testing.T) {
	b, err := convertOutgoingBlock(llm.ToolCallBlock{
		ID:        "tu_1",
		Name:      "echo",
		Arguments: json.RawMessage(`{"text":"x"}`),
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	encoded, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(encoded)
	if strings.Contains(got, `"thinking"`) || strings.Contains(got, `"signature"`) {
		t.Errorf("tool_use block leaked thinking fields: %s", got)
	}
	for _, want := range []string{`"type":"tool_use"`, `"id":"tu_1"`, `"name":"echo"`} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in serialized output: %s", want, got)
		}
	}
}
