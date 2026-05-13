package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// TestThinking_AdaptiveShapeWhenEffortSet verifies that setting
// ThinkingConfig.Effort emits the adaptive on-wire shape:
//
//	"thinking": {"type": "adaptive"}
//	"output_config": {"effort": "<level>"}
//
// AND that budget_tokens is OMITTED from the request — Opus 4.7
// returns 400 if budget_tokens appears alongside type=adaptive.
// Closes issue #20.
func TestThinking_AdaptiveShapeWhenEffortSet(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeOpus4_7,
		MaxTokens: 4096,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		Thinking:  &llm.ThinkingConfig{Effort: llm.EffortHigh},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("missing thinking field: %v", body)
	}
	if thinking["type"] != "adaptive" {
		t.Errorf("thinking.type: got %v, want adaptive", thinking["type"])
	}
	if _, present := thinking["budget_tokens"]; present {
		t.Errorf("thinking.budget_tokens present in adaptive request — Opus 4.7 will 400. got: %v", thinking)
	}

	oc, ok := body["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("missing output_config field: %v", body)
	}
	if oc["effort"] != "high" {
		t.Errorf("output_config.effort: got %v, want high", oc["effort"])
	}
}

// TestThinking_ManualShapeWhenOnlyBudgetTokensSet verifies that the
// legacy manual shape still works (Opus 4.5 and older models require
// it). budget_tokens lives INSIDE thinking; no output_config is sent.
func TestThinking_ManualShapeWhenOnlyBudgetTokensSet(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 4096,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		Thinking:  &llm.ThinkingConfig{BudgetTokens: 2048},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}

	thinking, ok := body["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("missing thinking field: %v", body)
	}
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type: got %v, want enabled", thinking["type"])
	}
	if thinking["budget_tokens"].(float64) != 2048 {
		t.Errorf("thinking.budget_tokens: got %v, want 2048", thinking["budget_tokens"])
	}
	if _, present := body["output_config"]; present {
		t.Errorf("output_config should NOT be present on manual-thinking request: %v", body)
	}
}

// TestThinking_EffortWinsWhenBothSet verifies the documented dispatch:
// when a caller sets both Effort AND BudgetTokens (e.g. during a
// migration where they're not sure which model they'll hit), the
// adaptive shape wins. BudgetTokens is silently dropped from the wire.
func TestThinking_EffortWinsWhenBothSet(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeOpus4_7,
		MaxTokens: 4096,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		Thinking: &llm.ThinkingConfig{
			Effort:       llm.EffortMedium,
			BudgetTokens: 9999, // should NOT appear on wire
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	thinking, _ := body["thinking"].(map[string]any)
	if _, present := thinking["budget_tokens"]; present {
		t.Errorf("budget_tokens leaked into adaptive request: %v", thinking)
	}
	oc, _ := body["output_config"].(map[string]any)
	if oc["effort"] != "medium" {
		t.Errorf("output_config.effort: got %v, want medium", oc["effort"])
	}
}

// TestThinking_NilConfigOmitsBothFields verifies that no thinking
// fields are sent when Request.Thinking is nil (the no-thinking case).
func TestThinking_NilConfigOmitsBothFields(t *testing.T) {
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

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if _, present := body["thinking"]; present {
		t.Errorf("thinking should be absent when ThinkingConfig is nil: %v", body)
	}
	if _, present := body["output_config"]; present {
		t.Errorf("output_config should be absent when ThinkingConfig is nil: %v", body)
	}
}

// TestThinking_EmptyConfigOmitsBothFields verifies that a non-nil but
// empty ThinkingConfig (zero-value Effort and BudgetTokens) ALSO
// produces no wire fields. The dispatch falls through both branches.
// Important: prevents a caller-side mistake (passing &ThinkingConfig{}
// to "explicitly disable") from leaking a malformed request.
func TestThinking_EmptyConfigOmitsBothFields(t *testing.T) {
	t.Parallel()

	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 1024,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		Thinking:  &llm.ThinkingConfig{}, // both fields zero
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	if _, present := body["thinking"]; present {
		t.Errorf("thinking should be absent for empty ThinkingConfig: %v", body)
	}
	if _, present := body["output_config"]; present {
		t.Errorf("output_config should be absent for empty ThinkingConfig: %v", body)
	}
}
