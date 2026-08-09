package gemini_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/gemini"
)

const signedToolCallPayload = `data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call_signed","name":"lookup","args":{"q":"x"}},"thoughtSignature":"tool-signature"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":4,"totalTokenCount":16}}

`

const signedStreamingTextPayload = `data: {"candidates":[{"content":{"parts":[{"text":"final answer"}],"role":"model"},"index":0}]}

data: {"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"text-signature"}],"role":"model"},"index":0}]}

data: {"candidates":[{"content":{"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}}

`

const signedGemini25ThoughtPayload = `data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"checking the request","thoughtSignature":"gemini-25-signature"},{"functionCall":{"name":"lookup","args":{"q":"x"}}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":4,"totalTokenCount":20,"thoughtsTokenCount":4}}

`

func TestThoughtSignature_FunctionCallCaptureAndReplay(t *testing.T) {
	fs := &fakeServer{payload: signedToolCallPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	user := llm.Message{Role: llm.RoleUser, Content: []llm.Block{
		llm.TextBlock{Text: "look up x"},
	}}
	tool := llm.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}
	assistant, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    gemini.Gemini3_1ProPreview,
		Tools:    []llm.Tool{tool},
		Messages: []llm.Message{user},
	})
	if err != nil {
		t.Fatalf("first Complete: %v", err)
	}

	call, ok := assistant.Content[0].(llm.ToolCallBlock)
	if !ok {
		t.Fatalf("assistant.Content[0]=%T, want ToolCallBlock", assistant.Content[0])
	}
	if call.Signature != "tool-signature" {
		t.Fatalf("ToolCallBlock.Signature=%q, want tool-signature", call.Signature)
	}

	_, err = llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini3_1ProPreview,
		Tools: []llm.Tool{tool},
		Messages: []llm.Message{
			user,
			*assistant,
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: call.ID, Content: "found"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("second Complete: %v", err)
	}

	parts := capturedParts(t, fs.lastBody, 1)
	if got := parts[0]["thoughtSignature"]; got != "tool-signature" {
		t.Fatalf("replayed thoughtSignature=%v, want tool-signature", got)
	}
	if _, ok := parts[0]["functionCall"].(map[string]any); !ok {
		t.Fatalf("signed part lost functionCall: %#v", parts[0])
	}
}

func TestThoughtSignature_FinalEmptyTextChunkCaptureAndReplay(t *testing.T) {
	fs := &fakeServer{payload: signedStreamingTextPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	user := llm.Message{Role: llm.RoleUser, Content: []llm.Block{
		llm.TextBlock{Text: "answer"},
	}}
	assistant, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    gemini.Gemini3_1ProPreview,
		Messages: []llm.Message{user},
	})
	if err != nil {
		t.Fatalf("first Complete: %v", err)
	}
	if len(assistant.Content) != 2 {
		t.Fatalf("assistant.Content len=%d, want prose plus signed empty part", len(assistant.Content))
	}
	text, ok := assistant.Content[0].(llm.TextBlock)
	if !ok {
		t.Fatalf("assistant.Content[0]=%T, want TextBlock", assistant.Content[0])
	}
	if text.Text != "final answer" || text.Signature != "" {
		t.Fatalf("first TextBlock=%+v, want unsigned final answer", text)
	}
	signed, ok := assistant.Content[1].(llm.TextBlock)
	if !ok {
		t.Fatalf("assistant.Content[1]=%T, want TextBlock", assistant.Content[1])
	}
	if signed.Text != "" || signed.Signature != "text-signature" {
		t.Fatalf("second TextBlock=%+v, want signed empty part", signed)
	}

	_, err = llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini3_1ProPreview,
		Messages: []llm.Message{
			user,
			*assistant,
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "continue"}}},
		},
	})
	if err != nil {
		t.Fatalf("second Complete: %v", err)
	}

	parts := capturedParts(t, fs.lastBody, 1)
	if len(parts) != 2 {
		t.Fatalf("replayed model parts=%d, want 2", len(parts))
	}
	if got := parts[0]["text"]; got != "final answer" {
		t.Fatalf("replayed text=%v, want final answer", got)
	}
	if _, present := parts[0]["thoughtSignature"]; present {
		t.Fatalf("unsigned prose part gained a signature: %#v", parts[0])
	}
	if got := parts[1]["text"]; got != "" {
		t.Fatalf("signed part text=%v, want empty", got)
	}
	if got := parts[1]["thoughtSignature"]; got != "text-signature" {
		t.Fatalf("replayed thoughtSignature=%v, want text-signature", got)
	}
}

func TestThoughtSignature_Gemini25FirstThoughtPartCaptureAndReplay(t *testing.T) {
	fs := &fakeServer{payload: signedGemini25ThoughtPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	user := llm.Message{Role: llm.RoleUser, Content: []llm.Block{
		llm.TextBlock{Text: "look up x"},
	}}
	tool := llm.Tool{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}
	assistant, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    gemini.Gemini2_5Pro,
		Tools:    []llm.Tool{tool},
		Messages: []llm.Message{user},
		Thinking: &llm.ThinkingConfig{BudgetTokens: 1024},
	})
	if err != nil {
		t.Fatalf("first Complete: %v", err)
	}
	if len(assistant.Content) != 2 {
		t.Fatalf("assistant.Content len=%d, want thought plus tool call", len(assistant.Content))
	}
	thought, ok := assistant.Content[0].(llm.ThinkingBlock)
	if !ok {
		t.Fatalf("assistant.Content[0]=%T, want ThinkingBlock", assistant.Content[0])
	}
	if thought.Thinking != "checking the request" || thought.Signature != "gemini-25-signature" {
		t.Fatalf("ThinkingBlock=%+v, want signed Gemini 2.5 first part", thought)
	}
	call, ok := assistant.Content[1].(llm.ToolCallBlock)
	if !ok {
		t.Fatalf("assistant.Content[1]=%T, want ToolCallBlock", assistant.Content[1])
	}
	if call.Signature != "" {
		t.Fatalf("second part signature=%q, want empty", call.Signature)
	}

	_, err = llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Pro,
		Tools: []llm.Tool{tool},
		Messages: []llm.Message{
			user,
			*assistant,
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: call.ID, Content: "found"},
			}},
		},
		Thinking: &llm.ThinkingConfig{BudgetTokens: 1024},
	})
	if err != nil {
		t.Fatalf("second Complete: %v", err)
	}

	parts := capturedParts(t, fs.lastBody, 1)
	if len(parts) != 2 {
		t.Fatalf("replayed model parts=%d, want 2", len(parts))
	}
	if got := parts[0]["thoughtSignature"]; got != "gemini-25-signature" {
		t.Fatalf("first-part thoughtSignature=%v, want gemini-25-signature", got)
	}
	if got := parts[0]["thought"]; got != true {
		t.Fatalf("first-part thought=%v, want true", got)
	}
	if _, ok := parts[1]["thoughtSignature"]; ok {
		t.Fatalf("unsigned function-call part gained a signature: %#v", parts[1])
	}
}

func TestThoughtSignature_EmptyTextPartRemainsWireTextPart(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini3_1ProPreview,
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: []llm.Block{
				llm.TextBlock{Text: "", Signature: "signature-only"},
			}},
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "continue"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	part := capturedParts(t, fs.lastBody, 0)[0]
	text, present := part["text"]
	if !present || text != "" {
		t.Fatalf("empty text part was not preserved: %#v", part)
	}
	if got := part["thoughtSignature"]; got != "signature-only" {
		t.Fatalf("thoughtSignature=%v, want signature-only", got)
	}
}

func capturedParts(t *testing.T, body json.RawMessage, contentIndex int) []map[string]any {
	t.Helper()
	var decoded struct {
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if contentIndex >= len(decoded.Contents) {
		t.Fatalf("content index %d out of range for %d contents", contentIndex, len(decoded.Contents))
	}
	return decoded.Contents[contentIndex].Parts
}
