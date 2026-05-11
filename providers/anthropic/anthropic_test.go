package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/amittimalsina/pi-llm-go"
	"github.com/amittimalsina/pi-llm-go/providers/anthropic"
)

// fakeServer serves a canned SSE stream over httptest. payload is the raw
// SSE body to ship after a 200 OK; statusCode and statusBody let tests
// simulate provider errors instead.
type fakeServer struct {
	payload    string
	statusCode int
	statusBody string

	// lastBody captures the JSON body the test posted, for assertions.
	lastBody json.RawMessage
}

func (f *fakeServer) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		f.lastBody = json.RawMessage(body)
		if f.statusCode != 0 && f.statusCode != http.StatusOK {
			w.WriteHeader(f.statusCode)
			_, _ = io.WriteString(w, f.statusBody)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, f.payload)
	}
}

func newProvider(t *testing.T, srv *httptest.Server) *anthropic.Provider {
	t.Helper()
	p, err := anthropic.New(anthropic.Options{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// Canned SSE: text-only response. Mirrors actual Anthropic frame ordering.
const textOnlyPayload = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-4-7","usage":{"input_tokens":7,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

`

func TestStreamTextOnly(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	var events []llm.StreamEvent
	for ev, err := range p.Stream(context.Background(), llm.Request{
		Model: anthropic.ClaudeOpus4_7,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
		MaxTokens: 32,
	}) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		events = append(events, ev)
	}

	wantTypes := []string{
		"anthropic.EventMessageStart",
		"anthropic.EventTextStart",
		"anthropic.EventTextDelta",
		"anthropic.EventTextDelta",
		"anthropic.EventTextEnd",
		"anthropic.EventMessageEnd",
	}
	if got := typeNames(events); !equalTypeNames(got, wantTypes) {
		t.Fatalf("event type sequence wrong:\ngot:  %v\nwant: %v", got, wantTypes)
	}

	// Verify the final event carries normalized stop reason + usage.
	end := events[len(events)-1].(llm.EventMessageEnd)
	if end.StopReason != llm.StopReasonEnd {
		t.Errorf("StopReason=%v, want %v", end.StopReason, llm.StopReasonEnd)
	}
	if end.Usage.InputTokens != 7 || end.Usage.OutputTokens != 4 || end.Usage.TotalTokens != 11 {
		t.Errorf("Usage wrong: %+v", end.Usage)
	}

	// And spot-check the assembled deltas.
	var assembled strings.Builder
	for _, ev := range events {
		if d, ok := ev.(llm.EventTextDelta); ok {
			assembled.WriteString(d.Delta)
		}
	}
	if got := assembled.String(); got != "Hello!" {
		t.Errorf("assembled text = %q, want %q", got, "Hello!")
	}
}

const toolCallPayload = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-4-7","usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01abc","name":"add","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"137,\"b\":84}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}

event: message_stop
data: {"type":"message_stop"}

`

func TestStreamToolCall(t *testing.T) {
	fs := &fakeServer{payload: toolCallPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	final, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeOpus4_7,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "add 137 and 84"}}}},
		Tools:     []llm.Tool{{Name: "add", Description: "add two ints", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if final.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason=%v, want %v", final.StopReason, llm.StopReasonToolUse)
	}
	if len(final.Content) != 1 {
		t.Fatalf("Content len=%d, want 1", len(final.Content))
	}
	tc, ok := final.Content[0].(llm.ToolCallBlock)
	if !ok {
		t.Fatalf("Content[0] type=%T, want ToolCallBlock", final.Content[0])
	}
	if tc.ID != "toolu_01abc" || tc.Name != "add" {
		t.Errorf("ToolCallBlock fields wrong: %+v", tc)
	}
	var args struct{ A, B int }
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("Unmarshal Arguments: %v (raw=%q)", err, string(tc.Arguments))
	}
	if args.A != 137 || args.B != 84 {
		t.Errorf("Arguments parsed wrong: %+v", args)
	}
}

func TestStreamRequestBodyShape(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	temp := 0.5
	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:       anthropic.ClaudeOpus4_7,
		System:      "you are terse",
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		Temperature: &temp,
		MaxTokens:   1024,
		Thinking:    &llm.ThinkingConfig{BudgetTokens: 2048},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode last body: %v", err)
	}
	if body["model"] != anthropic.ClaudeOpus4_7 {
		t.Errorf("model=%v", body["model"])
	}
	if body["system"] != "you are terse" {
		t.Errorf("system=%v", body["system"])
	}
	if body["max_tokens"].(float64) != 1024 {
		t.Errorf("max_tokens=%v", body["max_tokens"])
	}
	if body["temperature"].(float64) != 0.5 {
		t.Errorf("temperature=%v", body["temperature"])
	}
	if body["stream"] != true {
		t.Errorf("stream=%v, want true", body["stream"])
	}
	thinking, ok := body["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" || thinking["budget_tokens"].(float64) != 2048 {
		t.Errorf("thinking shape wrong: %v", body["thinking"])
	}
}

func TestStreamHTTPError(t *testing.T) {
	fs := &fakeServer{statusCode: http.StatusUnauthorized, statusBody: `{"error":"bad key"}`}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	var gotErr error
	for _, err := range p.Stream(context.Background(), llm.Request{
		Model:     anthropic.ClaudeOpus4_7,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		MaxTokens: 16,
	}) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Fatal("want error, got nil")
	}
	if !errors.Is(gotErr, llm.ErrAuth) {
		t.Errorf("err=%v, want wrapping ErrAuth", gotErr)
	}
	var apiErr *llm.APIError
	if !errors.As(gotErr, &apiErr) || apiErr.Status != http.StatusUnauthorized {
		t.Errorf("APIError extraction failed: %v", gotErr)
	}
	if !strings.Contains(string(apiErr.Body), "bad key") {
		t.Errorf("APIError body lost: %q", string(apiErr.Body))
	}
}

func TestStreamContextCancellation(t *testing.T) {
	// Server that holds the connection open until the client disconnects.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, "event: ping\ndata: {}\n\n")
		flusher.Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	p := newProvider(t, srv)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	var sawErr error
	for _, err := range p.Stream(ctx, llm.Request{
		Model:     anthropic.ClaudeOpus4_7,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		MaxTokens: 16,
	}) {
		if err != nil {
			sawErr = err
		}
	}
	if sawErr == nil {
		t.Fatal("expected cancellation error, got nil")
	}
	// The cancellation surfaces either as ctx.Err() or via an http transport
	// "context deadline exceeded" / "context canceled" wrap.
	if !errors.Is(sawErr, context.Canceled) && !errors.Is(sawErr, context.DeadlineExceeded) {
		// Some transports wrap with their own error; tolerate as long as we
		// got something terminal.
		if !strings.Contains(sawErr.Error(), "context") {
			t.Errorf("expected context-related error, got %v", sawErr)
		}
	}
}

func typeNames(events []llm.StreamEvent) []string {
	out := make([]string, len(events))
	for i, ev := range events {
		// Produce "<package>.<TypeName>" for readability.
		t := reflectTypeName(ev)
		out[i] = "anthropic." + t
	}
	return out
}

func reflectTypeName(v any) string {
	switch v.(type) {
	case llm.EventMessageStart:
		return "EventMessageStart"
	case llm.EventTextStart:
		return "EventTextStart"
	case llm.EventTextDelta:
		return "EventTextDelta"
	case llm.EventTextEnd:
		return "EventTextEnd"
	case llm.EventThinkingStart:
		return "EventThinkingStart"
	case llm.EventThinkingDelta:
		return "EventThinkingDelta"
	case llm.EventThinkingEnd:
		return "EventThinkingEnd"
	case llm.EventToolCallStart:
		return "EventToolCallStart"
	case llm.EventToolCallDelta:
		return "EventToolCallDelta"
	case llm.EventToolCallEnd:
		return "EventToolCallEnd"
	case llm.EventMessageEnd:
		return "EventMessageEnd"
	default:
		return "Unknown"
	}
}

func equalTypeNames(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
