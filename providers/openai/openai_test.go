package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amittimalsina/pi-llm-go"
	"github.com/amittimalsina/pi-llm-go/providers/openai"
)

type fakeServer struct {
	payload    string
	statusCode int
	statusBody string
	lastBody   json.RawMessage
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

func newProvider(t *testing.T, srv *httptest.Server) *openai.Provider {
	t.Helper()
	p, err := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

const textOnlyPayload = `data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{"content":"lo!"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}

data: [DONE]

`

func TestStreamTextOnly(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	final, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    openai.GPT5_5,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if final.Model != "gpt-5.5" {
		t.Errorf("Model=%q", final.Model)
	}
	if final.StopReason != llm.StopReasonEnd {
		t.Errorf("StopReason=%v", final.StopReason)
	}
	if final.Usage.InputTokens != 7 || final.Usage.OutputTokens != 4 {
		t.Errorf("Usage=%+v", final.Usage)
	}
	if len(final.Content) != 1 {
		t.Fatalf("Content len=%d, want 1", len(final.Content))
	}
	tb, ok := final.Content[0].(llm.TextBlock)
	if !ok {
		t.Fatalf("Content[0] type=%T", final.Content[0])
	}
	if tb.Text != "Hello!" {
		t.Errorf("Text=%q, want %q", tb.Text, "Hello!")
	}
}

const toolCallPayload = `data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"add","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"137,\"b\":84}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-1","model":"gpt-5.5","choices":[],"usage":{"prompt_tokens":18,"completion_tokens":9,"total_tokens":27}}

data: [DONE]

`

func TestStreamToolCall(t *testing.T) {
	fs := &fakeServer{payload: toolCallPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	final, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    openai.GPT5_5,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "add 137 and 84"}}}},
		Tools:    []llm.Tool{{Name: "add", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if final.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason=%v", final.StopReason)
	}
	if len(final.Content) != 1 {
		t.Fatalf("Content len=%d", len(final.Content))
	}
	tc, ok := final.Content[0].(llm.ToolCallBlock)
	if !ok {
		t.Fatalf("Content[0] type=%T", final.Content[0])
	}
	if tc.ID != "call_abc" || tc.Name != "add" {
		t.Errorf("ToolCallBlock fields wrong: %+v", tc)
	}
	var args struct{ A, B int }
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("Unmarshal Arguments: %v (raw=%q)", err, string(tc.Arguments))
	}
	if args.A != 137 || args.B != 84 {
		t.Errorf("Arguments wrong: %+v", args)
	}
}

func TestStreamBodyShape(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	temp := 0.3
	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    openai.GPT5_5,
		System:   "you are helpful",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
		Tools: []llm.Tool{
			{Name: "noop", Description: "does nothing", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Temperature: &temp,
		MaxTokens:   256,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	msgs := body["messages"].([]any)
	if msgs[0].(map[string]any)["role"] != "system" {
		t.Errorf("first message not system: %v", msgs[0])
	}
	if msgs[1].(map[string]any)["role"] != "user" {
		t.Errorf("second message not user: %v", msgs[1])
	}
	tools := body["tools"].([]any)
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "noop" {
		t.Errorf("tool function name wrong: %v", fn)
	}
	streamOpts := body["stream_options"].(map[string]any)
	if streamOpts["include_usage"] != true {
		t.Errorf("include_usage missing")
	}
}

func TestStreamToolResultRoundTrip(t *testing.T) {
	// Verifies that a llm.Message with RoleTool and multiple ToolResultBlocks
	// expands into multiple OpenAI tool messages, one per block.
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: openai.GPT5_5,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
			{Role: llm.RoleAssistant, Content: []llm.Block{
				llm.ToolCallBlock{ID: "call_a", Name: "x", Arguments: json.RawMessage(`{}`)},
				llm.ToolCallBlock{ID: "call_b", Name: "y", Arguments: json.RawMessage(`{}`)},
			}},
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: "call_a", Content: "a-result"},
				llm.ToolResultBlock{ToolCallID: "call_b", Content: "b-result"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	msgs := body["messages"].([]any)
	// Expect: user, assistant, tool(a), tool(b) = 4 messages.
	if len(msgs) != 4 {
		t.Fatalf("messages len=%d, want 4: %v", len(msgs), msgs)
	}
	if r := msgs[2].(map[string]any)["role"]; r != "tool" {
		t.Errorf("msg[2].role=%v", r)
	}
	if id := msgs[2].(map[string]any)["tool_call_id"]; id != "call_a" {
		t.Errorf("msg[2].tool_call_id=%v", id)
	}
	if c := msgs[2].(map[string]any)["content"]; c != "a-result" {
		t.Errorf("msg[2].content=%v", c)
	}
}

func TestStreamHTTPError(t *testing.T) {
	fs := &fakeServer{statusCode: http.StatusTooManyRequests, statusBody: `{"error":{"message":"rate limited"}}`}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	var gotErr error
	for _, err := range p.Stream(context.Background(), llm.Request{
		Model:    openai.GPT5_5,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}) {
		if err != nil {
			gotErr = err
		}
	}
	if !errors.Is(gotErr, llm.ErrRateLimit) {
		t.Errorf("err=%v, want wrapping ErrRateLimit", gotErr)
	}
	var apiErr *llm.APIError
	if !errors.As(gotErr, &apiErr) || !strings.Contains(string(apiErr.Body), "rate limited") {
		t.Errorf("APIError body lost: %v", gotErr)
	}
}
