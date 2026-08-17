package openai_responses_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	openai_responses "github.com/amit-timalsina/pi-llm-go/providers/openai_responses"
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

func newProvider(t *testing.T, srv *httptest.Server) *openai_responses.Provider {
	t.Helper()
	p, err := openai_responses.New(openai_responses.Options{
		APIKey:  "test",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// textOnlyPayload mimics Responses API streaming a single text item.
const textOnlyPayload = `event: response.created
data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hel"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"lo!"}

event: response.output_text.done
data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello!"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"completed","usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}}

`

func TestStreamTextOnly(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	final, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.4-mini",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if final.Model != "gpt-5.4-mini" {
		t.Errorf("Model=%q", final.Model)
	}
	if final.StopReason != llm.StopReasonEnd {
		t.Errorf("StopReason=%v", final.StopReason)
	}
	if final.Usage.InputTokens != 7 || final.Usage.OutputTokens != 3 || final.Usage.TotalTokens != 10 {
		t.Errorf("Usage=%+v", final.Usage)
	}
	if len(final.Content) != 1 {
		t.Fatalf("Content len=%d", len(final.Content))
	}
	tb, ok := final.Content[0].(llm.TextBlock)
	if !ok {
		t.Fatalf("Content[0] type=%T", final.Content[0])
	}
	if tb.Text != "Hello!" {
		t.Errorf("Text=%q", tb.Text)
	}
}

// toolCallPayload: response emits a function_call item then completed.
const toolCallPayload = `event: response.created
data: {"type":"response.created","response":{"id":"resp_2","model":"gpt-5.4-mini","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"add","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"137,\"b\":84}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"a\":137,\"b\":84}"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_2","model":"gpt-5.4-mini","status":"completed","usage":{"input_tokens":18,"output_tokens":9,"total_tokens":27}}}

`

func TestStreamToolCall(t *testing.T) {
	fs := &fakeServer{payload: toolCallPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	final, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.4-mini",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "add 137 and 84"}}}},
		Tools:    []llm.Tool{{Name: "add", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if final.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason=%v, want ToolUse", final.StopReason)
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

// reasoningSummaryPayload: response emits a reasoning summary then text.
const reasoningSummaryPayload = `event: response.created
data: {"type":"response.created","response":{"id":"resp_3","model":"gpt-5.4-mini","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"Plan: "}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"add the numbers."}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","output_index":0,"text":"Plan: add the numbers."}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_2","status":"in_progress","role":"assistant","content":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":1,"delta":"221"}

event: response.output_text.done
data: {"type":"response.output_text.done","output_index":1,"text":"221"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_3","model":"gpt-5.4-mini","status":"completed","usage":{"input_tokens":12,"output_tokens":15,"total_tokens":27}}}

`

func TestStreamReasoningSummary(t *testing.T) {
	fs := &fakeServer{payload: reasoningSummaryPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	final, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.4-mini",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "137+84?"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(final.Content) != 2 {
		t.Fatalf("Content len=%d, want 2 (thinking + text)", len(final.Content))
	}
	tk, ok := final.Content[0].(llm.ThinkingBlock)
	if !ok {
		t.Fatalf("Content[0] type=%T, want ThinkingBlock", final.Content[0])
	}
	if tk.Thinking != "Plan: add the numbers." {
		t.Errorf("Thinking=%q", tk.Thinking)
	}
	tb, ok := final.Content[1].(llm.TextBlock)
	if !ok {
		t.Fatalf("Content[1] type=%T, want TextBlock", final.Content[1])
	}
	if tb.Text != "221" {
		t.Errorf("Text=%q", tb.Text)
	}
}

func TestStreamRequestBodyShape(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai_responses.New(openai_responses.Options{
		APIKey:                  "test",
		BaseURL:                 srv.URL,
		ReasoningEffort:         openai_responses.ReasoningMedium,
		IncludeReasoningSummary: true,
	})
	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:  "gpt-5.4-mini",
		System: "you are terse",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
		Tools: []llm.Tool{
			{Name: "ping", Description: "ping", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		MaxTokens: 256,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body["model"] != "gpt-5.4-mini" {
		t.Errorf("model=%v", body["model"])
	}
	if body["instructions"] != "you are terse" {
		t.Errorf("instructions=%v", body["instructions"])
	}
	if body["max_output_tokens"].(float64) != 256 {
		t.Errorf("max_output_tokens=%v", body["max_output_tokens"])
	}
	r := body["reasoning"].(map[string]any)
	if r["effort"] != "medium" {
		t.Errorf("reasoning.effort=%v", r["effort"])
	}
	if r["summary"] != "auto" {
		t.Errorf("reasoning.summary=%v", r["summary"])
	}
	// Input items shape.
	input := body["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input len=%d", len(input))
	}
	if input[0].(map[string]any)["role"] != "user" {
		t.Errorf("input[0].role=%v", input[0])
	}
}

// TestStreamToolLoopReplayShape pins the second-turn input array of a tool
// loop: the assistant's tool call must be replayed as a function_call item
// ahead of its function_call_output, or the API rejects the request.
func TestStreamToolLoopReplayShape(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: "gpt-5.4-mini",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "add 137 and 84"}}},
			{Role: llm.RoleAssistant, Content: []llm.Block{
				llm.ThinkingBlock{Thinking: "sum them"},
				llm.TextBlock{Text: "Adding."},
				llm.ToolCallBlock{ID: "call_abc", Name: "add", Arguments: json.RawMessage(`{"a":137,"b":84}`)},
			}},
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: "call_abc", Content: "221"},
			}},
		},
		Tools: []llm.Tool{{Name: "add", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	var types []string
	for _, it := range body.Input {
		types = append(types, it["type"].(string))
	}
	want := []string{"message", "message", "function_call", "function_call_output"}
	if strings.Join(types, ",") != strings.Join(want, ",") {
		t.Fatalf("input types=%v, want %v", types, want)
	}
	call := body.Input[2]
	if call["call_id"] != "call_abc" || call["name"] != "add" {
		t.Errorf("function_call identity wrong: %+v", call)
	}
	if call["arguments"] != `{"a":137,"b":84}` {
		t.Errorf("function_call arguments=%v", call["arguments"])
	}
	if _, ok := call["id"]; ok {
		t.Errorf("function_call must not carry an item id: %+v", call)
	}
	if body.Input[3]["call_id"] != "call_abc" {
		t.Errorf("function_call_output call_id=%v", body.Input[3]["call_id"])
	}
	if body.Input[1]["role"] != "assistant" {
		t.Errorf("assistant text item=%+v", body.Input[1])
	}
}

// TestStreamAssistantToolCallOnly covers an assistant turn with no text: it
// emits the function_call item and no empty message item.
func TestStreamAssistantToolCallOnly(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: "gpt-5.4-mini",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "ping"}}},
			{Role: llm.RoleAssistant, Content: []llm.Block{
				llm.ToolCallBlock{ID: "call_xyz", Name: "ping"},
			}},
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: "call_xyz", Content: "pong"},
			}},
		},
		Tools: []llm.Tool{{Name: "ping", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if len(body.Input) != 3 {
		t.Fatalf("input len=%d, want 3: %+v", len(body.Input), body.Input)
	}
	if body.Input[1]["type"] != "function_call" {
		t.Fatalf("input[1]=%+v", body.Input[1])
	}
	if body.Input[1]["arguments"] != "{}" {
		t.Errorf("empty Arguments must serialize as {}, got %v", body.Input[1]["arguments"])
	}
}

// reasoningUsagePayload carries a live gpt-5.6-sol usage block, where 516 of
// 913 output tokens are reasoning.
const reasoningUsagePayload = `event: response.created
data: {"type":"response.created","response":{"id":"resp_4","model":"gpt-5.6-sol","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_4","status":"in_progress","role":"assistant","content":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"delta":"42"}

event: response.output_text.done
data: {"type":"response.output_text.done","output_index":0,"text":"42"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_4","model":"gpt-5.6-sol","status":"completed","usage":{"input_tokens":32,"input_tokens_details":{"cached_tokens":16,"cache_write_tokens":8},"output_tokens":913,"output_tokens_details":{"reasoning_tokens":516},"total_tokens":945}}}

`

func TestStreamReasoningUsage(t *testing.T) {
	fs := &fakeServer{payload: reasoningUsagePayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.6-sol",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if msg.Usage.ReasoningTokens != 516 {
		t.Errorf("Usage.ReasoningTokens=%d, want 516", msg.Usage.ReasoningTokens)
	}
	if msg.Usage.OutputTokens != 913 {
		t.Errorf("Usage.OutputTokens=%d, want 913 (reasoning stays nested)", msg.Usage.OutputTokens)
	}
	if msg.Usage.CacheReadTokens != 16 {
		t.Errorf("Usage.CacheReadTokens=%d, want 16", msg.Usage.CacheReadTokens)
	}
	if msg.Usage.CacheWriteTokens != 8 {
		t.Errorf("Usage.CacheWriteTokens=%d, want 8", msg.Usage.CacheWriteTokens)
	}
}

// Non-reasoning model: the details blocks are absent from the wire.
func TestStreamUsageWithoutDetails(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.4-mini",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if msg.Usage.ReasoningTokens != 0 || msg.Usage.CacheReadTokens != 0 || msg.Usage.CacheWriteTokens != 0 {
		t.Errorf("want zero details, got %+v", msg.Usage)
	}
	if msg.Usage.OutputTokens != 3 {
		t.Errorf("Usage.OutputTokens=%d, want 3", msg.Usage.OutputTokens)
	}
}

func TestStreamHTTPError(t *testing.T) {
	fs := &fakeServer{statusCode: http.StatusUnauthorized, statusBody: `{"error":"bad key"}`}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)
	var gotErr error
	for _, err := range p.Stream(context.Background(), llm.Request{
		Model:    "x",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil {
		t.Fatal("want error, got nil")
	}
	var apiErr *llm.APIError
	if !strings.Contains(gotErr.Error(), "bad key") {
		t.Errorf("err missing body: %v", gotErr)
	}
	if !apiErrCast(&apiErr, gotErr) || apiErr.Status != http.StatusUnauthorized {
		t.Errorf("APIError extraction failed: %v", gotErr)
	}
}

func apiErrCast(target **llm.APIError, err error) bool {
	for cur := err; cur != nil; {
		if a, ok := cur.(*llm.APIError); ok {
			*target = a
			return true
		}
		u, ok := cur.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		cur = u.Unwrap()
	}
	return false
}

// Effort is a per-request field here now: a caller running a cheap
// elicitation and an expensive authoring turn no longer needs two providers.
func TestStreamRequestThinkingOverridesOptions(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai_responses.New(openai_responses.Options{
		APIKey:          "test",
		BaseURL:         srv.URL,
		ReasoningEffort: openai_responses.ReasoningLow,
	})

	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.6-sol",
		Thinking: &llm.ThinkingConfig{Effort: llm.EffortHigh, Display: llm.DisplaySummarized},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	r, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("no reasoning block: %#v", body)
	}
	if r["effort"] != "high" {
		t.Errorf("effort=%v, want high (request beats Options)", r["effort"])
	}
	if r["summary"] != "auto" {
		t.Errorf("summary=%v, want auto (DisplaySummarized)", r["summary"])
	}
}

// DisplayOmitted turns a provider-level summary default back off.
func TestStreamRequestDisplayOmittedSuppressesSummary(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai_responses.New(openai_responses.Options{
		APIKey:                  "test",
		BaseURL:                 srv.URL,
		ReasoningEffort:         openai_responses.ReasoningMedium,
		IncludeReasoningSummary: true,
	})

	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.6-sol",
		Thinking: &llm.ThinkingConfig{Display: llm.DisplayOmitted},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	r := body["reasoning"].(map[string]any)
	if _, present := r["summary"]; present {
		t.Errorf("summary present despite DisplayOmitted: %#v", r)
	}
	if r["effort"] != "medium" {
		t.Errorf("effort=%v, want medium (Options fills the silence)", r["effort"])
	}
}

// A provider built with a house effort keeps working when the request is
// silent — the Options remain the default, not a dead field.
func TestStreamOptionsRemainDefaultWithoutThinking(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai_responses.New(openai_responses.Options{
		APIKey:          "test",
		BaseURL:         srv.URL,
		ReasoningEffort: openai_responses.ReasoningHigh,
	})

	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.6-sol",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	if body["reasoning"].(map[string]any)["effort"] != "high" {
		t.Errorf("Options effort was lost: %#v", body["reasoning"])
	}
}

func TestStreamBudgetTokensRejected(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:    "gpt-5.6-sol",
		Thinking: &llm.ThinkingConfig{BudgetTokens: 4096},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("want ErrUnsupportedThinking, got nil")
	}
	if !errors.Is(err, llm.ErrUnsupportedThinking) {
		t.Errorf("errors.Is(err, ErrUnsupportedThinking)=false: %v", err)
	}
}
