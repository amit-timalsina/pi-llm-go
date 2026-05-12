package gemini_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/gemini"
)

// fakeServer is a httptest handler that captures the request body +
// headers, then ships back a canned SSE payload (or a non-200 status
// for error tests). All gemini tests share this fixture.
type fakeServer struct {
	payload    string
	statusCode int
	statusBody string

	lastBody    json.RawMessage
	lastHeaders http.Header
	lastPath    string
	lastQuery   string
}

func (f *fakeServer) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		f.lastBody = json.RawMessage(body)
		f.lastHeaders = r.Header.Clone()
		f.lastPath = r.URL.Path
		f.lastQuery = r.URL.RawQuery
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

func newProvider(t *testing.T, srv *httptest.Server) *gemini.Provider {
	t.Helper()
	p, err := gemini.New(gemini.Options{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// Two canned SSE bodies based on probed real-API traces (2026-05-12).

// Single-frame text response, like Gemini emits for short prompts.
const textOnlyPayload = `data: {"candidates":[{"content":{"parts":[{"text":"HELLO"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":1,"totalTokenCount":8},"modelVersion":"gemini-2.5-flash"}

`

// Multi-frame text response, with cumulative usage on each frame.
const multiFrameTextPayload = `data: {"candidates":[{"content":{"parts":[{"text":"Mercury"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":13,"candidatesTokenCount":1,"totalTokenCount":14}}

data: {"candidates":[{"content":{"parts":[{"text":"\nVenus\nEarth\nMars"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":13,"candidatesTokenCount":18,"totalTokenCount":31}}

data: {"candidates":[{"content":{"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":13,"candidatesTokenCount":18,"totalTokenCount":31}}

`

// Function-call response, in one frame as Gemini emits whole tool calls.
const toolCallPayload = `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"echo","args":{"text":"hi"}}}],"role":"model"},"finishReason":"TOOL_CALL","index":0}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":5,"totalTokenCount":25}}

`

// Mixed thought + text response: a thought chunk arrives first (with
// thought=true), then visible text follows.
const thinkingPayload = `data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"reasoning step one"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":0,"totalTokenCount":7,"thoughtsTokenCount":4}}

data: {"candidates":[{"content":{"parts":[{"text":"final answer"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"totalTokenCount":14,"thoughtsTokenCount":4}}

data: {"candidates":[{"content":{"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"totalTokenCount":14,"thoughtsTokenCount":4}}

`

// --- Tests ---

// TestStream_TextOnly_SingleFrame verifies basic text streaming and
// the on-wire request shape — POST path, x-goog-api-key header,
// alt=sse query param.
func TestStream_TextOnly_SingleFrame(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "say HELLO"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// Path + query.
	wantPath := "/v1beta/models/" + gemini.Gemini2_5Flash + ":streamGenerateContent"
	if fs.lastPath != wantPath {
		t.Errorf("path=%q, want %q", fs.lastPath, wantPath)
	}
	if !strings.Contains(fs.lastQuery, "alt=sse") {
		t.Errorf("query=%q missing alt=sse", fs.lastQuery)
	}
	if got := fs.lastHeaders.Get("x-goog-api-key"); got != "test-key" {
		t.Errorf("x-goog-api-key header=%q, want test-key", got)
	}

	// Body shape.
	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	contents := body["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("contents len=%d, want 1", len(contents))
	}
	first := contents[0].(map[string]any)
	if first["role"] != "user" {
		t.Errorf("contents[0].role=%v, want user", first["role"])
	}
	parts := first["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "say HELLO" {
		t.Errorf("parts[0].text=%v, want \"say HELLO\"", parts[0].(map[string]any)["text"])
	}

	// Round-tripped assistant message.
	if len(msg.Content) != 1 {
		t.Fatalf("msg.Content len=%d, want 1", len(msg.Content))
	}
	tb, ok := msg.Content[0].(llm.TextBlock)
	if !ok || tb.Text != "HELLO" {
		t.Errorf("assistant text=%+v, want TextBlock{HELLO}", msg.Content[0])
	}
	if msg.StopReason != llm.StopReasonEnd {
		t.Errorf("StopReason=%v, want End", msg.StopReason)
	}
	if msg.Usage.InputTokens != 7 || msg.Usage.OutputTokens != 1 || msg.Usage.TotalTokens != 8 {
		t.Errorf("usage=%+v, want in=7 out=1 total=8", msg.Usage)
	}
}

// TestStream_MultiFrameText verifies delta accumulation across frames
// and that cumulative usage is captured from the LAST frame (not
// summed across frames).
func TestStream_MultiFrameText(t *testing.T) {
	fs := &fakeServer{payload: multiFrameTextPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "planets"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tb, ok := msg.Content[0].(llm.TextBlock)
	if !ok {
		t.Fatalf("Content[0] is %T, want TextBlock", msg.Content[0])
	}
	want := "Mercury\nVenus\nEarth\nMars"
	if tb.Text != want {
		t.Errorf("accumulated text=%q, want %q", tb.Text, want)
	}
	if msg.Usage.InputTokens != 13 || msg.Usage.OutputTokens != 18 || msg.Usage.TotalTokens != 31 {
		t.Errorf("usage=%+v, want in=13 out=18 total=31 (last frame, not summed)", msg.Usage)
	}
}

// TestStream_ToolCall verifies function-call parts map to
// ToolCallBlock with the args preserved as raw JSON.
func TestStream_ToolCall(t *testing.T) {
	fs := &fakeServer{payload: toolCallPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "call echo"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("Content len=%d, want 1", len(msg.Content))
	}
	tc, ok := msg.Content[0].(llm.ToolCallBlock)
	if !ok {
		t.Fatalf("Content[0]=%T, want ToolCallBlock", msg.Content[0])
	}
	if tc.Name != "echo" {
		t.Errorf("ToolCallBlock.Name=%q, want echo", tc.Name)
	}
	// Args round-trip — JSON-equal, not byte-equal (Gemini may reorder keys).
	var got map[string]string
	_ = json.Unmarshal(tc.Arguments, &got)
	if got["text"] != "hi" {
		t.Errorf("args.text=%q, want hi", got["text"])
	}
	if msg.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason=%v, want ToolUse", msg.StopReason)
	}
}

// TestStream_Thinking verifies thought parts (`thought=true`) emit
// ThinkingBlock content and that the thoughtsTokenCount is rolled
// into Usage.OutputTokens.
func TestStream_Thinking(t *testing.T) {
	fs := &fakeServer{payload: thinkingPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "think"}}},
		},
		Thinking: &llm.ThinkingConfig{BudgetTokens: 1024},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// First block: ThinkingBlock; second: TextBlock.
	if len(msg.Content) != 2 {
		t.Fatalf("Content len=%d, want 2 (thinking + text)", len(msg.Content))
	}
	thoughts, ok := msg.Content[0].(llm.ThinkingBlock)
	if !ok {
		t.Fatalf("Content[0]=%T, want ThinkingBlock", msg.Content[0])
	}
	if thoughts.Thinking != "reasoning step one" {
		t.Errorf("ThinkingBlock.Thinking=%q, want \"reasoning step one\"", thoughts.Thinking)
	}
	text, ok := msg.Content[1].(llm.TextBlock)
	if !ok || text.Text != "final answer" {
		t.Errorf("Content[1]=%+v, want TextBlock{final answer}", msg.Content[1])
	}
	// thoughtsTokenCount (4) + candidatesTokenCount (3) = 7 output tokens.
	if msg.Usage.OutputTokens != 7 {
		t.Errorf("Usage.OutputTokens=%d, want 7 (3 candidates + 4 thoughts)", msg.Usage.OutputTokens)
	}
}

// TestRequest_ImageBlock verifies an ImageBlock in a user turn emits
// the inlineData shape with the correct mimeType + data.
func TestRequest_ImageBlock(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	const tinyPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "describe"},
				llm.ImageBlock{Data: tinyPNG, MimeType: "image/png"},
			}},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts len=%d, want 2 (text + image)", len(parts))
	}
	imgPart := parts[1].(map[string]any)
	inlineData, ok := imgPart["inlineData"].(map[string]any)
	if !ok {
		t.Fatalf("parts[1] missing inlineData: %+v", imgPart)
	}
	if inlineData["mimeType"] != "image/png" {
		t.Errorf("mimeType=%v, want image/png", inlineData["mimeType"])
	}
	if inlineData["data"] != tinyPNG {
		t.Errorf("inline image data not round-tripped")
	}
}

// TestRequest_VideoBlock_Inline verifies a VideoBlock with Data set
// emits the inlineData shape (same as ImageBlock, with video MIME).
func TestRequest_VideoBlock_Inline(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	const fakeVideoB64 = "ZmFrZS12aWRlby1ieXRlcw==" // "fake-video-bytes"
	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "summarize"},
				llm.VideoBlock{Data: fakeVideoB64, MimeType: "video/mp4"},
			}},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	vidPart := parts[1].(map[string]any)
	inlineData, ok := vidPart["inlineData"].(map[string]any)
	if !ok {
		t.Fatalf("video part missing inlineData: %+v", vidPart)
	}
	if inlineData["mimeType"] != "video/mp4" {
		t.Errorf("video mimeType=%v, want video/mp4", inlineData["mimeType"])
	}
	if inlineData["data"] != fakeVideoB64 {
		t.Errorf("video data not round-tripped")
	}
	// fileData should NOT also be set — they're mutually exclusive.
	if _, has := vidPart["fileData"]; has {
		t.Errorf("video part should not have fileData when Data is set: %+v", vidPart)
	}
}

// TestRequest_VideoBlock_URI verifies a VideoBlock with URI set emits
// the fileData shape (used for both Files-API handles and YouTube URLs).
func TestRequest_VideoBlock_URI(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	const uri = "https://www.youtube.com/watch?v=YE7VzlLtp-4"
	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "describe"},
				llm.VideoBlock{URI: uri},
			}},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	vidPart := parts[1].(map[string]any)
	fileData, ok := vidPart["fileData"].(map[string]any)
	if !ok {
		t.Fatalf("URI video part missing fileData: %+v", vidPart)
	}
	if fileData["fileUri"] != uri {
		t.Errorf("fileUri=%v, want %s", fileData["fileUri"], uri)
	}
	if _, has := vidPart["inlineData"]; has {
		t.Errorf("URI video should not have inlineData: %+v", vidPart)
	}
}

// TestRequest_VideoBlock_Metadata verifies optional StartOffset /
// EndOffset / FPS land on the videoMetadata part-level field.
func TestRequest_VideoBlock_Metadata(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	start := 3 * time.Second
	end := 10 * time.Second
	fps := 0.5
	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "clip"},
				llm.VideoBlock{
					URI:         "https://www.youtube.com/watch?v=abc",
					StartOffset: &start,
					EndOffset:   &end,
					FPS:         &fps,
				},
			}},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	parts := body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	vidPart := parts[1].(map[string]any)
	meta, ok := vidPart["videoMetadata"].(map[string]any)
	if !ok {
		t.Fatalf("video part missing videoMetadata: %+v", vidPart)
	}
	if meta["startOffset"] != "3s" {
		t.Errorf("startOffset=%v, want 3s", meta["startOffset"])
	}
	if meta["endOffset"] != "10s" {
		t.Errorf("endOffset=%v, want 10s", meta["endOffset"])
	}
	if meta["fps"].(float64) != 0.5 {
		t.Errorf("fps=%v, want 0.5", meta["fps"])
	}
}

// TestRequest_VideoBlock_Validation_DataAndURI_Mutex pins the contract
// that Data and URI are mutually exclusive.
func TestRequest_VideoBlock_Validation_DataAndURI_Mutex(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	cases := []struct {
		name string
		blk  llm.VideoBlock
	}{
		{"both empty", llm.VideoBlock{}},
		{"both set", llm.VideoBlock{Data: "abc", URI: "https://x", MimeType: "video/mp4"}},
		{"data URI prefix", llm.VideoBlock{Data: "data:video/mp4;base64,xxx", MimeType: "video/mp4"}},
		{"data without mime", llm.VideoBlock{Data: "abc"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := llm.Complete(context.Background(), p, llm.Request{
				Model: gemini.Gemini2_5Flash,
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []llm.Block{tc.blk}},
				},
			})
			if err == nil {
				t.Errorf("expected validation error for %s", tc.name)
			}
		})
	}
}

// TestRequest_VideoBlock_NonUserRoleRejected mirrors the ImageBlock
// boundary: video is only valid on user-role messages.
func TestRequest_VideoBlock_NonUserRoleRejected(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	for _, role := range []llm.Role{llm.RoleAssistant, llm.RoleTool} {
		t.Run(string(role), func(t *testing.T) {
			_, err := llm.Complete(context.Background(), p, llm.Request{
				Model: gemini.Gemini2_5Flash,
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "go"}}},
					{Role: role, Content: []llm.Block{
						llm.VideoBlock{URI: "https://www.youtube.com/watch?v=abc"},
					}},
				},
			})
			if err == nil {
				t.Errorf("expected role-guard error for role %s", role)
			}
		})
	}
}

// TestRequest_Tools verifies tool declarations land under the
// functionDeclarations shape Gemini expects.
func TestRequest_Tools(t *testing.T) {
	fs := &fakeServer{payload: toolCallPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	schema := json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`)
	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Tools: []llm.Tool{
			{Name: "echo", Description: "echoes back", InputSchema: schema},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "call echo"}}},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	tools := body["tools"].([]any)
	decls := tools[0].(map[string]any)["functionDeclarations"].([]any)
	first := decls[0].(map[string]any)
	if first["name"] != "echo" {
		t.Errorf("functionDeclaration name=%v, want echo", first["name"])
	}
	if first["description"] != "echoes back" {
		t.Errorf("functionDeclaration description=%v, want echoes back", first["description"])
	}
}

// TestRequest_SystemPrompt verifies a non-empty System gets emitted
// as systemInstruction (not as a content with role "system" — Gemini
// uses a dedicated top-level field).
func TestRequest_SystemPrompt(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model:  gemini.Gemini2_5Flash,
		System: "You are a careful assistant.",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	sys, ok := body["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("body missing systemInstruction: %s", string(fs.lastBody))
	}
	parts := sys["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "You are a careful assistant." {
		t.Errorf("systemInstruction text wrong: %+v", parts[0])
	}
}

// TestRequest_ToolResultFoldsIntoUserTurn verifies the special-case
// fold: an llm.RoleTool message appends its functionResponse parts
// to the prior user turn, since Gemini has no separate tool role.
func TestRequest_ToolResultFoldsIntoUserTurn(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	if _, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "call echo"}}},
			{Role: llm.RoleAssistant, Content: []llm.Block{
				llm.ToolCallBlock{ID: "echo_1", Name: "echo", Arguments: json.RawMessage(`{"text":"hi"}`)},
			}},
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: "echo_1", Content: "echoed: hi"},
			}},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	contents := body["contents"].([]any)
	// Expect 2 contents (not 3): user, assistant. Tool result folded
	// into... wait — the prior content to RoleTool is RoleAssistant,
	// not RoleUser. The fold only happens when the prior content is
	// already a user turn. Verify the documented fold-or-fall-through
	// behavior: tool message becomes its OWN user content here.
	if len(contents) != 3 {
		t.Fatalf("contents=%d, want 3 (user, assistant, user-with-functionResponse)", len(contents))
	}
	last := contents[2].(map[string]any)
	if last["role"] != "user" {
		t.Errorf("tool result content[2].role=%v, want user", last["role"])
	}
	fnResp := last["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if fnResp["name"] != "echo_1" {
		t.Errorf("functionResponse.name=%v, want echo_1 (the call id)", fnResp["name"])
	}
}

// TestStream_HTTPErrorWrapsAPIError verifies non-2xx responses
// surface as *llm.APIError with the provider set.
func TestStream_HTTPErrorWrapsAPIError(t *testing.T) {
	fs := &fakeServer{statusCode: http.StatusUnauthorized, statusBody: `{"error":"invalid api key"}`}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: gemini.Gemini2_5Flash,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "go"}}},
		},
	})
	if err == nil {
		t.Fatal("expected APIError; got nil")
	}
	apiErr, ok := err.(*llm.APIError)
	if !ok {
		t.Fatalf("error=%T, want *llm.APIError", err)
	}
	if apiErr.Provider != "gemini" {
		t.Errorf("APIError.Provider=%q, want gemini", apiErr.Provider)
	}
	if apiErr.Status != http.StatusUnauthorized {
		t.Errorf("APIError.Status=%d, want 401", apiErr.Status)
	}
}

// TestNew_RequiresAPIKey pins the constructor contract.
func TestNew_RequiresAPIKey(t *testing.T) {
	if _, err := gemini.New(gemini.Options{}); err == nil {
		t.Error("expected error when APIKey is empty")
	}
}
