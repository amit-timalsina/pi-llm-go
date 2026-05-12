package openai_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/openai"
)

// TestImageBlock_OpenAIEmptyContentOmitted verifies the wire-format
// regression guard introduced when apiMessage.Content changed from
// `string` to `any` to support multimodal arrays. The string-typed
// field with `,omitempty` previously omitted empty strings; the
// interface-typed field does NOT omit empty strings unless we set
// the value to nil explicitly. Verify three paths:
//
//   - text-only user message with empty text — no "content" key
//   - assistant message with only tool_calls (no text) — no "content" key
//   - tool message with empty Content — no "content" key
//
// A regression here would break compatibility with strict OpenAI-
// compatible hosts that reject `"content": ""`.
func TestImageBlock_OpenAIEmptyContentOmitted(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	// Assistant message with ONLY a tool_call (no text); tool message
	// with empty Content. Both should serialize without a "content" key.
	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: openai.GPT5_5,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "go"}}},
			{Role: llm.RoleAssistant, Content: []llm.Block{
				llm.ToolCallBlock{ID: "c1", Name: "x", Arguments: json.RawMessage(`{}`)},
			}},
			{Role: llm.RoleTool, Content: []llm.Block{
				llm.ToolResultBlock{ToolCallID: "c1", Content: ""},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// Search the raw body for the regression substring. JSON-decoding to
	// a map would obscure it because `"content":""` would deserialize as
	// `"content": ""` and look the same as the absent case after the fact.
	body := string(fs.lastBody)
	if strings.Contains(body, `"content":""`) {
		t.Errorf("regression: empty content was emitted as \"content\":\"\"; should be omitted entirely.\nBody:\n%s", body)
	}
}

// Reuses textOnlyPayload + fakeServer + newProvider from openai_test.go.

const tinyPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

// TestImageBlock_OpenAITextOnlyMessageStaysString verifies the
// compatibility-preserving fast path: when no ImageBlock is present, a
// user message's content is emitted as a plain string (not an array).
// This matches existing v0.2.0 behavior and keeps the wire format
// compatible with hosts that only accept the legacy shape.
func TestImageBlock_OpenAITextOnlyMessageStaysString(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: openai.GPT5_5,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hi"}}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	c := body["messages"].([]any)[0].(map[string]any)["content"]
	if _, isString := c.(string); !isString {
		t.Errorf("text-only user content=%T, want string fast path", c)
	}
}

// TestImageBlock_OpenAIWireShape verifies an ImageBlock triggers the
// array form with {type: "image_url", image_url: {url: "data:<mime>;base64,..."}}.
func TestImageBlock_OpenAIWireShape(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: openai.GPT5_5,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "describe this image"},
				llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/png"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(fs.lastBody, &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	content, ok := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if !ok {
		t.Fatalf("multimodal content=%T, want array form", body["messages"].([]any)[0].(map[string]any)["content"])
	}
	if len(content) != 2 {
		t.Fatalf("content parts=%d, want 2", len(content))
	}

	first := content[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "describe this image" {
		t.Errorf("first part shape wrong: %+v", first)
	}

	second := content[1].(map[string]any)
	if second["type"] != "image_url" {
		t.Fatalf("second part type=%v, want image_url", second["type"])
	}
	imageURL, ok := second["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("second part missing image_url object: %+v", second)
	}
	wantPrefix := "data:image/png;base64,"
	url, _ := imageURL["url"].(string)
	if !strings.HasPrefix(url, wantPrefix) {
		t.Errorf("image_url.url=%q, want prefix %q", url, wantPrefix)
	}
	if !strings.HasSuffix(url, tinyPNGBase64) {
		t.Errorf("image_url.url missing the base64 body suffix")
	}
}

// TestImageBlock_OpenAIMultipleImagesPreserveOrder verifies block order
// in the wire array matches input order, with interleaved text and
// multiple media types.
func TestImageBlock_OpenAIMultipleImagesPreserveOrder(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: openai.GPT5_5,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "first"},
				llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/jpeg"},
				llm.TextBlock{Text: "middle"},
				llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/png"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 4 {
		t.Fatalf("content parts=%d, want 4", len(content))
	}
	wantTypes := []string{"text", "image_url", "text", "image_url"}
	for i, raw := range content {
		got := raw.(map[string]any)["type"]
		if got != wantTypes[i] {
			t.Errorf("content[%d].type=%v, want %v", i, got, wantTypes[i])
		}
	}
	url1 := content[1].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(url1, "data:image/jpeg;base64,") {
		t.Errorf("first image URL=%q, want image/jpeg prefix", url1)
	}
}

// TestImageBlock_OpenAIRejectsEmptyData mirrors the Anthropic boundary
// validation: empty Data, empty MimeType, or a "data:" URI prefix all
// return an error.
func TestImageBlock_OpenAIRejectsEmptyData(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	cases := []struct {
		name string
		blk  llm.ImageBlock
	}{
		{"empty Data", llm.ImageBlock{Data: "", MimeType: "image/png"}},
		{"empty MimeType", llm.ImageBlock{Data: tinyPNGBase64, MimeType: ""}},
		{"data: URI prefix in Data", llm.ImageBlock{
			Data:     "data:image/png;base64," + tinyPNGBase64,
			MimeType: "image/png",
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := llm.Complete(context.Background(), p, llm.Request{
				Model: openai.GPT5_5,
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []llm.Block{tc.blk}},
				},
			})
			if err == nil {
				t.Errorf("expected error for %s; got nil", tc.name)
			}
		})
	}
}

// TestImageBlock_OpenAIRejectsNonUserRoles verifies the role guard:
// ImageBlock on assistant/tool messages errors at the boundary.
func TestImageBlock_OpenAIRejectsNonUserRoles(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	roles := []llm.Role{llm.RoleAssistant, llm.RoleTool}
	for _, role := range roles {
		t.Run(string(role), func(t *testing.T) {
			_, err := llm.Complete(context.Background(), p, llm.Request{
				Model: openai.GPT5_5,
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "go"}}},
					{Role: role, Content: []llm.Block{
						llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/png"},
					}},
				},
			})
			if err == nil {
				t.Errorf("expected role-guard error for role=%s; got nil", role)
			}
		})
	}
}

// TestVideoBlock_OpenAIRejected pins the contract that OpenAI Chat
// Completions has no native video input — VideoBlock must error
// loudly instead of silently dropping.
func TestVideoBlock_OpenAIRejected(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: openai.GPT5_5,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.VideoBlock{URI: "https://www.youtube.com/watch?v=abc"},
			}},
		},
	})
	if err == nil {
		t.Fatal("expected VideoBlock-not-supported error from OpenAI; got nil")
	}
	if !strings.Contains(err.Error(), "VideoBlock") {
		t.Errorf("error %q should mention VideoBlock", err.Error())
	}
}

// TestImageBlock_OpenAIImageOnlyNoPlaceholder verifies the asymmetry
// vs Anthropic: OpenAI Chat Completions accepts image-only user
// messages, so no synthetic placeholder text is injected. The wire
// array should contain exactly one image_url part.
func TestImageBlock_OpenAIImageOnlyNoPlaceholder(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p, _ := openai.New(openai.Options{APIKey: "test", BaseURL: srv.URL})

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: openai.GPT5_5,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/png"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 1 {
		t.Errorf("image-only user message: got %d content parts, want exactly 1 (no placeholder)", len(content))
	}
	if content[0].(map[string]any)["type"] != "image_url" {
		t.Errorf("solo part should be image_url; got %+v", content[0])
	}
}
