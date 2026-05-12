package openai_responses_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// Reuses textOnlyPayload + fakeServer + newProvider from responses_test.go.

const tinyPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

// TestImageBlock_ResponsesWireShape verifies an ImageBlock is emitted
// as {type: "input_image", image_url: "data:<mime>;base64,..."} on
// the Responses API (image_url is a flat STRING here, unlike Chat
// Completions which wraps it in an object).
func TestImageBlock_ResponsesWireShape(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: "gpt-5.5",
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
	input := body["input"].([]any)
	if len(input) != 1 {
		t.Fatalf("input items=%d, want 1", len(input))
	}
	msg := input[0].(map[string]any)
	if msg["role"] != "user" {
		t.Errorf("input[0].role=%v, want user", msg["role"])
	}
	content, ok := msg["content"].([]any)
	if !ok {
		t.Fatalf("content=%T, want []any", msg["content"])
	}
	if len(content) != 2 {
		t.Fatalf("content parts=%d, want 2", len(content))
	}

	first := content[0].(map[string]any)
	if first["type"] != "input_text" || first["text"] != "describe this image" {
		t.Errorf("first part shape wrong: %+v", first)
	}

	second := content[1].(map[string]any)
	if second["type"] != "input_image" {
		t.Fatalf("second part type=%v, want input_image", second["type"])
	}
	url, _ := second["image_url"].(string)
	if !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Errorf("input_image.image_url=%q, want data: prefix", url)
	}
	if !strings.HasSuffix(url, tinyPNGBase64) {
		t.Errorf("input_image.image_url missing base64 body suffix")
	}
	// Responses API takes image_url as a flat string, not nested object.
	// If a "url" sub-key appeared, we'd be emitting the wrong wire shape.
	if _, has := second["url"]; has {
		t.Errorf("second part should not have a 'url' top-level field; got %+v", second)
	}
}

// TestImageBlock_ResponsesTextOnlyStaysCollapsed verifies the
// no-image fast path still emits a single input_text content part
// (existing v0.2.0 wire shape).
func TestImageBlock_ResponsesTextOnlyStaysCollapsed(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: "gpt-5.5",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "line 1"},
				llm.TextBlock{Text: "line 2"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	content := body["input"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("text-only fast path emitted %d parts, want 1 (concatenated)", len(content))
	}
	part := content[0].(map[string]any)
	if part["type"] != "input_text" || part["text"] != "line 1\nline 2" {
		t.Errorf("collapsed text part wrong: %+v", part)
	}
}

// TestImageBlock_ResponsesRejectsEmptyData mirrors the boundary
// validation: empty Data, empty MimeType, or a "data:" URI prefix all
// return an error.
func TestImageBlock_ResponsesRejectsEmptyData(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

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
				Model: "gpt-5.5",
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

// TestVideoBlock_ResponsesRejected pins the contract that the OpenAI
// Responses API has no native video input — VideoBlock must error.
func TestVideoBlock_ResponsesRejected(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model: "gpt-5.5",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.VideoBlock{URI: "https://www.youtube.com/watch?v=abc"},
			}},
		},
	})
	if err == nil {
		t.Fatal("expected VideoBlock-not-supported error from Responses API; got nil")
	}
	if !strings.Contains(err.Error(), "VideoBlock") {
		t.Errorf("error %q should mention VideoBlock", err.Error())
	}
}

// TestImageBlock_ResponsesRejectsNonUserRoles verifies the role guard
// on the Responses API: ImageBlock outside user role errors.
func TestImageBlock_ResponsesRejectsNonUserRoles(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	roles := []llm.Role{llm.RoleAssistant, llm.RoleTool}
	for _, role := range roles {
		t.Run(string(role), func(t *testing.T) {
			_, err := llm.Complete(context.Background(), p, llm.Request{
				Model: "gpt-5.5",
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
