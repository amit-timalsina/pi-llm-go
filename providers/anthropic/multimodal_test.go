package anthropic_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// Reuses textOnlyPayload + fakeServer + newProvider from anthropic_test.go.

const (
	// 1x1 transparent PNG (smallest valid PNG, base64-encoded).
	tinyPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

// TestImageBlock_AnthropicWireShape verifies an ImageBlock in a user
// message is emitted with the {type: "image", source: {type: "base64",
// media_type, data}} shape that Anthropic expects.
func TestImageBlock_AnthropicWireShape(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
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
		t.Fatalf("decode body: %v", err)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content blocks=%d, want 2 (text + image, no placeholder)", len(content))
	}

	first := content[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "describe this image" {
		t.Errorf("first block shape wrong: %+v", first)
	}

	second := content[1].(map[string]any)
	if second["type"] != "image" {
		t.Fatalf("second block type=%v, want image", second["type"])
	}
	src, ok := second["source"].(map[string]any)
	if !ok {
		t.Fatalf("image block missing source: %+v", second)
	}
	if src["type"] != "base64" {
		t.Errorf("source.type=%v, want base64", src["type"])
	}
	if src["media_type"] != "image/png" {
		t.Errorf("source.media_type=%v, want image/png", src["media_type"])
	}
	if src["data"] != tinyPNGBase64 {
		t.Errorf("source.data round-trip mismatch")
	}
	// Source should not carry stray fields (e.g. url) at this version.
	if _, has := src["url"]; has {
		t.Errorf("source unexpectedly carries 'url' field: %+v", src)
	}
}

// TestImageBlock_AnthropicPlaceholderTextWhenImageOnly verifies the
// "(see attached image)" placeholder gets prepended when a user message
// contains an image but no text block — matches Mario's pi-ai behavior
// and Anthropic's preference for accompanying text.
func TestImageBlock_AnthropicPlaceholderTextWhenImageOnly(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
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

	if len(content) != 2 {
		t.Fatalf("content blocks=%d, want 2 (placeholder text + image)", len(content))
	}
	first := content[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "(see attached image)" {
		t.Errorf("first block should be placeholder text; got %+v", first)
	}
	if content[1].(map[string]any)["type"] != "image" {
		t.Errorf("second block should be image; got %+v", content[1])
	}
}

// TestImageBlock_AnthropicNoPlaceholderWhenTextPresent confirms the
// placeholder is NOT injected when any text block already accompanies
// the image, regardless of order.
func TestImageBlock_AnthropicNoPlaceholderWhenTextPresent(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/png"},
				llm.TextBlock{Text: "what is this?"}, // text AFTER image
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Errorf("content blocks=%d, want exactly 2 (no placeholder injected)", len(content))
	}
}

// TestImageBlock_AnthropicMultipleImagesPreserveOrder verifies multiple
// images and interleaved text preserve their order in the wire format.
func TestImageBlock_AnthropicMultipleImagesPreserveOrder(t *testing.T) {
	fs := &fakeServer{payload: textOnlyPayload}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	p := newProvider(t, srv)

	_, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     anthropic.ClaudeSonnet4_6,
		MaxTokens: 64,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "first"},
				llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/png"},
				llm.TextBlock{Text: "middle"},
				llm.ImageBlock{Data: tinyPNGBase64, MimeType: "image/jpeg"},
				llm.TextBlock{Text: "last"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(fs.lastBody, &body)
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 5 {
		t.Fatalf("content blocks=%d, want 5", len(content))
	}
	wantTypes := []string{"text", "image", "text", "image", "text"}
	for i, raw := range content {
		got := raw.(map[string]any)["type"]
		if got != wantTypes[i] {
			t.Errorf("content[%d].type=%v, want %v", i, got, wantTypes[i])
		}
	}
	// Second image's media_type must be image/jpeg, not image/png.
	img2 := content[3].(map[string]any)["source"].(map[string]any)
	if img2["media_type"] != "image/jpeg" {
		t.Errorf("content[3].source.media_type=%v, want image/jpeg", img2["media_type"])
	}
}

// TestImageBlock_AnthropicRejectsEmptyData covers the boundary
// validation: empty Data or MimeType returns an error rather than
// emitting an obviously-broken wire payload.
func TestImageBlock_AnthropicRejectsEmptyData(t *testing.T) {
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
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := llm.Complete(context.Background(), p, llm.Request{
				Model:     anthropic.ClaudeSonnet4_6,
				MaxTokens: 64,
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
