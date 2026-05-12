// Package gemini implements the pi-llm-go LLM interface against Google's
// Gemini API (https://generativelanguage.googleapis.com). Supports text,
// image, and video input across the Gemini 2.5 and Gemini 3 model
// families, plus tool calling and extended thinking ("thoughts").
//
// Video input is Gemini-exclusive among the providers pi-llm-go ships;
// the Anthropic and OpenAI providers reject llm.VideoBlock at the
// boundary so a misrouted multimodal request fails loudly instead of
// silently dropping data.
//
// VideoBlock supports two emission shapes today:
//
//   - **Inline base64** via VideoBlock.Data + MimeType, for files
//     under ~20MB.
//   - **URI reference** via VideoBlock.URI, which accepts either a
//     YouTube URL (public videos only; free-tier 8h/day quota) or a
//     pre-uploaded Files API handle (https://generativelanguage.googleapis.com/v1beta/files/...).
//
// A first-party Files API uploader (Upload / Wait / Delete helpers
// with multipart + ACTIVE-state polling) is planned for v0.5.0.
// Callers needing it today can use Google's official genai-go SDK to
// upload, then pass the resulting URI to VideoBlock.URI — pi-llm-go is
// URI-agnostic, no special handling required.
package gemini

import (
	"context"
	"fmt"
	"io"
	"iter"
	"net/http"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// DefaultBaseURL is the standard Google AI endpoint. Override via
// Options.BaseURL for proxies, mocks, or non-standard regional
// endpoints. Vertex AI (which uses a different URL scheme and OAuth
// instead of an API key) is intentionally NOT supported at v0.4.0;
// adding it requires a separate Backend option and OAuth token plumbing.
const DefaultBaseURL = "https://generativelanguage.googleapis.com"

// Options configures the Gemini provider.
type Options struct {
	// APIKey is the Google AI API key. Sent as the x-goog-api-key
	// header. Required.
	APIKey string

	// BaseURL overrides DefaultBaseURL — useful for proxies / mocks.
	BaseURL string

	// HTTPClient overrides the default net/http client. Required for
	// OpenTelemetry instrumentation, custom timeouts, or test fakes.
	HTTPClient *http.Client
}

// Provider implements llm.LLM against the Gemini API.
type Provider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// New constructs a Provider. Returns an error if APIKey is empty.
func New(opts Options) (*Provider, error) {
	if opts.APIKey == "" {
		return nil, fmt.Errorf("gemini: APIKey is required")
	}
	base := opts.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Provider{
		apiKey:  opts.APIKey,
		baseURL: base,
		client:  client,
	}, nil
}

// Stream implements llm.LLM. Posts to
// {base}/v1beta/models/{model}:streamGenerateContent?alt=sse and parses
// the SSE response into llm.StreamEvent values.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, err := buildRequestBody(req)
		if err != nil {
			yield(nil, fmt.Errorf("gemini: build request: %w", err))
			return
		}
		url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse", p.baseURL, req.Model)
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
		if err != nil {
			yield(nil, fmt.Errorf("gemini: new request: %w", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-goog-api-key", p.apiKey)
		httpReq.Header.Set("Accept", "text/event-stream")

		resp, err := p.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("gemini: do request: %w", err))
			return
		}
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, &llm.APIError{
				Provider: "gemini",
				Status:   resp.StatusCode,
				Body:     respBody,
				Inner:    llm.SentinelForStatus(resp.StatusCode),
			})
			return
		}

		decodeStream(resp.Body, req.Model, yield)
	}
}
