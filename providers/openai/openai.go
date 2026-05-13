// Package openai is the OpenAI-compatible Chat Completions provider for
// pi-llm-go. It targets the Chat Completions wire format, so it works with
// OpenAI itself, Groq, Together, vLLM, OpenRouter, Ollama, and any other
// service that speaks the same API.
//
// Configure the BaseURL to point at the desired host:
//
//	// OpenAI
//	openai.New(openai.Options{APIKey: key})
//
//	// Groq
//	openai.New(openai.Options{
//	    APIKey:  groqKey,
//	    BaseURL: "https://api.groq.com/openai/v1",
//	})
//
// The provider does NOT honor llm.Request.Thinking at v1 — reasoning-effort
// dialects vary too much across compatible providers to map portably.
package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// Canonical OpenAI model IDs as of May 2026. Convenience constants for IDE
// autocomplete — llm.Request.Model accepts any string, so newer or
// vendor-specific models (Groq, Together, vLLM, OpenRouter, Ollama) work
// without code changes.
//
// Verify the latest via platform.openai.com/docs/models before pinning a
// constant in production code; OpenAI churns model IDs more aggressively
// than Anthropic.
const (
	GPT5_5     = "gpt-5.5"
	GPT5_4     = "gpt-5.4"
	GPT5_4Mini = "gpt-5.4-mini"
	GPT5_4Nano = "gpt-5.4-nano"
	GPT4_1     = "gpt-4.1"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Options configures a Provider at construction time.
type Options struct {
	APIKey     string
	BaseURL    string       // default "https://api.openai.com/v1"
	HTTPClient *http.Client // default http.DefaultClient
	OrgID      string       // optional, sent as OpenAI-Organization header
	Project    string       // optional, sent as OpenAI-Project header

	// URL, if non-empty, is used verbatim as the chat-completions endpoint
	// instead of BaseURL + "/chat/completions". Use this for hosts where the
	// chat path differs from the default — most notably Azure OpenAI, whose
	// endpoint embeds a deployment name and an api-version query:
	//
	//   URL: "https://<resource>.cognitiveservices.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-12-01-preview"
	//
	// When set, BaseURL is ignored.
	URL string

	// Headers are merged into the outgoing HTTP request, after the default
	// headers are applied. Use this for hosts that require non-standard auth
	// — most notably Azure OpenAI's "api-key: <value>" header (instead of
	// the default "Authorization: Bearer ..."):
	//
	//   openai.New(openai.Options{
	//       URL:     "https://<resource>.cognitiveservices.azure.com/...",
	//       Headers: map[string]string{"api-key": dataPlaneKey},
	//       // APIKey can be left empty; Headers supplies auth.
	//   })
	//
	// If APIKey is also non-empty, "Authorization: Bearer <APIKey>" is set
	// first; Headers can override it. Header values supplied here win over
	// any default the provider would otherwise set.
	Headers map[string]string

	// Retry, when non-nil, configures provider-side retry on retriable
	// errors (rate-limit / 5xx / network). nil disables retry. See
	// llm.RetryPolicy for the contract; llm.DefaultRetryPolicy returns a
	// sane starting point. Mid-stream connection breaks are not retried.
	Retry *llm.RetryPolicy
}

// Provider is the OpenAI-compatible implementation of llm.LLM. Safe for
// concurrent use; each Stream call issues its own HTTP request.
type Provider struct {
	apiKey  string
	baseURL string
	url     string
	orgID   string
	project string
	headers map[string]string
	client  *http.Client
	retry   llm.RetryPolicy
}

// New constructs a Provider. APIKey is required unless Headers supplies an
// alternative authentication header (e.g. Azure's "api-key" header).
func New(opts Options) (*Provider, error) {
	if opts.APIKey == "" && len(opts.Headers) == 0 {
		return nil, errors.New("openai: APIKey or Headers (with auth header) is required")
	}
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = http.DefaultClient
	}
	// Copy the headers map so later caller mutations don't affect the provider.
	var headers map[string]string
	if len(opts.Headers) > 0 {
		headers = make(map[string]string, len(opts.Headers))
		for k, v := range opts.Headers {
			headers[k] = v
		}
	}
	p := &Provider{
		apiKey:  opts.APIKey,
		baseURL: opts.BaseURL,
		url:     opts.URL,
		orgID:   opts.OrgID,
		project: opts.Project,
		headers: headers,
		client:  opts.HTTPClient,
	}
	if opts.Retry != nil {
		p.retry = *opts.Retry
	}
	return p, nil
}

// endpoint returns the URL the provider should POST to.
func (p *Provider) endpoint() string {
	if p.url != "" {
		return p.url
	}
	return p.baseURL + "/chat/completions"
}

// Stream issues a streaming completion. Provider errors surface as
// *llm.APIError via the iterator's error half. Options.Retry, when
// set, retries the initial HTTP attempt; mid-stream connection breaks
// are NOT retried.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		resp, err := llm.RunWithRetry(ctx, p.retry, func() (*http.Response, error) {
			return p.doStreamRequest(ctx, req)
		})
		if err != nil {
			yield(nil, err)
			return
		}
		defer func() { _ = resp.Body.Close() }()

		streamErr := newStreamDecoder().decode(resp.Body, yield)
		if streamErr != nil && !errors.Is(streamErr, errIterationStopped) {
			yield(nil, streamErr)
		}
	}
}

func (p *Provider) doStreamRequest(ctx context.Context, req llm.Request) (*http.Response, error) {
	body, err := buildRequestBody(req)
	if err != nil {
		return nil, fmt.Errorf("openai: build request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), body)
	if err != nil {
		return nil, fmt.Errorf("openai: new request: %w", err)
	}
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if p.orgID != "" {
		httpReq.Header.Set("OpenAI-Organization", p.orgID)
	}
	if p.project != "" {
		httpReq.Header.Set("OpenAI-Project", p.project)
	}
	// User-supplied headers win over defaults — lets callers override
	// auth (Azure's "api-key:" instead of "Authorization: Bearer").
	for k, v := range p.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: do request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, &llm.APIError{
			Provider:   "openai",
			Status:     resp.StatusCode,
			Body:       respBody,
			Inner:      llm.SentinelFor(resp.StatusCode, respBody),
			RetryAfter: llm.ParseRetryAfter(resp.Header),
		}
	}
	return resp, nil
}
