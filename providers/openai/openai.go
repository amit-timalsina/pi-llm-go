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

	llm "github.com/amittimalsina/pi-llm-go"
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
}

// Provider is the OpenAI-compatible implementation of llm.LLM. Safe for
// concurrent use; each Stream call issues its own HTTP request.
type Provider struct {
	apiKey  string
	baseURL string
	orgID   string
	project string
	client  *http.Client
}

// New constructs a Provider. APIKey is required.
func New(opts Options) (*Provider, error) {
	if opts.APIKey == "" {
		return nil, errors.New("openai: APIKey is required")
	}
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = http.DefaultClient
	}
	return &Provider{
		apiKey:  opts.APIKey,
		baseURL: opts.BaseURL,
		orgID:   opts.OrgID,
		project: opts.Project,
		client:  opts.HTTPClient,
	}, nil
}

// Stream issues a streaming completion. Provider errors surface as
// *llm.APIError via the iterator's error half.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, err := buildRequestBody(req)
		if err != nil {
			yield(nil, fmt.Errorf("openai: build request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", body)
		if err != nil {
			yield(nil, fmt.Errorf("openai: new request: %w", err))
			return
		}
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		if p.orgID != "" {
			httpReq.Header.Set("OpenAI-Organization", p.orgID)
		}
		if p.project != "" {
			httpReq.Header.Set("OpenAI-Project", p.project)
		}

		resp, err := p.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("openai: do request: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, &llm.APIError{
				Provider: "openai",
				Status:   resp.StatusCode,
				Body:     respBody,
				Inner:    llm.SentinelForStatus(resp.StatusCode),
			})
			return
		}

		streamErr := newStreamDecoder().decode(resp.Body, yield)
		if streamErr != nil && !errors.Is(streamErr, errIterationStopped) {
			yield(nil, streamErr)
		}
	}
}
