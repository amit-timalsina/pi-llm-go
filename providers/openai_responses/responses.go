// Package openai_responses is the OpenAI Responses API (/v1/responses)
// provider for pi-llm-go. It's a sibling of providers/openai (Chat
// Completions) covering the newer endpoint shape: structured input items,
// server-side state via previous_response_id, reasoning summary streaming,
// and built-in tool primitives.
//
// When to pick which provider:
//
//   - providers/openai           — Chat Completions /v1/chat/completions.
//                                  The workhorse, supports tool calling,
//                                  works against every "OpenAI-compatible"
//                                  host (OpenAI, Azure, Groq, Together,
//                                  vLLM, OpenRouter, Ollama, …).
//   - providers/openai_responses — Responses /v1/responses. Required for
//                                  GPT-5-family server-side state, reasoning
//                                  summaries, and the built-in tool stack
//                                  (web_search, file_search, code_interpreter).
//
// At v1 this package covers a core slice of the 53-event Responses
// streaming protocol: text output (response.output_text.delta/done),
// function tool calls (response.function_call_arguments.delta/done),
// reasoning summaries (response.reasoning_summary_text.delta/done — mapped
// to llm.ThinkingBlock events), and the response lifecycle envelope
// (response.created / response.completed / response.failed / error).
// MCP, built-in tools, image generation, audio, and the more exotic
// event types are not yet mapped; the SSE parser ignores them gracefully.
package openai_responses

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// Reasoning-effort levels supported by the Responses API on reasoning models
// (GPT-5, o1, o3 families).
type ReasoningEffort string

const (
	ReasoningMinimal ReasoningEffort = "minimal"
	ReasoningLow     ReasoningEffort = "low"
	ReasoningMedium  ReasoningEffort = "medium"
	ReasoningHigh    ReasoningEffort = "high"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Options configures a Provider at construction time. Mirrors the
// providers/openai Options surface so users can switch endpoints with
// minimal code change.
type Options struct {
	APIKey     string
	BaseURL    string       // default "https://api.openai.com/v1"; full /responses URL via URL override
	HTTPClient *http.Client // default http.DefaultClient
	OrgID      string
	Project    string

	// URL, if non-empty, is used verbatim as the responses endpoint instead
	// of BaseURL + "/responses". Use for Azure OpenAI etc.
	URL string

	// Headers are merged into the outgoing request after defaults; user
	// values win. Use for Azure's "api-key:" auth.
	Headers map[string]string

	// ReasoningEffort, when non-empty, is forwarded as the "reasoning.effort"
	// request field. Honored by reasoning-capable models (GPT-5, o1, o3).
	// Ignored by other models.
	ReasoningEffort ReasoningEffort

	// IncludeReasoningSummary, when true, requests reasoning summary
	// streaming (response.reasoning_summary_* events). The summary is
	// surfaced as llm.ThinkingBlock content. Honored by reasoning models.
	IncludeReasoningSummary bool
}

// Provider is the Responses API implementation of llm.LLM.
type Provider struct {
	apiKey                  string
	baseURL                 string
	url                     string
	orgID                   string
	project                 string
	headers                 map[string]string
	reasoningEffort         ReasoningEffort
	includeReasoningSummary bool
	client                  *http.Client
}

// New constructs a Provider. APIKey is required unless Headers supplies an
// alternative authentication header.
func New(opts Options) (*Provider, error) {
	if opts.APIKey == "" && len(opts.Headers) == 0 {
		return nil, errors.New("openai_responses: APIKey or Headers (with auth header) is required")
	}
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = http.DefaultClient
	}
	var headers map[string]string
	if len(opts.Headers) > 0 {
		headers = make(map[string]string, len(opts.Headers))
		for k, v := range opts.Headers {
			headers[k] = v
		}
	}
	return &Provider{
		apiKey:                  opts.APIKey,
		baseURL:                 opts.BaseURL,
		url:                     opts.URL,
		orgID:                   opts.OrgID,
		project:                 opts.Project,
		headers:                 headers,
		reasoningEffort:         opts.ReasoningEffort,
		includeReasoningSummary: opts.IncludeReasoningSummary,
		client:                  opts.HTTPClient,
	}, nil
}

func (p *Provider) endpoint() string {
	if p.url != "" {
		return p.url
	}
	return p.baseURL + "/responses"
}

// Stream issues a streaming request to /responses. Provider errors surface
// through the iterator as *llm.APIError.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, err := buildRequestBody(req, p.reasoningEffort, p.includeReasoningSummary)
		if err != nil {
			yield(nil, fmt.Errorf("openai_responses: build request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), body)
		if err != nil {
			yield(nil, fmt.Errorf("openai_responses: new request: %w", err))
			return
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
		for k, v := range p.headers {
			httpReq.Header.Set(k, v)
		}

		resp, err := p.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("openai_responses: do request: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, &llm.APIError{
				Provider: "openai_responses",
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
