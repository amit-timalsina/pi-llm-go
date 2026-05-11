// Package anthropic is the Anthropic Messages provider for pi-llm-go.
//
// It implements llm.LLM against https://api.anthropic.com/v1/messages
// using raw net/http for fine-grained control of the streaming response.
//
//	p, err := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
//	if err != nil { ... }
//	for event, err := range p.Stream(ctx, llm.Request{Model: anthropic.ClaudeOpus4_7, ...}) {
//	    ...
//	}
//
// Honors llm.Request.Thinking. Surfaces ToolCallBlock content when the
// model issues tool calls; the caller (or the pi-agent-go loop) is
// responsible for executing tools and feeding ToolResultBlock messages
// back on the next call.
package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// Canonical Claude model aliases as of May 2026. The Anthropic API accepts
// these dateless aliases and pins them to the current snapshot; for
// reproducible runs, use the dated ID directly (e.g. "claude-haiku-4-5-20251001").
//
// These are convenience constants only — llm.Request.Model takes any string.
const (
	ClaudeOpus4_7   = "claude-opus-4-7"
	ClaudeSonnet4_6 = "claude-sonnet-4-6"
	ClaudeHaiku4_5  = "claude-haiku-4-5"
)

// Default endpoints and versions.
const (
	defaultBaseURL = "https://api.anthropic.com"
	defaultVersion = "2023-06-01"
)

// Options configures a Provider at construction time.
type Options struct {
	APIKey     string       // required; falls back to ANTHROPIC_API_KEY in os.Getenv
	BaseURL    string       // default "https://api.anthropic.com"
	Version    string       // default "2023-06-01"
	HTTPClient *http.Client // default http.DefaultClient
	Beta       []string     // optional anthropic-beta header values
}

// Provider is the Anthropic implementation of llm.LLM. Safe for concurrent
// use across goroutines. Each Stream call issues an independent HTTP
// request; the returned iterator is single-consumer.
type Provider struct {
	apiKey  string
	baseURL string
	version string
	beta    []string
	client  *http.Client
}

// New constructs a Provider. Returns an error if APIKey is empty after
// considering Options.APIKey and the ANTHROPIC_API_KEY environment
// variable in main isn't checked by this package — callers are expected to
// resolve their own env lookup before constructing.
func New(opts Options) (*Provider, error) {
	if opts.APIKey == "" {
		return nil, errors.New("anthropic: APIKey is required")
	}
	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}
	if opts.Version == "" {
		opts.Version = defaultVersion
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = http.DefaultClient
	}
	return &Provider{
		apiKey:  opts.APIKey,
		baseURL: opts.BaseURL,
		version: opts.Version,
		beta:    append([]string{}, opts.Beta...),
		client:  opts.HTTPClient,
	}, nil
}

// Stream issues a streaming completion request. The iterator yields events
// in order; an error value terminates iteration. Provider HTTP errors are
// wrapped in *llm.APIError so callers can use errors.Is on the sentinels.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, err := buildRequestBody(req)
		if err != nil {
			yield(nil, fmt.Errorf("anthropic: build request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/messages", body)
		if err != nil {
			yield(nil, fmt.Errorf("anthropic: new request: %w", err))
			return
		}
		httpReq.Header.Set("x-api-key", p.apiKey)
		httpReq.Header.Set("anthropic-version", p.version)
		httpReq.Header.Set("content-type", "application/json")
		httpReq.Header.Set("accept", "text/event-stream")
		for _, b := range p.beta {
			httpReq.Header.Add("anthropic-beta", b)
		}

		resp, err := p.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("anthropic: do request: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, &llm.APIError{
				Provider: "anthropic",
				Status:   resp.StatusCode,
				Body:     respBody,
				Inner:    llm.SentinelForStatus(resp.StatusCode),
			})
			return
		}

		// streamEvents drains the SSE response, translating to llm.StreamEvent.
		// It writes to a channel of (event, err) pairs that the iterator
		// drains. Decoupling lets us treat callback-style SSE reads as an
		// iterator while still supporting yield-based early termination.
		streamErr := newStreamDecoder().decode(resp.Body, yield)
		if streamErr != nil && !errors.Is(streamErr, errIterationStopped) {
			yield(nil, streamErr)
		}
	}
}
