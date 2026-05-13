package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// countTokensRequestBody is the JSON body for POST /v1/messages/count_tokens.
// Mirrors requestBody but drops MaxTokens and Stream — the count endpoint
// rejects both. cache_control markers are silently allowed but ignored;
// we omit them to keep the body lean and skip the auto-beta machinery.
type countTokensRequestBody struct {
	Model    string             `json:"model"`
	System   string             `json:"system,omitempty"`
	Messages []apiMessage       `json:"messages"`
	Tools    []apiTool          `json:"tools,omitempty"`
	Thinking *apiThinkingConfig `json:"thinking,omitempty"`
}

type countTokensResponseBody struct {
	InputTokens int `json:"input_tokens"`
}

// CountTokens implements llm.TokenCounter against Anthropic's
// /v1/messages/count_tokens endpoint. Returns the input-token count
// the request would consume if streamed. Does not warm the prompt
// cache and bills nothing.
func (p *Provider) CountTokens(ctx context.Context, req llm.Request) (int, error) {
	if req.Model == "" {
		return 0, fmt.Errorf("anthropic count_tokens: model is required")
	}

	body := countTokensRequestBody{
		Model:  req.Model,
		System: req.System,
	}
	if req.Thinking != nil {
		body.Thinking = &apiThinkingConfig{
			Type:         "enabled",
			BudgetTokens: req.Thinking.BudgetTokens,
		}
	}
	for _, t := range req.Tools {
		body.Tools = append(body.Tools, apiTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	for _, m := range req.Messages {
		apiMsg, err := convertOutgoingMessage(m)
		if err != nil {
			return 0, fmt.Errorf("anthropic count_tokens: convert message: %w", err)
		}
		body.Messages = append(body.Messages, apiMsg)
	}

	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(body); err != nil {
		return 0, fmt.Errorf("anthropic count_tokens: encode body: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/messages/count_tokens", buf)
	if err != nil {
		return 0, fmt.Errorf("anthropic count_tokens: new request: %w", err)
	}
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", p.version)
	httpReq.Header.Set("content-type", "application/json")
	for _, b := range p.beta {
		httpReq.Header.Add("anthropic-beta", b)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return 0, fmt.Errorf("anthropic count_tokens: do request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return 0, &llm.APIError{
			Provider:   "anthropic",
			Status:     resp.StatusCode,
			Body:       respBody,
			Inner:      llm.SentinelForStatus(resp.StatusCode),
			RetryAfter: llm.ParseRetryAfter(resp.Header),
		}
	}

	var out countTokensResponseBody
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, fmt.Errorf("anthropic count_tokens: decode response: %w", err)
	}
	return out.InputTokens, nil
}
