package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	llm "github.com/amit-timalsina/pi-llm-go"
)

type countTokensResponse struct {
	TotalTokens int `json:"totalTokens"`
}

// CountTokens implements llm.TokenCounter against Gemini's
// :countTokens endpoint. Returns the input-token count the request
// would consume if streamed. Bills nothing and does not warm any
// context cache.
//
// The endpoint accepts the same request body as :generateContent
// (contents, systemInstruction, tools, generationConfig) and
// ignores generation-side fields like maxOutputTokens; we reuse
// buildRequestBody verbatim rather than maintaining a parallel
// converter.
func (p *Provider) CountTokens(ctx context.Context, req llm.Request) (int, error) {
	if req.Model == "" {
		return 0, fmt.Errorf("gemini count_tokens: model is required")
	}

	body, err := buildRequestBody(req)
	if err != nil {
		return 0, fmt.Errorf("gemini count_tokens: build request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:countTokens", p.baseURL, req.Model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return 0, fmt.Errorf("gemini count_tokens: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", p.apiKey)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return 0, fmt.Errorf("gemini count_tokens: do request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return 0, &llm.APIError{
			Provider:   "gemini",
			Status:     resp.StatusCode,
			Body:       respBody,
			Inner:      llm.SentinelForStatus(resp.StatusCode),
			RetryAfter: llm.ParseRetryAfter(resp.Header),
		}
	}

	var out countTokensResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, fmt.Errorf("gemini count_tokens: decode response: %w", err)
	}
	return out.TotalTokens, nil
}
