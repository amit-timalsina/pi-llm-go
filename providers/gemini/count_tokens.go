package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// countTokensRequest is the on-wire shape for POST :countTokens.
//
// The endpoint accepts two body shapes:
//
//  1. {"contents": [...]}                          — counts the contents only.
//  2. {"generateContentRequest": {model, contents,
//     systemInstruction, tools, generationConfig}} — counts the FULL request.
//
// pi-llm-go uses shape (2) because shape (1) ignores the system
// instruction and tool definitions, which both contribute to the
// real input-token count Gemini bills against. A live-smoke check
// against the v1beta endpoint surfaced this — shape (1) under-counts
// any request with a non-empty System.
//
// The nested model field MUST be the fully-qualified resource name
// ("models/gemini-2.5-flash"), NOT the bare id. Mismatching this
// against the URL's model path makes the endpoint return a 400.
type countTokensRequest struct {
	GenerateContentRequest *generateContentForCount `json:"generateContentRequest"`
}

type generateContentForCount struct {
	Model             string            `json:"model"`
	Contents          []apiContent      `json:"contents"`
	SystemInstruction *apiSystem        `json:"systemInstruction,omitempty"`
	Tools             []apiTool         `json:"tools,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

type countTokensResponse struct {
	TotalTokens int `json:"totalTokens"`
}

// CountTokens implements llm.TokenCounter against Gemini's
// :countTokens endpoint. Returns the input-token count the request
// would consume if streamed. Bills nothing and does not warm any
// context cache.
//
// Wraps the request body in {"generateContentRequest": {...}} so the
// system instruction and tool definitions contribute to the count.
// Without the wrapper, the endpoint silently ignores those fields
// and returns a low-balled count.
func (p *Provider) CountTokens(ctx context.Context, req llm.Request) (int, error) {
	if req.Model == "" {
		return 0, fmt.Errorf("gemini count_tokens: model is required")
	}

	inner, err := buildCountInnerBody(req)
	if err != nil {
		return 0, fmt.Errorf("gemini count_tokens: build request: %w", err)
	}

	wrapped := countTokensRequest{GenerateContentRequest: inner}
	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(wrapped); err != nil {
		return 0, fmt.Errorf("gemini count_tokens: encode body: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:countTokens", p.baseURL, req.Model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, buf)
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

// buildCountInnerBody constructs the inner generateContentRequest shape
// for the count_tokens endpoint. Shares converter helpers with the
// streaming buildRequestBody (convertOutgoingMessage etc.) so message
// shaping stays consistent across the two endpoints.
func buildCountInnerBody(req llm.Request) (*generateContentForCount, error) {
	out := &generateContentForCount{
		Model: "models/" + req.Model,
	}

	if req.System != "" {
		out.SystemInstruction = &apiSystem{
			Parts: []apiPart{{Text: req.System}},
		}
	}

	gc := &generationConfig{
		Temperature:     req.Temperature,
		MaxOutputTokens: req.MaxTokens,
		StopSequences:   req.StopReasons,
	}
	if req.Thinking != nil {
		gc.ThinkingConfig = &thinkingCfg{
			ThinkingBudget:  req.Thinking.BudgetTokens,
			IncludeThoughts: true,
		}
	}
	if hasGenConfig(gc) {
		out.GenerationConfig = gc
	}

	if len(req.Tools) > 0 {
		decls := make([]apiFunctionDecl, 0, len(req.Tools))
		for _, t := range req.Tools {
			decls = append(decls, apiFunctionDecl{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			})
		}
		out.Tools = []apiTool{{FunctionDeclarations: decls}}
	}

	// Pre-walk to build a tool-call-id -> function-name index. Same
	// pattern as buildRequestBody (see its comment for rationale).
	toolNameByID := map[string]string{}
	for _, m := range req.Messages {
		if m.Role != llm.RoleAssistant {
			continue
		}
		for _, b := range m.Content {
			if tc, ok := b.(llm.ToolCallBlock); ok && tc.ID != "" {
				toolNameByID[tc.ID] = tc.Name
			}
		}
	}

	for _, m := range req.Messages {
		converted, err := convertOutgoingMessage(m, toolNameByID)
		if err != nil {
			return nil, fmt.Errorf("convert message: %w", err)
		}
		if m.Role == llm.RoleTool && len(out.Contents) > 0 {
			last := &out.Contents[len(out.Contents)-1]
			if last.Role == "user" {
				last.Parts = append(last.Parts, converted.Parts...)
				continue
			}
		}
		out.Contents = append(out.Contents, converted)
	}

	return out, nil
}
