package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// extendedCacheTTLBeta is the Anthropic beta header required for 1h cache
// TTL. Auto-applied by buildRequestBody when any breakpoint declares
// TTL == "1h". See https://docs.claude.com/en/docs/build-with-claude/prompt-caching
const extendedCacheTTLBeta = "extended-cache-ttl-2025-04-11"

// requestBody is the JSON body sent to /v1/messages.
//
// System is `any` (not `string`) because Anthropic accepts two shapes:
//
//   - "system": "<plain string>"                  — when no cache marker
//   - "system": [{type: "text", text: "...",
//     cache_control: {...}}]          — when SystemCacheControl set
//
// We switch shapes at build time based on Request.SystemCacheControl.
type requestBody struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	System        any                `json:"system,omitempty"`
	Messages      []apiMessage       `json:"messages"`
	Tools         []apiTool          `json:"tools,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Thinking      *apiThinkingConfig `json:"thinking,omitempty"`
	Stream        bool               `json:"stream"`
}

type apiMessage struct {
	Role    string     `json:"role"`
	Content []apiBlock `json:"content"`
}

// apiCacheControl is the on-wire shape Anthropic accepts. Type is always
// "ephemeral" today; TTL is "" (default ~5min) or "1h" (extended, beta).
type apiCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// apiBlock is the discriminated-union shape Anthropic accepts in message
// content. We use a single struct with all possible fields and rely on
// "omitempty" tags + Type to discriminate.
type apiBlock struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// thinking
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	// For text-only tool results, Content is a single string. For richer
	// results (future: images) it becomes an array; v1 always emits string.
	Content string `json:"content,omitempty"`
	IsError bool   `json:"is_error,omitempty"`

	// Optional cache breakpoint. Honored on any block type Anthropic
	// supports caching on (text, thinking, tool_use, tool_result).
	CacheControl *apiCacheControl `json:"cache_control,omitempty"`
}

type apiTool struct {
	Name         string           `json:"name"`
	Description  string           `json:"description,omitempty"`
	InputSchema  json.RawMessage  `json:"input_schema"`
	CacheControl *apiCacheControl `json:"cache_control,omitempty"`
}

type apiThinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

// buildRequestBody serializes a llm.Request into Anthropic's wire format.
// Returns the body, any beta-header values that must be auto-applied
// (e.g. extended-cache-ttl when any breakpoint declares TTL "1h"), and
// an error.
func buildRequestBody(req llm.Request) (io.Reader, []string, error) {
	if req.Model == "" {
		return nil, nil, fmt.Errorf("model is required")
	}
	if req.MaxTokens <= 0 {
		// Anthropic requires max_tokens. Pick a sane default rather than fail.
		req.MaxTokens = 4096
	}

	// Track whether any breakpoint requests the extended 1h TTL; if so,
	// auto-add the beta header. Centralizing the check here keeps the
	// caller from having to wire it manually.
	var needsExtendedTTL bool
	noteCacheControl := func(cc *llm.CacheControl) {
		if cc != nil && cc.TTL == "1h" {
			needsExtendedTTL = true
		}
	}

	body := requestBody{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		StopSequences: req.StopReasons,
		Stream:        true,
	}

	// System: plain string by default; structured array when caching.
	if req.SystemCacheControl != nil && req.System != "" {
		noteCacheControl(req.SystemCacheControl)
		body.System = []apiBlock{{
			Type:         "text",
			Text:         req.System,
			CacheControl: toAPICacheControl(req.SystemCacheControl),
		}}
	} else if req.System != "" {
		body.System = req.System
	}

	if req.Thinking != nil {
		body.Thinking = &apiThinkingConfig{
			Type:         "enabled",
			BudgetTokens: req.Thinking.BudgetTokens,
		}
	}

	// Tools: per-tool CacheControl. If Request.ToolsCacheControl is set
	// AND the last tool doesn't already have its own CacheControl, the
	// shortcut applies to that last tool.
	for _, t := range req.Tools {
		noteCacheControl(t.CacheControl)
		body.Tools = append(body.Tools, apiTool{
			Name:         t.Name,
			Description:  t.Description,
			InputSchema:  t.InputSchema,
			CacheControl: toAPICacheControl(t.CacheControl),
		})
	}
	if req.ToolsCacheControl != nil && len(body.Tools) > 0 {
		noteCacheControl(req.ToolsCacheControl)
		last := len(body.Tools) - 1
		if body.Tools[last].CacheControl == nil {
			body.Tools[last].CacheControl = toAPICacheControl(req.ToolsCacheControl)
		}
	}

	for _, m := range req.Messages {
		apiMsg, err := convertOutgoingMessage(m, noteCacheControl)
		if err != nil {
			return nil, nil, fmt.Errorf("convert message: %w", err)
		}
		body.Messages = append(body.Messages, apiMsg)
	}

	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(body); err != nil {
		return nil, nil, fmt.Errorf("encode body: %w", err)
	}

	var autoBeta []string
	if needsExtendedTTL {
		autoBeta = append(autoBeta, extendedCacheTTLBeta)
	}
	return buf, autoBeta, nil
}

// convertOutgoingMessage maps a llm.Message to Anthropic's wire format.
// pi-llm-go uses Role: RoleTool for tool-result messages; Anthropic accepts
// tool results inside Role: "user" messages with tool_result content blocks.
// We translate at the boundary so callers stay on the pi-llm-go shape.
//
// noteCC is called for every CacheControl encountered on a block, so the
// outer build pass can detect features that require a beta header (e.g.
// 1h TTL).
func convertOutgoingMessage(m llm.Message, noteCC func(*llm.CacheControl)) (apiMessage, error) {
	role := string(m.Role)
	if m.Role == llm.RoleTool {
		role = "user"
	}

	apiMsg := apiMessage{Role: role}
	for _, block := range m.Content {
		ab, err := convertOutgoingBlock(block, noteCC)
		if err != nil {
			return apiMessage{}, err
		}
		apiMsg.Content = append(apiMsg.Content, ab)
	}
	return apiMsg, nil
}

func convertOutgoingBlock(b llm.Block, noteCC func(*llm.CacheControl)) (apiBlock, error) {
	switch v := b.(type) {
	case llm.TextBlock:
		noteCC(v.CacheControl)
		return apiBlock{
			Type:         "text",
			Text:         v.Text,
			CacheControl: toAPICacheControl(v.CacheControl),
		}, nil
	case llm.ThinkingBlock:
		// Redacted thinking has empty Thinking but a Signature payload; the
		// wire type for those is "redacted_thinking" with a "data" field, but
		// at v1 we elide that complexity — round-tripped thinking always uses
		// the "thinking" type with both fields. Real round-tripping of
		// redacted blocks is post-v1.
		noteCC(v.CacheControl)
		return apiBlock{
			Type:         "thinking",
			Thinking:     v.Thinking,
			Signature:    v.Signature,
			CacheControl: toAPICacheControl(v.CacheControl),
		}, nil
	case llm.ToolCallBlock:
		noteCC(v.CacheControl)
		return apiBlock{
			Type:         "tool_use",
			ID:           v.ID,
			Name:         v.Name,
			Input:        v.Arguments,
			CacheControl: toAPICacheControl(v.CacheControl),
		}, nil
	case llm.ToolResultBlock:
		noteCC(v.CacheControl)
		return apiBlock{
			Type:         "tool_result",
			ToolUseID:    v.ToolCallID,
			Content:      v.Content,
			IsError:      v.IsError,
			CacheControl: toAPICacheControl(v.CacheControl),
		}, nil
	default:
		return apiBlock{}, fmt.Errorf("unsupported block type %T", b)
	}
}

func toAPICacheControl(cc *llm.CacheControl) *apiCacheControl {
	if cc == nil {
		return nil
	}
	t := cc.Type
	if t == "" {
		// Default to "ephemeral" if the caller used a zero-value
		// CacheControl{} or only set TTL — Anthropic's only mode today.
		t = "ephemeral"
	}
	return &apiCacheControl{Type: t, TTL: cc.TTL}
}

// stopReasonFromAPI maps Anthropic's stop_reason strings to llm.StopReason.
// "pause_turn" and "refusal" are treated as ends — the assistant content
// carries the explanation; callers inspect it if they need richer
// disposition.
func stopReasonFromAPI(s string) llm.StopReason {
	switch s {
	case "end_turn", "pause_turn", "refusal":
		return llm.StopReasonEnd
	case "max_tokens":
		return llm.StopReasonMaxTokens
	case "tool_use":
		return llm.StopReasonToolUse
	case "stop_sequence":
		return llm.StopReasonStop
	default:
		return llm.StopReasonEnd
	}
}
