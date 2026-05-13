package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// extendedCacheTTLBeta is the Anthropic beta header required for 1h cache
// TTL. Auto-applied by buildRequestBody when CacheRetention == "long".
// See https://docs.claude.com/en/docs/build-with-claude/prompt-caching
const extendedCacheTTLBeta = "extended-cache-ttl-2025-04-11"

// requestBody is the JSON body sent to /v1/messages.
//
// System is `any` (not `string`) because Anthropic accepts two shapes:
//
//   - "system": "<plain string>"                  — when no cache marker
//   - "system": [{type: "text", text: "...",
//     cache_control: {...}}]          — when CacheRetention places a marker
//
// We switch shapes at build time based on Request.CacheRetention.
type requestBody struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	System        any                `json:"system,omitempty"`
	Messages      []apiMessage       `json:"messages"`
	Tools         []apiTool          `json:"tools,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Thinking      *apiThinkingConfig `json:"thinking,omitempty"`
	// OutputConfig carries the adaptive-thinking effort enum on
	// Opus 4.6+ models. It's a TOP-LEVEL request field (not nested
	// under thinking) per Anthropic's wire contract, separate from
	// the legacy budget_tokens path that lives inside thinking.
	OutputConfig *apiOutputConfig `json:"output_config,omitempty"`
	Stream       bool             `json:"stream"`
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

	// image — present iff Type == "image".
	Source *apiImageSource `json:"source,omitempty"`

	// Optional cache breakpoint. Auto-placed by buildRequestBody based on
	// Request.CacheRetention; callers don't set this directly.
	CacheControl *apiCacheControl `json:"cache_control,omitempty"`
}

// apiImageSource is Anthropic's nested image source descriptor.
// At v0.3.0 we only emit Type="base64". The URL variant
// ({type:"url", url:"https://..."}) is a future addition.
type apiImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type apiTool struct {
	Name         string           `json:"name"`
	Description  string           `json:"description,omitempty"`
	InputSchema  json.RawMessage  `json:"input_schema"`
	CacheControl *apiCacheControl `json:"cache_control,omitempty"`
}

// apiThinkingConfig is the on-wire shape for the `thinking` field.
// Two flavors:
//
//   - Adaptive (Opus 4.6+, REQUIRED on 4.7+): {"type":"adaptive"} —
//     BudgetTokens MUST be omitted; Opus 4.7 returns 400 if present.
//   - Manual (Opus 4.5- / Sonnet 3.7, deprecated on 4.6 family):
//     {"type":"enabled", "budget_tokens": N}.
//
// `budget_tokens` is tagged `omitempty` so the adaptive shape doesn't
// leak a `budget_tokens: 0` field that Opus 4.7 would reject.
type apiThinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// applyThinkingConfig returns the on-wire thinking / output_config
// pair for a caller-supplied llm.ThinkingConfig. Both returned
// pointers may be nil (no thinking requested), or only the first
// (manual mode), or both (adaptive mode).
//
// Single source of truth for the dispatch — both Stream's
// buildRequestBody and CountTokens's doCountTokens delegate to this
// so they can't silently diverge.
//
// Dispatch:
//   - t == nil OR both fields zero            → (nil, nil): no thinking
//   - t.Effort != ""                          → adaptive shape; Effort wins
//     even if t.BudgetTokens > 0 (lets callers pre-set both during
//     a migration)
//   - t.BudgetTokens > 0 (Effort empty)       → manual shape
func applyThinkingConfig(t *llm.ThinkingConfig) (*apiThinkingConfig, *apiOutputConfig) {
	if t == nil {
		return nil, nil
	}
	switch {
	case t.Effort != "":
		return &apiThinkingConfig{Type: "adaptive"},
			&apiOutputConfig{Effort: string(t.Effort)}
	case t.BudgetTokens > 0:
		return &apiThinkingConfig{
			Type:         "enabled",
			BudgetTokens: t.BudgetTokens,
		}, nil
	}
	return nil, nil
}

// apiOutputConfig is the top-level `output_config` request field that
// carries the adaptive-thinking effort enum. Separate from the
// `thinking` block per Anthropic's wire contract — `effort` lives at
// the request root, not under thinking.
type apiOutputConfig struct {
	Effort string `json:"effort,omitempty"`
}

// buildRequestBody serializes a llm.Request into Anthropic's wire format.
// Returns the body, any beta-header values that must be auto-applied
// (e.g. extended-cache-ttl when CacheRetention == "long"), and an error.
//
// Cache placement: when req.CacheRetention is "short" or "long", we auto-
// place ephemeral cache_control markers at the static prefix boundary —
// the System prompt's trailing block, the final Tool in Tools, and the
// last text block of the most recent user message. This matches Mario
// Zechner's pi-ai design: the caller picks a retention tier, the provider
// owns where the markers go.
func buildRequestBody(req llm.Request) (io.Reader, []string, error) {
	if req.Model == "" {
		return nil, nil, fmt.Errorf("model is required")
	}
	if req.MaxTokens <= 0 {
		// Anthropic requires max_tokens. Pick a sane default rather than fail.
		req.MaxTokens = 4096
	}

	marker := cacheMarker(req.CacheRetention)

	body := requestBody{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		StopSequences: req.StopReasons,
		Stream:        true,
	}

	// System: plain string by default; structured array when caching.
	if marker != nil && req.System != "" {
		body.System = []apiBlock{{
			Type:         "text",
			Text:         req.System,
			CacheControl: marker,
		}}
	} else if req.System != "" {
		body.System = req.System
	}

	body.Thinking, body.OutputConfig = applyThinkingConfig(req.Thinking)

	for _, t := range req.Tools {
		body.Tools = append(body.Tools, apiTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	if marker != nil && len(body.Tools) > 0 {
		last := len(body.Tools) - 1
		body.Tools[last].CacheControl = marker
	}

	for _, m := range req.Messages {
		apiMsg, err := convertOutgoingMessage(m)
		if err != nil {
			return nil, nil, fmt.Errorf("convert message: %w", err)
		}
		body.Messages = append(body.Messages, apiMsg)
	}

	// Place a marker on the last text block of the most recent user message.
	// We walk backwards because the "most recent user message" is the most
	// recently-mutated section of the prompt — caching up to and including
	// it captures the largest stable prefix for the next turn.
	if marker != nil {
		placeUserCacheMarker(body.Messages, marker)
	}

	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(body); err != nil {
		return nil, nil, fmt.Errorf("encode body: %w", err)
	}

	var autoBeta []string
	if req.CacheRetention == llm.CacheRetentionLong {
		autoBeta = append(autoBeta, extendedCacheTTLBeta)
	}
	return buf, autoBeta, nil
}

// cacheMarker returns the wire-level cache_control value for the given
// retention tier, or nil if no marker should be placed.
func cacheMarker(r llm.CacheRetention) *apiCacheControl {
	switch r {
	case llm.CacheRetentionShort:
		return &apiCacheControl{Type: "ephemeral"}
	case llm.CacheRetentionLong:
		return &apiCacheControl{Type: "ephemeral", TTL: "1h"}
	default:
		return nil
	}
}

// placeUserCacheMarker attaches the cache marker to the last block of
// the most recent user-role message on the wire. Anthropic accepts
// cache_control on every block type we emit (text, tool_use, tool_result,
// thinking), so the placement is type-agnostic — that matches Mario's
// pi-ai cacheRetention: by including the trailing tool_result block in
// the cached prefix, subsequent calls in a tool loop reuse the full
// round-trip from the cache instead of re-billing it as fresh input.
// No-op if no user-role message exists or the last one has no content.
func placeUserCacheMarker(messages []apiMessage, marker *apiCacheControl) {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "user" {
			continue
		}
		content := messages[i].Content
		if len(content) == 0 {
			return
		}
		content[len(content)-1].CacheControl = marker
		return
	}
}

// convertOutgoingMessage maps a llm.Message to Anthropic's wire format.
// pi-llm-go uses Role: RoleTool for tool-result messages; Anthropic accepts
// tool results inside Role: "user" messages with tool_result content blocks.
// We translate at the boundary so callers stay on the pi-llm-go shape.
//
// User-role messages that contain images but no text get a synthetic
// "(see attached image)" text block prepended. Anthropic's API works
// best when image input is accompanied by at least one text block;
// this matches Mario Zechner's pi-ai placeholder convention.
//
// ImageBlock is allowed only on user-role messages. Assistant- and
// tool-role ImageBlocks are rejected at this boundary because (a)
// Anthropic does not accept images in those roles on the wire and (b)
// silently dropping them would make a model-bug look like a library-bug.
func convertOutgoingMessage(m llm.Message) (apiMessage, error) {
	role := string(m.Role)
	if m.Role == llm.RoleTool {
		role = "user"
	}

	if m.Role != llm.RoleUser {
		for _, b := range m.Content {
			if _, ok := b.(llm.ImageBlock); ok {
				return apiMessage{}, fmt.Errorf("anthropic: ImageBlock is only valid on user-role messages (got role %q)", m.Role)
			}
		}
	}

	apiMsg := apiMessage{Role: role}
	for _, block := range m.Content {
		ab, err := convertOutgoingBlock(block)
		if err != nil {
			return apiMessage{}, err
		}
		apiMsg.Content = append(apiMsg.Content, ab)
	}

	if role == "user" && needsImagePlaceholder(apiMsg.Content) {
		placeholder := apiBlock{Type: "text", Text: "(see attached image)"}
		apiMsg.Content = append([]apiBlock{placeholder}, apiMsg.Content...)
	}
	return apiMsg, nil
}

// needsImagePlaceholder reports whether content contains at least one
// "image" block and no "text" block. Anthropic accepts image-only
// content but the model performs noticeably better with at least one
// accompanying text block, so we inject a placeholder. Matches Mario's
// behavior in pi-ai/providers/anthropic.ts:convertContentBlocks.
func needsImagePlaceholder(content []apiBlock) bool {
	hasImage, hasText := false, false
	for _, b := range content {
		switch b.Type {
		case "image":
			hasImage = true
		case "text":
			hasText = true
		}
	}
	return hasImage && !hasText
}

func convertOutgoingBlock(b llm.Block) (apiBlock, error) {
	switch v := b.(type) {
	case llm.TextBlock:
		return apiBlock{Type: "text", Text: v.Text}, nil
	case llm.ThinkingBlock:
		// Redacted thinking has empty Thinking but a Signature payload; the
		// wire type for those is "redacted_thinking" with a "data" field, but
		// at v1 we elide that complexity — round-tripped thinking always uses
		// the "thinking" type with both fields. Real round-tripping of
		// redacted blocks is post-v1.
		return apiBlock{
			Type:      "thinking",
			Thinking:  v.Thinking,
			Signature: v.Signature,
		}, nil
	case llm.ToolCallBlock:
		return apiBlock{
			Type:  "tool_use",
			ID:    v.ID,
			Name:  v.Name,
			Input: v.Arguments,
		}, nil
	case llm.ToolResultBlock:
		return apiBlock{
			Type:      "tool_result",
			ToolUseID: v.ToolCallID,
			Content:   v.Content,
			IsError:   v.IsError,
		}, nil
	case llm.ImageBlock:
		if err := v.Validate(); err != nil {
			return apiBlock{}, fmt.Errorf("anthropic: %w", err)
		}
		return apiBlock{
			Type: "image",
			Source: &apiImageSource{
				Type:      "base64",
				MediaType: v.MimeType,
				Data:      v.Data,
			},
		}, nil
	case llm.VideoBlock:
		return apiBlock{}, fmt.Errorf("anthropic: VideoBlock is not supported; Anthropic Claude has no native video input. Extract frames client-side and submit as ImageBlocks (see ai.google.dev for Gemini's native video support)")
	default:
		return apiBlock{}, fmt.Errorf("unsupported block type %T", b)
	}
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
