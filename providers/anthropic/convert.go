package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// requestBody is the JSON body sent to /v1/messages.
type requestBody struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	System        string             `json:"system,omitempty"`
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
}

type apiTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type apiThinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

// buildRequestBody serializes a llm.Request into Anthropic's wire format.
// Returns the request body as an io.Reader for http.NewRequestWithContext.
func buildRequestBody(req llm.Request) (io.Reader, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("Model is required")
	}
	if req.MaxTokens <= 0 {
		// Anthropic requires max_tokens. Pick a sane default rather than fail.
		req.MaxTokens = 4096
	}

	body := requestBody{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		System:        req.System,
		Temperature:   req.Temperature,
		StopSequences: req.StopReasons,
		Stream:        true,
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
			return nil, fmt.Errorf("convert message: %w", err)
		}
		body.Messages = append(body.Messages, apiMsg)
	}

	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(body); err != nil {
		return nil, fmt.Errorf("encode body: %w", err)
	}
	return buf, nil
}

// convertOutgoingMessage maps a llm.Message to Anthropic's wire format.
// pi-llm-go uses Role: RoleTool for tool-result messages; Anthropic accepts
// tool results inside Role: "user" messages with tool_result content blocks.
// We translate at the boundary so callers stay on the pi-llm-go shape.
func convertOutgoingMessage(m llm.Message) (apiMessage, error) {
	role := string(m.Role)
	if m.Role == llm.RoleTool {
		role = "user"
	}

	apiMsg := apiMessage{Role: role}
	for _, block := range m.Content {
		ab, err := convertOutgoingBlock(block)
		if err != nil {
			return apiMessage{}, err
		}
		apiMsg.Content = append(apiMsg.Content, ab)
	}
	return apiMsg, nil
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
