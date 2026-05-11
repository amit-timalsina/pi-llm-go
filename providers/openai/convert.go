package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// requestBody is the wire shape sent to /v1/chat/completions.
//
// max_completion_tokens (not max_tokens) is used: OpenAI deprecated
// max_tokens in late 2024; GPT-5, o1, o3 and similar reject the legacy
// field outright. All modern OpenAI-compatible hosts (OpenAI, Azure
// OpenAI, Groq, Together, vLLM v0.5+, Ollama recent versions) accept
// the new name.
type requestBody struct {
	Model               string         `json:"model"`
	Messages            []apiMessage   `json:"messages"`
	Tools               []apiTool      `json:"tools,omitempty"`
	Stream              bool           `json:"stream"`
	StreamOptions       *streamOptions `json:"stream_options,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
	Stop                []string       `json:"stop,omitempty"`
}

// streamOptions opts into the usage block at the tail of the stream.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// apiMessage is the OpenAI Chat Completions message shape. Role is one of
// "system", "user", "assistant", "tool". ToolCalls populates assistant
// messages that issued function calls; ToolCallID populates tool messages
// (results being fed back).
type apiMessage struct {
	Role       string        `json:"role"`
	Content    string        `json:"content,omitempty"`
	Name       string        `json:"name,omitempty"`
	ToolCalls  []apiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

type apiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // always "function"
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // JSON-as-string per OpenAI shape
	} `json:"function"`
}

type apiTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

func buildRequestBody(req llm.Request) (io.Reader, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("Model is required")
	}

	body := requestBody{
		Model:               req.Model,
		Stream:              true,
		StreamOptions:       &streamOptions{IncludeUsage: true},
		Temperature:         req.Temperature,
		MaxCompletionTokens: req.MaxTokens,
		Stop:                req.StopReasons,
	}

	// System prompt becomes a prepended system message.
	if req.System != "" {
		body.Messages = append(body.Messages, apiMessage{Role: "system", Content: req.System})
	}

	for _, t := range req.Tools {
		var at apiTool
		at.Type = "function"
		at.Function.Name = t.Name
		at.Function.Description = t.Description
		at.Function.Parameters = t.InputSchema
		body.Tools = append(body.Tools, at)
	}

	for _, m := range req.Messages {
		msgs, err := convertOutgoingMessage(m)
		if err != nil {
			return nil, fmt.Errorf("convert message: %w", err)
		}
		body.Messages = append(body.Messages, msgs...)
	}

	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(body); err != nil {
		return nil, fmt.Errorf("encode body: %w", err)
	}
	return buf, nil
}

// convertOutgoingMessage maps a llm.Message to one or more OpenAI messages.
// One llm.Message with multiple ToolResultBlocks expands into N tool
// messages (OpenAI wants one tool message per tool result).
func convertOutgoingMessage(m llm.Message) ([]apiMessage, error) {
	switch m.Role {
	case llm.RoleUser:
		// Concatenate text blocks; ignore non-text content for v1 (no images yet).
		var sb strings.Builder
		for _, b := range m.Content {
			if tb, ok := b.(llm.TextBlock); ok {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(tb.Text)
			}
		}
		return []apiMessage{{Role: "user", Content: sb.String()}}, nil

	case llm.RoleAssistant:
		var sb strings.Builder
		var calls []apiToolCall
		for _, b := range m.Content {
			switch v := b.(type) {
			case llm.TextBlock:
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(v.Text)
			case llm.ThinkingBlock:
				// Drop on send — OpenAI Chat Completions doesn't accept
				// thinking blocks back as assistant content.
			case llm.ToolCallBlock:
				var c apiToolCall
				c.ID = v.ID
				c.Type = "function"
				c.Function.Name = v.Name
				c.Function.Arguments = string(v.Arguments)
				if c.Function.Arguments == "" {
					c.Function.Arguments = "{}"
				}
				calls = append(calls, c)
			default:
				return nil, fmt.Errorf("unsupported assistant block %T", b)
			}
		}
		return []apiMessage{{Role: "assistant", Content: sb.String(), ToolCalls: calls}}, nil

	case llm.RoleTool:
		// Each tool-result block becomes its own tool message.
		var out []apiMessage
		for _, b := range m.Content {
			tr, ok := b.(llm.ToolResultBlock)
			if !ok {
				return nil, fmt.Errorf("tool message contains non-result block %T", b)
			}
			out = append(out, apiMessage{
				Role:       "tool",
				ToolCallID: tr.ToolCallID,
				Content:    tr.Content,
			})
		}
		return out, nil

	default:
		return nil, fmt.Errorf("unknown role %q", m.Role)
	}
}

// stopReasonFromAPI maps OpenAI's finish_reason to llm.StopReason.
// content_filter is treated as a generic provider error path — callers
// shouldn't need to special-case it in normal flows.
func stopReasonFromAPI(s string) (llm.StopReason, error) {
	switch s {
	case "stop", "":
		return llm.StopReasonEnd, nil
	case "length":
		return llm.StopReasonMaxTokens, nil
	case "tool_calls", "function_call":
		return llm.StopReasonToolUse, nil
	case "stop_sequence":
		return llm.StopReasonStop, nil
	case "content_filter":
		return llm.StopReasonEnd, fmt.Errorf("content filtered by provider")
	default:
		return llm.StopReasonEnd, nil
	}
}
