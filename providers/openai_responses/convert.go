package openai_responses

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// requestBody is the wire shape posted to /v1/responses. Subset of the full
// spec — covers what pi-llm-go currently surfaces (text input/output,
// function tool calls, reasoning effort).
type requestBody struct {
	Model        string         `json:"model"`
	Input        []inputItem    `json:"input"`
	Instructions string         `json:"instructions,omitempty"`
	Tools        []apiTool      `json:"tools,omitempty"`
	Stream       bool           `json:"stream"`
	Reasoning    *reasoningOpts `json:"reasoning,omitempty"`
	Temperature  *float64       `json:"temperature,omitempty"`
	MaxOutput    int            `json:"max_output_tokens,omitempty"`
	Include      []string       `json:"include,omitempty"`
}

type reasoningOpts struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

// inputItem is one entry in the Responses API input array. Items can be
// messages (with role+content), function call outputs (tool results), or
// other types. We emit message + function_call_output here.
type inputItem struct {
	Type       string             `json:"type"`               // "message" | "function_call_output"
	Role       string             `json:"role,omitempty"`     // for message: "user" | "assistant" | "system"
	Content    []inputContentPart `json:"content,omitempty"`  // for message
	CallID     string             `json:"call_id,omitempty"`  // for function_call_output
	Output     string             `json:"output,omitempty"`   // for function_call_output
}

// inputContentPart is one piece of a message's content. Type indicates the
// kind: "input_text" for user-written prompts, "output_text" for replayed
// assistant text.
type inputContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type apiTool struct {
	Type        string          `json:"type"`        // "function"
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// buildRequestBody serializes a llm.Request into Responses API format.
func buildRequestBody(req llm.Request, effort ReasoningEffort, includeReasoningSummary bool) (io.Reader, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("Model is required")
	}
	body := requestBody{
		Model:        req.Model,
		Instructions: req.System,
		Stream:       true,
		Temperature:  req.Temperature,
		MaxOutput:    req.MaxTokens,
	}

	if effort != "" || includeReasoningSummary {
		body.Reasoning = &reasoningOpts{Effort: string(effort)}
		if includeReasoningSummary {
			body.Reasoning.Summary = "auto"
			body.Include = append(body.Include, "reasoning.encrypted_content")
		}
	}

	for _, t := range req.Tools {
		body.Tools = append(body.Tools, apiTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}

	for _, m := range req.Messages {
		items, err := convertOutgoingMessage(m)
		if err != nil {
			return nil, fmt.Errorf("convert message: %w", err)
		}
		body.Input = append(body.Input, items...)
	}

	buf := &bytes.Buffer{}
	enc := json.NewEncoder(buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(body); err != nil {
		return nil, fmt.Errorf("encode body: %w", err)
	}
	return buf, nil
}

// convertOutgoingMessage maps a llm.Message into one or more inputItems.
// Tool-result messages expand into one function_call_output per
// ToolResultBlock.
func convertOutgoingMessage(m llm.Message) ([]inputItem, error) {
	switch m.Role {
	case llm.RoleUser:
		var sb strings.Builder
		for _, b := range m.Content {
			if tb, ok := b.(llm.TextBlock); ok {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(tb.Text)
			}
		}
		return []inputItem{{
			Type: "message",
			Role: "user",
			Content: []inputContentPart{
				{Type: "input_text", Text: sb.String()},
			},
		}}, nil

	case llm.RoleAssistant:
		// Replayed assistant text only — tool calls round-trip via the API
		// item ids on the server. Thinking blocks are not currently
		// round-tripped on the Responses API in this provider.
		var sb strings.Builder
		for _, b := range m.Content {
			if tb, ok := b.(llm.TextBlock); ok {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(tb.Text)
			}
		}
		if sb.Len() == 0 {
			return nil, nil
		}
		return []inputItem{{
			Type: "message",
			Role: "assistant",
			Content: []inputContentPart{
				{Type: "output_text", Text: sb.String()},
			},
		}}, nil

	case llm.RoleTool:
		var out []inputItem
		for _, b := range m.Content {
			tr, ok := b.(llm.ToolResultBlock)
			if !ok {
				return nil, fmt.Errorf("tool message contains non-result block %T", b)
			}
			out = append(out, inputItem{
				Type:   "function_call_output",
				CallID: tr.ToolCallID,
				Output: tr.Content,
			})
		}
		return out, nil

	default:
		return nil, fmt.Errorf("unknown role %q", m.Role)
	}
}

// stopReasonFromStatus maps Responses-API status / incomplete_details to
// our normalized StopReason.
func stopReasonFromStatus(status, incompleteReason string) llm.StopReason {
	switch status {
	case "completed":
		return llm.StopReasonEnd
	case "incomplete":
		switch incompleteReason {
		case "max_output_tokens":
			return llm.StopReasonMaxTokens
		default:
			return llm.StopReasonEnd
		}
	default:
		return llm.StopReasonEnd
	}
}
