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
	ToolChoice          any            `json:"tool_choice,omitempty"`
	Stream              bool           `json:"stream"`
	StreamOptions       *streamOptions `json:"stream_options,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
	Stop                []string       `json:"stop,omitempty"`
}

// apiToolChoiceFunc is the object shape OpenAI uses when forcing a
// specific named function: {"type":"function","function":{"name":"..."}}.
// The simple string forms ("auto" / "required" / "none") serialize as
// bare strings on the requestBody.ToolChoice `any` field.
type apiToolChoiceFunc struct {
	Type     string `json:"type"` // always "function"
	Function struct {
		Name string `json:"name"`
	} `json:"function"`
}

// toAPIToolChoice maps llm.ToolChoice to OpenAI's wire shape. Returns
// nil when t is nil (preserves the provider default), a bare string
// for auto/required/none, or an apiToolChoiceFunc for the named-tool
// case.
//
// Keyword mapping vs pi-llm-go's neutral enum:
//
//	llm.ToolChoiceAuto → "auto"
//	llm.ToolChoiceAny  → "required"   (OpenAI renames it)
//	llm.ToolChoiceNone → "none"
//	llm.ToolChoiceTool → {"type":"function","function":{"name":<Name>}}
func toAPIToolChoice(t *llm.ToolChoice) (any, error) {
	if t == nil {
		return nil, nil
	}
	switch t.Type {
	case llm.ToolChoiceAuto:
		return "auto", nil
	case llm.ToolChoiceAny:
		return "required", nil
	case llm.ToolChoiceNone:
		return "none", nil
	case llm.ToolChoiceTool:
		if t.Name == "" {
			return nil, fmt.Errorf("tool_choice: Type=Tool requires Name")
		}
		out := apiToolChoiceFunc{Type: "function"}
		out.Function.Name = t.Name
		return out, nil
	default:
		return nil, fmt.Errorf("tool_choice: unknown Type %q", t.Type)
	}
}

// streamOptions opts into the usage block at the tail of the stream.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// apiMessage is the OpenAI Chat Completions message shape. Role is one of
// "system", "user", "assistant", "tool". ToolCalls populates assistant
// messages that issued function calls; ToolCallID populates tool messages
// (results being fed back).
//
// Content is `any` because OpenAI accepts two shapes:
//
//   - "content": "<plain string>"               — text-only (preferred when
//     no images, smaller wire format)
//   - "content": [ {type:"text", text:"..."},
//     {type:"image_url", image_url:{url:"data:..."}} ]  — multimodal
//
// convertOutgoingMessage emits the array form iff the message contains
// at least one ImageBlock; otherwise the string form for back-compat
// with hosts that only support the legacy shape.
type apiMessage struct {
	Role       string        `json:"role"`
	Content    any           `json:"content,omitempty"`
	Name       string        `json:"name,omitempty"`
	ToolCalls  []apiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

// apiContentPart is one element of the multimodal-array content shape.
// "type" discriminates the variant:
//
//   - "text": Text populated.
//   - "image_url": ImageURL populated; its Url is a "data:<mime>;base64,<body>"
//     URI (we only emit base64 today, not remote URLs).
type apiContentPart struct {
	Type     string           `json:"type"`
	Text     string           `json:"text,omitempty"`
	ImageURL *apiContentImage `json:"image_url,omitempty"`
}

type apiContentImage struct {
	URL string `json:"url"`
}

// stringOrNil returns nil when s is empty so that
// apiMessage.Content (`json:"content,omitempty"`) actually omits the
// field rather than serializing an empty `"content":""`. Bare empty
// strings as interface values are non-nil and would not be elided by
// omitempty — pre-v0.3.0 the field was typed `string` and got elided
// implicitly. Keep that wire shape stable.
func stringOrNil(s string) any {
	if s == "" {
		return nil
	}
	return s
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
		// Strict opts the tool into grammar-constrained sampling. Lives
		// inside the `function` object on the OpenAI wire (peer of
		// name/description/parameters). omitempty so non-strict tools
		// don't emit `"strict":false`.
		Strict bool `json:"strict,omitempty"`
	} `json:"function"`
}

func buildRequestBody(req llm.Request) (io.Reader, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
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
		at.Function.Strict = t.Strict
		body.Tools = append(body.Tools, at)
	}

	tc, err := toAPIToolChoice(req.ToolChoice)
	if err != nil {
		return nil, err
	}
	body.ToolChoice = tc

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
//
// ImageBlock is allowed only on user-role messages — rejecting it on
// assistant / tool roles is friendlier than silently emitting an empty
// content array on the wire.
func convertOutgoingMessage(m llm.Message) ([]apiMessage, error) {
	// VideoBlock is not supported by the OpenAI Chat Completions API.
	// Reject early on every role so callers see the unsupported-feature
	// error immediately instead of having their video silently dropped
	// by the text-only fast path or buried in a downstream API error.
	for _, b := range m.Content {
		if _, ok := b.(llm.VideoBlock); ok {
			return nil, fmt.Errorf("openai: VideoBlock is not supported; the OpenAI Chat Completions API has no native video input. Extract frames client-side and submit as ImageBlocks")
		}
	}
	if m.Role != llm.RoleUser {
		for _, b := range m.Content {
			if _, ok := b.(llm.ImageBlock); ok {
				return nil, fmt.Errorf("openai: ImageBlock is only valid on user-role messages (got role %q)", m.Role)
			}
		}
	}
	switch m.Role {
	case llm.RoleUser:
		// Text-only fast path emits a plain string content for maximum
		// compatibility with hosts that don't yet accept the array form.
		// As soon as any ImageBlock is present, switch to the array form.
		hasImage := false
		for _, b := range m.Content {
			if _, ok := b.(llm.ImageBlock); ok {
				hasImage = true
				break
			}
		}
		if !hasImage {
			var sb strings.Builder
			for _, b := range m.Content {
				if tb, ok := b.(llm.TextBlock); ok {
					if sb.Len() > 0 {
						sb.WriteString("\n")
					}
					sb.WriteString(tb.Text)
				}
			}
			return []apiMessage{{Role: "user", Content: stringOrNil(sb.String())}}, nil
		}

		// Multimodal: preserve block order in the wire array.
		var parts []apiContentPart
		for _, b := range m.Content {
			switch v := b.(type) {
			case llm.TextBlock:
				parts = append(parts, apiContentPart{Type: "text", Text: v.Text})
			case llm.ImageBlock:
				if err := v.Validate(); err != nil {
					return nil, fmt.Errorf("openai: %w", err)
				}
				parts = append(parts, apiContentPart{
					Type: "image_url",
					ImageURL: &apiContentImage{
						URL: "data:" + v.MimeType + ";base64," + v.Data,
					},
				})
			default:
				// Ignore unsupported block types on user messages (e.g.
				// stray ThinkingBlock from a copy-paste). Same conservative
				// behavior as the text-only fast path.
			}
		}
		return []apiMessage{{Role: "user", Content: parts}}, nil

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
		return []apiMessage{{Role: "assistant", Content: stringOrNil(sb.String()), ToolCalls: calls}}, nil

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
				Content:    stringOrNil(tr.Content),
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
