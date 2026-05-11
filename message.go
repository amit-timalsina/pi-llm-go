package llm

import "encoding/json"

// Role enumerates message roles in a transcript. RoleTool messages carry
// tool results back to the model and may hold only ToolResultBlock content.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// StopReason is the normalized reason a model stopped generating.
// Provider-specific stop reasons map to one of these; unmappable values
// surface as errors rather than leaking provider strings.
type StopReason string

const (
	StopReasonEnd       StopReason = "end"        // natural end of turn
	StopReasonMaxTokens StopReason = "max_tokens" // hit MaxTokens cap
	StopReasonToolUse   StopReason = "tool_use"   // model requested tool calls
	StopReasonStop      StopReason = "stop"       // matched a stop sequence
)

// Message is one turn in the transcript. Content holds a sequence of blocks
// — the model emits assistant messages with mixed text / thinking / tool-call
// content; the caller sends user messages with text and tool messages with
// tool-result content.
//
// Usage, StopReason, and Model are populated on assistant messages produced
// by Complete or Accumulate; they are zero on user / tool messages and on
// messages sent into Stream.
type Message struct {
	Role    Role
	Content []Block

	Usage      Usage
	StopReason StopReason
	Model      string
}

// Block is the sealed sum type for message content. Concrete implementations
// are TextBlock, ThinkingBlock, ToolCallBlock, and ToolResultBlock — all
// defined in this package. The unexported marker method keeps the set
// closed: provider converters need exhaustive type-switches to serialize
// content correctly, so new block types must be added inside the package.
type Block interface {
	isBlock()
}

// TextBlock holds plain text content.
type TextBlock struct {
	Text string
}

// ThinkingBlock holds an extended-thinking segment emitted by reasoning
// models. Signature is an opaque provider-supplied token that must be
// preserved and replayed for multi-turn thinking continuity (Anthropic).
type ThinkingBlock struct {
	Thinking  string
	Signature string
}

// ToolCallBlock represents a tool invocation requested by the model.
// Arguments is the raw JSON object the model emitted, matching the tool's
// declared InputSchema. The agent layer validates and dispatches.
type ToolCallBlock struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResultBlock carries the result of a tool invocation back to the model.
// ToolCallID matches the ID on the originating ToolCallBlock.
type ToolResultBlock struct {
	ToolCallID string
	Content    string
	IsError    bool
}

func (TextBlock) isBlock()       {}
func (ThinkingBlock) isBlock()   {}
func (ToolCallBlock) isBlock()   {}
func (ToolResultBlock) isBlock() {}
