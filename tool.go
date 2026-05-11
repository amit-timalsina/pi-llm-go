package llm

import "encoding/json"

// Tool is the wire-level declaration of a callable function exposed to the
// model. pi-llm-go does not execute tools — it surfaces ToolCallBlocks on
// the response and accepts ToolResultBlocks in follow-up messages. Execution
// lives in pi-agent-go.
//
// InputSchema is a JSON Schema document describing the tool's expected
// input. Both Anthropic and OpenAI accept JSON Schema draft-07; the schema
// is forwarded to the provider as-is, so the caller is responsible for
// dialect choice.
type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}
