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

	// Strict opts this tool into grammar-constrained sampling: the
	// model's token sampler is constrained to schema-valid tokens, so
	// the emitted tool input is GUARANTEED to match InputSchema (no
	// app-side validate-and-retry round-trip).
	//
	// Per-provider behavior:
	//   - Anthropic Messages: forwarded as the top-level `strict` field
	//     on the tool definition (peer of name / description /
	//     input_schema). Compiled into a grammar; schemas are cached
	//     server-side for up to 24h. JSON Schema subset applies (see
	//     Anthropic's "JSON Schema limitations" docs).
	//   - OpenAI Chat Completions: forwarded as `function.strict` on
	//     the tool definition. Requires `additionalProperties: false`
	//     at every object level and every property in `required`.
	//   - OpenAI Responses API: same as Chat Completions.
	//   - Gemini: ignored (Gemini has no per-tool strict mode; structured
	//     output uses a different surface, response_schema).
	//
	// Caveat: strict mode adds first-call latency for grammar compilation
	// (cached thereafter). For tools called once across many runs the
	// non-strict + caller-side-validate path may be cheaper.
	Strict bool
}

// ToolChoice controls whether and which tool the model must call on a
// turn. nil on Request.ToolChoice preserves provider defaults:
// `auto` when Tools is non-empty, `none` otherwise.
//
// Per-provider behavior:
//
//   - Anthropic: forwarded as `tool_choice` object on the wire. EXTENDED
//     THINKING INCOMPATIBILITY — `Any` and `Tool` are rejected by
//     Anthropic when `ThinkingConfig` is also set (only `Auto` / `None`
//     are compatible with thinking). The provider does not validate
//     this combination client-side; a 400 surfaces at request time.
//   - OpenAI Chat Completions: forwarded as `tool_choice`. The keyword
//     mapping differs from Anthropic — `Any` becomes OpenAI's
//     `"required"`. `Tool` becomes the
//     `{"type":"function","function":{"name":"..."}}` object shape.
//   - OpenAI Responses API: forwarded the same as Chat Completions
//     (same wire shape).
//   - Gemini: forwarded as `tool_config.function_calling_config` with
//     `mode` mapping: `Auto`→AUTO, `Any`→ANY, `None`→NONE; `Tool`
//     becomes ANY with `allowed_function_names: [Name]`. The allowlist
//     constrains the model exclusively — even if Tools declares N
//     other functions, the model can only emit a call to `Name`.
//     Semantically stronger than Anthropic's `tool` mode (the model
//     may also choose not to call any tool there).
type ToolChoice struct {
	// Type controls the choice mode. Must be one of the ToolChoice*
	// constants; the provider rejects unknown values at request build
	// time.
	Type ToolChoiceType

	// Name is the tool name the model is forced to call. REQUIRED when
	// Type == ToolChoiceTool; ignored otherwise.
	Name string
}

// ToolChoiceType enumerates the four tool-choice modes pi-llm-go
// abstracts over. Provider converters map these onto the provider's
// concrete wire shape (Anthropic / OpenAI / Gemini all differ).
type ToolChoiceType string

const (
	// ToolChoiceAuto lets the model decide whether to call a tool.
	// Provider default when Tools is non-empty.
	ToolChoiceAuto ToolChoiceType = "auto"

	// ToolChoiceAny forces the model to call SOME tool (which one is
	// the model's choice). Maps to Anthropic's `any`, OpenAI's
	// `required`, Gemini's `ANY`.
	ToolChoiceAny ToolChoiceType = "any"

	// ToolChoiceTool forces the model to call a specific tool by name.
	// ToolChoice.Name is required; an empty Name is a build-time error.
	ToolChoiceTool ToolChoiceType = "tool"

	// ToolChoiceNone disables tool use even when Tools is non-empty.
	// Useful for a final-answer-only turn after a tool round-trip.
	ToolChoiceNone ToolChoiceType = "none"
)
