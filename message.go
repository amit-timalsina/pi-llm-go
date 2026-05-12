package llm

import (
	"encoding/json"
	"errors"
	"strings"
	"time"
)

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

// ImageBlock holds image data for multimodal input. Data is the raw
// base64-encoded image bytes (no "data:" URI prefix); MimeType is the
// standard MIME identifier (e.g. "image/png"). Providers convert to
// their on-wire format at the boundary.
//
// pi-llm-go does NOT fetch image URLs. Callers that want to attach a
// remote image must download it themselves first, then construct an
// ImageBlock with the resulting bytes encoded. This keeps the library
// network-free except for the LLM provider call itself — no surprise
// timeouts, no surprise 404s mid-stream.
//
// Portable MIME types accepted by every built-in provider:
//
//   - "image/jpeg"
//   - "image/png"
//   - "image/gif"
//   - "image/webp"
//
// Other types may work on specific providers (e.g. OpenAI accepts more)
// but pi-llm-go does not pre-validate — the provider returns an
// ErrInvalidRequest if the type is unsupported.
//
// v0.3.0 supports ImageBlock as USER-message input only. Assistant
// image output is provider-specific and a separate, future feature.
type ImageBlock struct {
	// Data is the raw base64-encoded image bytes. Do NOT include the
	// "data:<mime>;base64," prefix — providers add it where required.
	Data string

	// MimeType is the image's MIME type (e.g. "image/png"). Required.
	MimeType string
}

// VideoBlock holds video data for multimodal input. Today only the
// built-in Gemini provider accepts video natively; Anthropic and OpenAI
// providers reject VideoBlock at the wire boundary (callers wanting
// video understanding on those providers must extract frames client-side
// and submit them as ImageBlocks).
//
// VideoBlock has three mutually-exclusive emission shapes:
//
//   - **Data + MimeType set**: inline base64. Total request body must
//     stay under the provider's inline cap (Gemini: ~20 MB).
//   - **URI set**: a pre-uploaded reference. For Gemini, this is either
//     an `https://generativelanguage.googleapis.com/v1beta/files/...`
//     handle from the Files API (see providers/gemini/files) or a
//     YouTube URL (public videos only; free-tier 8h/day cap).
//   - Exactly one of (Data, URI) must be non-empty; both empty or both
//     set is a contract violation rejected by Validate().
//
// Optional StartOffset, EndOffset, and FPS let callers clip the
// segment and override Gemini's default 1 FPS sampling. nil = use the
// provider's default. FPS is float for fractional rates (e.g. 0.5).
//
// MimeType uses standard video MIME identifiers: video/mp4,
// video/quicktime, video/webm, video/mpeg, etc. Required when Data is
// set; ignored when only URI is set (server infers from the file).
type VideoBlock struct {
	// Data is the raw base64-encoded video bytes for inline emission.
	// Do NOT include the "data:" URI prefix. Mutually exclusive with URI.
	Data string

	// URI is a pre-uploaded reference (Files API handle, YouTube URL,
	// or provider-specific URI). Mutually exclusive with Data.
	URI string

	// MimeType is the video's MIME type (e.g. "video/mp4"). Required
	// when Data is set; optional when only URI is set.
	MimeType string

	// StartOffset, when non-nil, clips the start of the segment. The
	// provider must support clipping; Gemini does via videoMetadata.
	StartOffset *time.Duration

	// EndOffset, when non-nil, clips the end of the segment.
	EndOffset *time.Duration

	// FPS, when non-nil, overrides the provider's default sampling rate.
	// Gemini defaults to 1 FPS (1 video frame per second analyzed);
	// pass a higher value for action-dense content or lower for long
	// static footage. Float for fractional rates (0.5 = 1 frame per 2s).
	FPS *float64
}

func (TextBlock) isBlock()       {}
func (ThinkingBlock) isBlock()   {}
func (ToolCallBlock) isBlock()   {}
func (ToolResultBlock) isBlock() {}
func (ImageBlock) isBlock()      {}
func (VideoBlock) isBlock()      {}

// Validate enforces the VideoBlock contract:
//   - exactly one of (Data, URI) must be non-empty
//   - Data must not carry a "data:" URI prefix (raw base64 only)
//   - MimeType is required when Data is set
func (v VideoBlock) Validate() error {
	hasData := v.Data != ""
	hasURI := v.URI != ""
	if !hasData && !hasURI {
		return errors.New("VideoBlock: exactly one of Data or URI must be set")
	}
	if hasData && hasURI {
		return errors.New("VideoBlock: Data and URI are mutually exclusive")
	}
	if hasData {
		if strings.HasPrefix(v.Data, "data:") {
			return errors.New("VideoBlock: Data must be raw base64; remove the leading \"data:\" URI prefix")
		}
		if v.MimeType == "" {
			return errors.New("VideoBlock: MimeType is required when Data is set")
		}
	}
	return nil
}

// Validate enforces the ImageBlock contract: Data must be raw
// base64-encoded bytes (without the "data:<mime>;base64," URI prefix)
// and MimeType must be set. Providers call this at the wire boundary
// and surface a wrapped error if the contract is violated.
func (i ImageBlock) Validate() error {
	if i.Data == "" {
		return errors.New("ImageBlock: Data is empty")
	}
	if i.MimeType == "" {
		return errors.New("ImageBlock: MimeType is empty")
	}
	// Common foot-gun: callers paste a `data:<mime>;base64,<body>` URI
	// into Data. Providers then build `data:data:<mime>;base64,<mime>...`
	// — clearly broken. Reject early.
	if strings.HasPrefix(i.Data, "data:") {
		return errors.New("ImageBlock: Data must be raw base64; remove the leading \"data:\" URI prefix")
	}
	return nil
}
