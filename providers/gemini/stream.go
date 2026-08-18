package gemini

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/internal/sse"
)

// errIterationStopped sentinel: yield returned false; bail without
// surfacing as an API error.
var errIterationStopped = errors.New("iteration stopped")

// streamResponse is the JSON shape inside each `data:` SSE event.
// Only the fields pi-llm-go cares about are decoded; the rest are
// silently ignored.
type streamResponse struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata *usageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion  string         `json:"modelVersion,omitempty"`
	ResponseID    string         `json:"responseId,omitempty"`
}

type candidate struct {
	Content      candidateContent `json:"content"`
	FinishReason string           `json:"finishReason,omitempty"`
	Index        int              `json:"index"`
}

type candidateContent struct {
	Role  string    `json:"role"`
	Parts []apiPart `json:"parts,omitempty"`
}

type usageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount,omitempty"`
}

// decodeStream parses Gemini's SSE stream and translates each event
// into one or more llm.StreamEvent values. Honors yield's bool return
// for early termination.
//
// Wire model (probed 2026-05-12):
//   - One `data: {...}` frame per chunk.
//   - candidates[0].content.parts[] is the DELTA, not a snapshot. Text
//     parts append to a running buffer; functionCall parts arrive in
//     one frame (Gemini doesn't split tool args across frames).
//   - The final frame has finishReason set and (usually) empty parts.
//   - usageMetadata is CUMULATIVE on every frame; we capture the last
//     non-zero one and emit at MessageEnd.
//   - thoughtsTokenCount is the reasoning-token count. The thought text
//     itself only appears when generationConfig.thinkingConfig.includeThoughts
//     is true.
//   - thoughtSignature is opaque part metadata. It can accompany text,
//     thought, or functionCall parts and may arrive on a final empty text
//     chunk, so capture does not depend on a non-empty text delta.
func decodeStream(r io.Reader, modelHint string, yield func(llm.StreamEvent, error) bool) {
	acc := newStreamAccumulator(modelHint)

	if !yield(llm.EventMessageStart{Model: modelHint}, nil) {
		return
	}

	err := sse.Read(r, func(f sse.Frame) error {
		// Gemini does not emit named events; the data line is JSON.
		if f.Data == "" {
			return nil
		}
		var resp streamResponse
		if err := json.Unmarshal([]byte(f.Data), &resp); err != nil {
			return fmt.Errorf("decode sse frame: %w (frame=%q)", err, f.Data)
		}
		for _, ev := range acc.consume(resp) {
			if !yield(ev, nil) {
				return errIterationStopped
			}
		}
		return nil
	})
	if err != nil && !errors.Is(err, errIterationStopped) {
		yield(nil, fmt.Errorf("gemini: stream: %w", err))
		return
	}

	for _, ev := range acc.finalize() {
		if !yield(ev, nil) {
			return
		}
	}
}

// streamAccumulator turns Gemini's delta-on-every-frame stream into
// the block-oriented event sequence pi-llm-go's StreamEvent demands.
// Tracks per-block-index state so concurrent multi-part responses
// (text + function-call interleaved) work even when Gemini reorders.
type streamAccumulator struct {
	modelHint string

	// Block bookkeeping: text/thinking blocks accumulate over deltas;
	// tool-call blocks arrive whole in one frame.
	nextBlockIndex int
	// textBlocks / thinkingBlocks map "currently-open" stream to its
	// emitted block index. Gemini sometimes interleaves a text chunk
	// and a thought chunk; we keep them separate.
	openTextIdx       int
	openThinkingIdx   int
	textOpen          bool
	thinkingOpen      bool
	textSignature     string
	thinkingSignature string

	lastUsage   *usageMetadata
	stopReason  llm.StopReason
	gotFinish   bool
	gotToolCall bool // any functionCall part seen this turn?
}

func newStreamAccumulator(modelHint string) *streamAccumulator {
	return &streamAccumulator{modelHint: modelHint}
}

func (a *streamAccumulator) consume(resp streamResponse) []llm.StreamEvent {
	var events []llm.StreamEvent
	if resp.UsageMetadata != nil {
		// Cumulative — keep overwriting.
		a.lastUsage = resp.UsageMetadata
	}
	if len(resp.Candidates) == 0 {
		return events
	}
	cand := resp.Candidates[0]

	for _, part := range cand.Content.Parts {
		events = append(events, a.consumePart(part)...)
	}

	if cand.FinishReason != "" {
		a.gotFinish = true
		a.stopReason = mapFinishReason(cand.FinishReason)
		// Close any open text / thinking blocks before MessageEnd.
		events = append(events, a.closeOpen()...)
	}
	return events
}

// consumePart turns one apiPart from the wire into zero or more
// StreamEvents.
func (a *streamAccumulator) consumePart(p apiPart) []llm.StreamEvent {
	switch {
	case p.FunctionCall != nil:
		// Close any open prose block first.
		events := a.closeOpen()
		a.gotToolCall = true
		// Prefer Gemini 3+'s wire-level id; fall back to function name
		// on Gemini 2.x where id is empty. Name-as-id collapses if the
		// model issues two calls to the same tool in one turn — that's
		// a Gemini 2.x limitation we can't fix client-side, but we
		// stably round-trip whatever the server emitted.
		id := p.FunctionCall.Id
		if id == "" {
			id = p.FunctionCall.Name
		}
		idx := a.nextBlockIndex
		a.nextBlockIndex++
		events = append(events,
			llm.EventToolCallStart{
				BlockIndex: idx,
				ID:         id,
				Name:       p.FunctionCall.Name,
			},
			llm.EventToolCallDelta{
				BlockIndex: idx,
				Delta:      string(p.FunctionCall.Args),
			},
			llm.EventToolCallEnd{
				BlockIndex: idx,
				Arguments:  p.FunctionCall.Args,
				Signature:  p.ThoughtSignature,
			},
		)
		return events

	case p.Thought:
		// Thinking-mode chunk. Open a thinking block on first thought.
		events := []llm.StreamEvent{}
		if p.ThoughtSignature != "" && a.thinkingOpen {
			// A signature belongs to this exact Part. Never merge the signed
			// chunk into an already-open unsigned thought block.
			events = append(events, a.closeThinking()...)
		}
		if !a.thinkingOpen {
			// If a text block is open it stays open; thoughts and text
			// can interleave on the wire — though Gemini typically
			// emits all thoughts first.
			a.openThinkingIdx = a.nextBlockIndex
			a.nextBlockIndex++
			a.thinkingOpen = true
			a.thinkingSignature = ""
			events = append(events, llm.EventThinkingStart{BlockIndex: a.openThinkingIdx})
		}
		if p.ThoughtSignature != "" {
			a.thinkingSignature = p.ThoughtSignature
		}
		if p.Text != nil && *p.Text != "" {
			events = append(events, llm.EventThinkingDelta{
				BlockIndex: a.openThinkingIdx,
				Delta:      *p.Text,
			})
		}
		if p.ThoughtSignature != "" {
			// Close signed parts immediately so a later unsigned stream chunk
			// cannot be merged into the signature's positional context.
			events = append(events, a.closeThinking()...)
		}
		return events

	case p.Text != nil:
		if *p.Text == "" && p.ThoughtSignature == "" {
			// Nothing to record. Opening a block would leave an empty
			// TextBlock, which other providers reject when the message is
			// replayed (Anthropic 400s on a content-less text block).
			return nil
		}
		events := []llm.StreamEvent{}
		// Close thinking if it was open and we're now in text territory.
		if a.thinkingOpen {
			events = append(events, a.closeThinking()...)
		}
		if p.ThoughtSignature != "" && a.textOpen {
			// The API forbids merging a signed Part with an unsigned Part.
			// A final empty-text signature chunk therefore becomes its own
			// TextBlock instead of metadata on the preceding prose block.
			events = append(events, a.closeText()...)
		}
		if !a.textOpen {
			a.openTextIdx = a.nextBlockIndex
			a.nextBlockIndex++
			a.textOpen = true
			a.textSignature = ""
			events = append(events, llm.EventTextStart{BlockIndex: a.openTextIdx})
		}
		if p.ThoughtSignature != "" {
			a.textSignature = p.ThoughtSignature
		}
		if *p.Text != "" {
			events = append(events, llm.EventTextDelta{
				BlockIndex: a.openTextIdx,
				Delta:      *p.Text,
			})
		}
		if p.ThoughtSignature != "" {
			events = append(events, a.closeText()...)
		}
		return events

	case p.ThoughtSignature != "":
		// Be liberal with a metadata-only part even though Gemini normally
		// serializes the empty text field explicitly. Preserve the signature
		// as an empty TextBlock rather than silently discarding it.
		events := a.closeOpen()
		if !a.textOpen {
			a.openTextIdx = a.nextBlockIndex
			a.nextBlockIndex++
			a.textOpen = true
			events = append(events, llm.EventTextStart{BlockIndex: a.openTextIdx})
		}
		a.textSignature = p.ThoughtSignature
		events = append(events, a.closeText()...)
		return events
	}
	return nil
}

func (a *streamAccumulator) closeOpen() []llm.StreamEvent {
	var events []llm.StreamEvent
	events = append(events, a.closeThinking()...)
	events = append(events, a.closeText()...)
	return events
}

func (a *streamAccumulator) closeThinking() []llm.StreamEvent {
	if !a.thinkingOpen {
		return nil
	}
	event := llm.EventThinkingEnd{
		BlockIndex: a.openThinkingIdx,
		Signature:  a.thinkingSignature,
	}
	a.thinkingOpen = false
	a.thinkingSignature = ""
	return []llm.StreamEvent{event}
}

func (a *streamAccumulator) closeText() []llm.StreamEvent {
	if !a.textOpen {
		return nil
	}
	event := llm.EventTextEnd{
		BlockIndex: a.openTextIdx,
		Signature:  a.textSignature,
	}
	a.textOpen = false
	a.textSignature = ""
	return []llm.StreamEvent{event}
}

func (a *streamAccumulator) finalize() []llm.StreamEvent {
	events := a.closeOpen()
	usage := llm.Usage{}
	if a.lastUsage != nil {
		usage.InputTokens = a.lastUsage.PromptTokenCount
		usage.OutputTokens = a.lastUsage.CandidatesTokenCount + a.lastUsage.ThoughtsTokenCount
		usage.ReasoningTokens = a.lastUsage.ThoughtsTokenCount
		usage.TotalTokens = a.lastUsage.TotalTokenCount
		// Gemini exposes prompt-cache hits via cachedContentTokenCount
		// on the wire; not surfaced at v0.4.0 — Gemini's caching is
		// opt-in via the CachedContent API rather than automatic.
		usage.CacheReadTokens = 0
		usage.CacheWriteTokens = 0
	}
	stop := a.stopReason
	if !a.gotFinish {
		stop = llm.StopReasonEnd
	}
	// Gemini's finishReason vocabulary has no tool-use terminator
	// (unlike Anthropic's "tool_use" or OpenAI's "tool_calls"). The
	// response carries functionCall parts alongside finishReason=STOP.
	// Synthesize StopReasonToolUse so downstream consumers branch on
	// it uniformly across providers.
	if a.gotToolCall {
		stop = llm.StopReasonToolUse
	}
	events = append(events, llm.EventMessageEnd{
		StopReason: stop,
		Usage:      usage,
	})
	return events
}

// mapFinishReason translates Gemini's stop reason strings onto
// pi-llm-go's normalized StopReason.
//
// Gemini's actual enum (verified against ai.google.dev/api/generate-content):
// STOP, MAX_TOKENS, SAFETY, RECITATION, LANGUAGE, OTHER, BLOCKLIST,
// PROHIBITED_CONTENT, SPII, MALFORMED_FUNCTION_CALL, IMAGE_SAFETY,
// UNEXPECTED_TOOL_CALL. There is NO tool-use terminator — callers
// detect tool calls from message content; finalize() synthesizes
// StopReasonToolUse from the gotToolCall flag.
//
// All non-STOP, non-MAX_TOKENS values currently collapse to
// StopReasonEnd at v0.4.0 — the assistant content carries the
// explanation. A future PR may add finer-grained values for the
// content-filter cases (SAFETY / RECITATION / BLOCKLIST / etc.) since
// those are actionable signal for the caller.
func mapFinishReason(r string) llm.StopReason {
	switch r {
	case "STOP":
		return llm.StopReasonEnd
	case "MAX_TOKENS":
		return llm.StopReasonMaxTokens
	default:
		return llm.StopReasonEnd
	}
}
