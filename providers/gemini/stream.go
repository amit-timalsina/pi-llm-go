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
//   - thoughtsTokenCount is the reasoning-token count; we surface as a
//     ThinkingBlock with no Signature (Gemini doesn't expose a thought
//     replay token at v0.4.0). The thought text itself only appears
//     when generationConfig.thinkingConfig.includeThoughts is true.
func decodeStream(r io.Reader, modelHint string, yield func(llm.StreamEvent, error) bool) {
	acc := newStreamAccumulator(modelHint)

	if !yield(llm.EventMessageStart{Model: modelHint}, nil) {
		return
	}

	err := sse.Read(r, 4*1024*1024, func(f sse.Frame) error {
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
	openTextIdx     int
	openThinkingIdx int
	textOpen        bool
	thinkingOpen    bool

	lastUsage  *usageMetadata
	stopReason llm.StopReason
	gotFinish  bool
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
		idx := a.nextBlockIndex
		a.nextBlockIndex++
		events = append(events,
			llm.EventToolCallStart{
				BlockIndex: idx,
				ID:         p.FunctionCall.Name, // Gemini doesn't issue an id; reuse name
				Name:       p.FunctionCall.Name,
			},
			llm.EventToolCallDelta{
				BlockIndex: idx,
				Delta:      string(p.FunctionCall.Args),
			},
			llm.EventToolCallEnd{
				BlockIndex: idx,
				Arguments:  p.FunctionCall.Args,
			},
		)
		return events

	case p.Thought:
		// Thinking-mode chunk. Open a thinking block on first thought.
		events := []llm.StreamEvent{}
		if !a.thinkingOpen {
			// If a text block is open it stays open; thoughts and text
			// can interleave on the wire — though Gemini typically
			// emits all thoughts first.
			a.openThinkingIdx = a.nextBlockIndex
			a.nextBlockIndex++
			a.thinkingOpen = true
			events = append(events, llm.EventThinkingStart{BlockIndex: a.openThinkingIdx})
		}
		if p.Text != "" {
			events = append(events, llm.EventThinkingDelta{
				BlockIndex: a.openThinkingIdx,
				Delta:      p.Text,
			})
		}
		return events

	case p.Text != "":
		events := []llm.StreamEvent{}
		// Close thinking if it was open and we're now in text territory.
		if a.thinkingOpen {
			events = append(events, llm.EventThinkingEnd{
				BlockIndex: a.openThinkingIdx,
				Signature:  "", // Gemini doesn't expose a thinking signature
			})
			a.thinkingOpen = false
		}
		if !a.textOpen {
			a.openTextIdx = a.nextBlockIndex
			a.nextBlockIndex++
			a.textOpen = true
			events = append(events, llm.EventTextStart{BlockIndex: a.openTextIdx})
		}
		events = append(events, llm.EventTextDelta{
			BlockIndex: a.openTextIdx,
			Delta:      p.Text,
		})
		return events
	}
	return nil
}

func (a *streamAccumulator) closeOpen() []llm.StreamEvent {
	var events []llm.StreamEvent
	if a.thinkingOpen {
		events = append(events, llm.EventThinkingEnd{
			BlockIndex: a.openThinkingIdx,
			Signature:  "",
		})
		a.thinkingOpen = false
	}
	if a.textOpen {
		events = append(events, llm.EventTextEnd{BlockIndex: a.openTextIdx})
		a.textOpen = false
	}
	return events
}

func (a *streamAccumulator) finalize() []llm.StreamEvent {
	events := a.closeOpen()
	usage := llm.Usage{}
	if a.lastUsage != nil {
		usage.InputTokens = a.lastUsage.PromptTokenCount
		usage.OutputTokens = a.lastUsage.CandidatesTokenCount + a.lastUsage.ThoughtsTokenCount
		usage.TotalTokens = a.lastUsage.TotalTokenCount
		usage.CacheReadTokens = 0 // Gemini exposes cached prompts via cachedContentTokenCount; not surfaced at v0.4.0
		usage.CacheWriteTokens = 0
	}
	stop := a.stopReason
	if !a.gotFinish {
		stop = llm.StopReasonEnd
	}
	events = append(events, llm.EventMessageEnd{
		StopReason: stop,
		Usage:      usage,
	})
	return events
}

// mapFinishReason translates Gemini's stop reason strings onto
// pi-llm-go's normalized StopReason. SAFETY / RECITATION etc. are
// treated as End — the assistant content carries the explanation.
func mapFinishReason(r string) llm.StopReason {
	switch r {
	case "STOP":
		return llm.StopReasonEnd
	case "MAX_TOKENS":
		return llm.StopReasonMaxTokens
	case "FINISH_REASON_TOOL_CODE", "TOOL_CALL":
		// Gemini's finish reason vocabulary doesn't include a tool-use
		// terminator equivalent to Anthropic's stop_reason: "tool_use".
		// If the response contains functionCall parts, callers can
		// extract them from the final message regardless of finish
		// reason. Keep this case open for future enum widening.
		return llm.StopReasonToolUse
	default:
		return llm.StopReasonEnd
	}
}
