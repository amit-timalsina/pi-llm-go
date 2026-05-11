package llm

import (
	"fmt"
	"iter"
	"strings"
)

// Accumulate folds a stream of events into a stream of progressively-built
// messages. Each yield emits a snapshot of the assistant message as it
// stands after the most recent event. The final yielded value (when err
// is nil) is the fully-assembled assistant message.
//
// Callers that only want the final message should prefer Complete. Use
// Accumulate when intermediate snapshots are useful (e.g. driving a UI that
// renders each delta against a complete message tree rather than against
// individual deltas).
//
// Each snapshot is an independent value — internal slices and strings are
// copied on emit so callers can retain previous snapshots without aliasing.
func Accumulate(events iter.Seq2[StreamEvent, error]) iter.Seq2[*Message, error] {
	return func(yield func(*Message, error) bool) {
		msg := &Message{Role: RoleAssistant}
		// builders tracks per-block state across deltas. Each entry maps a
		// block index to the in-progress text/JSON content for that block.
		textBuilders := map[int]*strings.Builder{}
		thinkBuilders := map[int]*strings.Builder{}
		toolArgBuilders := map[int]*strings.Builder{}

		emit := func() bool {
			return yield(snapshot(msg, textBuilders, thinkBuilders, toolArgBuilders), nil)
		}

		for event, err := range events {
			if err != nil {
				yield(snapshot(msg, textBuilders, thinkBuilders, toolArgBuilders), err)
				return
			}

			switch e := event.(type) {
			case EventMessageStart:
				msg.Model = e.Model
			case EventTextStart:
				msg.Content = appendBlockAt(msg.Content, e.BlockIndex, TextBlock{})
				textBuilders[e.BlockIndex] = &strings.Builder{}
			case EventTextDelta:
				if b, ok := textBuilders[e.BlockIndex]; ok {
					b.WriteString(e.Delta)
				}
			case EventTextEnd:
				if b, ok := textBuilders[e.BlockIndex]; ok {
					msg.Content[e.BlockIndex] = TextBlock{Text: b.String()}
				}
			case EventThinkingStart:
				msg.Content = appendBlockAt(msg.Content, e.BlockIndex, ThinkingBlock{})
				thinkBuilders[e.BlockIndex] = &strings.Builder{}
			case EventThinkingDelta:
				if b, ok := thinkBuilders[e.BlockIndex]; ok {
					b.WriteString(e.Delta)
				}
			case EventThinkingEnd:
				if b, ok := thinkBuilders[e.BlockIndex]; ok {
					msg.Content[e.BlockIndex] = ThinkingBlock{Thinking: b.String(), Signature: e.Signature}
				}
			case EventToolCallStart:
				msg.Content = appendBlockAt(msg.Content, e.BlockIndex, ToolCallBlock{ID: e.ID, Name: e.Name})
				toolArgBuilders[e.BlockIndex] = &strings.Builder{}
			case EventToolCallDelta:
				if b, ok := toolArgBuilders[e.BlockIndex]; ok {
					b.WriteString(e.Delta)
				}
			case EventToolCallEnd:
				existing, _ := msg.Content[e.BlockIndex].(ToolCallBlock)
				existing.Arguments = e.Arguments
				msg.Content[e.BlockIndex] = existing
			case EventMessageEnd:
				msg.StopReason = e.StopReason
				msg.Usage = e.Usage
			default:
				// Unknown event type — surface as error so users find out fast
				// rather than silently dropping data.
				yield(snapshot(msg, textBuilders, thinkBuilders, toolArgBuilders),
					fmt.Errorf("llm: unknown event type %T", event))
				return
			}

			if !emit() {
				return
			}
		}
	}
}

// snapshot returns a defensive copy of msg with in-progress builder content
// realized into the corresponding Block values. Builders for blocks that
// have already ended (i.e. their final value is set on msg.Content) are
// skipped.
func snapshot(
	msg *Message,
	textBuilders, thinkBuilders map[int]*strings.Builder,
	toolArgBuilders map[int]*strings.Builder,
) *Message {
	out := &Message{
		Role:       msg.Role,
		Model:      msg.Model,
		StopReason: msg.StopReason,
		Usage:      msg.Usage,
	}
	if len(msg.Content) > 0 {
		out.Content = make([]Block, len(msg.Content))
		copy(out.Content, msg.Content)
	}
	// Overlay in-progress builders so partial snapshots include current text.
	for idx, b := range textBuilders {
		if idx >= len(out.Content) {
			continue
		}
		if _, ok := out.Content[idx].(TextBlock); ok {
			out.Content[idx] = TextBlock{Text: b.String()}
		}
	}
	for idx, b := range thinkBuilders {
		if idx >= len(out.Content) {
			continue
		}
		if existing, ok := out.Content[idx].(ThinkingBlock); ok {
			existing.Thinking = b.String()
			out.Content[idx] = existing
		}
	}
	for idx, b := range toolArgBuilders {
		if idx >= len(out.Content) {
			continue
		}
		if existing, ok := out.Content[idx].(ToolCallBlock); ok {
			// Treat the in-progress builder as the current (possibly partial)
			// JSON; callers should not assume it parses until EventToolCallEnd.
			existing.Arguments = []byte(b.String())
			out.Content[idx] = existing
		}
	}
	return out
}

// appendBlockAt sets dst[idx] = block, growing dst with zero values as
// needed. Providers normally emit BlockIndex values in monotonic order
// (0, 1, 2, …), but the helper handles gaps so adversarial event streams
// don't panic.
func appendBlockAt(dst []Block, idx int, block Block) []Block {
	for len(dst) <= idx {
		dst = append(dst, nil)
	}
	dst[idx] = block
	return dst
}
