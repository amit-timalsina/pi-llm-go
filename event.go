package llm

import "encoding/json"

// StreamEvent is the sealed sum type emitted during a streaming completion.
// Errors flow through the iterator's error half rather than as event values,
// so consumers do not need an EventError variant.
//
// Event order for one assistant turn:
//
//	EventMessageStart
//	  ( EventTextStart, EventTextDelta*, EventTextEnd
//	  | EventThinkingStart, EventThinkingDelta*, EventThinkingEnd
//	  | EventToolCallStart, EventToolCallDelta*, EventToolCallEnd
//	  )*
//	EventMessageEnd
//
// BlockIndex on per-block events is the position of the block inside the
// emitted message's Content slice — it lets consumers route deltas when
// rendering or reconstructing incrementally.
type StreamEvent interface {
	isEvent()
}

// EventMessageStart is emitted once at the start of an assistant turn,
// before any block events.
type EventMessageStart struct {
	Model string
}

// EventTextStart marks the beginning of a TextBlock.
type EventTextStart struct {
	BlockIndex int
}

// EventTextDelta appends Delta to the text in the block at BlockIndex.
type EventTextDelta struct {
	BlockIndex int
	Delta      string
}

// EventTextEnd marks the end of the TextBlock at BlockIndex.
type EventTextEnd struct {
	BlockIndex int
}

// EventThinkingStart marks the beginning of a ThinkingBlock.
type EventThinkingStart struct {
	BlockIndex int
}

// EventThinkingDelta appends Delta to the thinking block at BlockIndex.
type EventThinkingDelta struct {
	BlockIndex int
	Delta      string
}

// EventThinkingEnd marks the end of the ThinkingBlock. Signature is the
// opaque provider-supplied token to round-trip on follow-up messages.
type EventThinkingEnd struct {
	BlockIndex int
	Signature  string
}

// EventToolCallStart marks the beginning of a ToolCallBlock. ID and Name
// are available immediately; arguments stream as deltas and are emitted
// in fully assembled form on EventToolCallEnd.
type EventToolCallStart struct {
	BlockIndex int
	ID         string
	Name       string
}

// EventToolCallDelta delivers a fragment of the streaming JSON arguments.
// Callers that need the assembled arguments should wait for EventToolCallEnd
// rather than accumulate Delta bytes themselves (provider JSON delta framing
// is not guaranteed to be at value boundaries).
type EventToolCallDelta struct {
	BlockIndex int
	Delta      string
}

// EventToolCallEnd marks the end of a ToolCallBlock and carries the
// assembled arguments.
type EventToolCallEnd struct {
	BlockIndex int
	Arguments  json.RawMessage
}

// EventMessageEnd is the terminal event. It carries the normalized stop
// reason and the final usage tally.
type EventMessageEnd struct {
	StopReason StopReason
	Usage      Usage
}

func (EventMessageStart) isEvent()  {}
func (EventTextStart) isEvent()     {}
func (EventTextDelta) isEvent()     {}
func (EventTextEnd) isEvent()       {}
func (EventThinkingStart) isEvent() {}
func (EventThinkingDelta) isEvent() {}
func (EventThinkingEnd) isEvent()   {}
func (EventToolCallStart) isEvent() {}
func (EventToolCallDelta) isEvent() {}
func (EventToolCallEnd) isEvent()   {}
func (EventMessageEnd) isEvent()    {}
