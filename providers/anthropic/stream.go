package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/internal/sse"
)

// errIterationStopped is a sentinel used to break out of the SSE callback
// when the user-supplied yield func returns false. We swallow it before
// surfacing errors to the iterator.
var errIterationStopped = errors.New("iteration stopped")

// streamDecoder owns the per-stream state needed to translate Anthropic SSE
// frames into llm.StreamEvent values. One decoder per HTTP response.
type streamDecoder struct {
	// inputTokens captured at message_start so we can compute totals on
	// message_delta where Anthropic ships final usage.
	inputTokens      int
	cacheReadTokens  int
	cacheWriteTokens int
	model            string
	stopReason       llm.StopReason

	// activeBlockKinds maps content_block index -> kind, so deltas can be
	// dispatched to the right delta event type and stop frames close the
	// right block.
	activeBlockKinds map[int]string

	// argsBuf collects partial_json deltas until content_block_stop so the
	// final EventToolCallEnd carries assembled JSON. Also reused to buffer
	// signature_delta payloads for thinking blocks (kinds never collide on
	// a single index).
	argsBuf map[int][]byte

	// cachedOutputTokens captures the final output_tokens count from
	// message_delta; emitted alongside usage in EventMessageEnd.
	cachedOutputTokens int
}

func newStreamDecoder() *streamDecoder {
	return &streamDecoder{
		activeBlockKinds: map[int]string{},
		argsBuf:          map[int][]byte{},
	}
}

// decode reads SSE frames from r and yields llm.StreamEvent values. Returns
// errIterationStopped if the yield function aborts the loop, any read or
// parse error otherwise, or nil on clean end of stream.
func (d *streamDecoder) decode(r io.Reader, yield func(llm.StreamEvent, error) bool) error {
	stopped := false
	err := sse.Read(r, 0, func(f sse.Frame) error {
		if stopped {
			return errIterationStopped
		}
		// Dispatch based on the SSE event name. Anthropic always sets event:
		// and data: so we don't need to inspect the data shape until inside
		// the handler.
		switch f.Event {
		case "ping":
			return nil
		case "message_start":
			return d.handleMessageStart(f.Data, yield, &stopped)
		case "content_block_start":
			return d.handleContentBlockStart(f.Data, yield, &stopped)
		case "content_block_delta":
			return d.handleContentBlockDelta(f.Data, yield, &stopped)
		case "content_block_stop":
			return d.handleContentBlockStop(f.Data, yield, &stopped)
		case "message_delta":
			return d.handleMessageDelta(f.Data)
		case "message_stop":
			return d.handleMessageStop(yield, &stopped)
		case "error":
			return d.handleError(f.Data)
		default:
			// Unknown event types — Anthropic reserves the right to add new
			// ones. Ignore quietly rather than fail.
			return nil
		}
	})
	if errors.Is(err, errIterationStopped) {
		return err
	}
	return err
}

func (d *streamDecoder) handleMessageStart(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		Type    string `json:"type"`
		Message struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			} `json:"usage"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("message_start: %w", err)
	}
	d.model = ev.Message.Model
	d.inputTokens = ev.Message.Usage.InputTokens
	d.cacheWriteTokens = ev.Message.Usage.CacheCreationInputTokens
	d.cacheReadTokens = ev.Message.Usage.CacheReadInputTokens
	if !yield(llm.EventMessageStart{Model: d.model}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleContentBlockStart(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		Index        int             `json:"index"`
		ContentBlock json.RawMessage `json:"content_block"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("content_block_start: %w", err)
	}
	var kindOnly struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(ev.ContentBlock, &kindOnly); err != nil {
		return fmt.Errorf("content_block_start kind: %w", err)
	}
	d.activeBlockKinds[ev.Index] = kindOnly.Type
	switch kindOnly.Type {
	case "text":
		if !yield(llm.EventTextStart{BlockIndex: ev.Index}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "thinking", "redacted_thinking":
		if !yield(llm.EventThinkingStart{BlockIndex: ev.Index}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "tool_use":
		var blk struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		}
		_ = json.Unmarshal(ev.ContentBlock, &blk)
		d.argsBuf[ev.Index] = nil
		if !yield(llm.EventToolCallStart{BlockIndex: ev.Index, ID: blk.ID, Name: blk.Name}, nil) {
			*stopped = true
			return errIterationStopped
		}
	default:
		// Unknown block type — skip; closing event will also be a no-op.
	}
	return nil
}

func (d *streamDecoder) handleContentBlockDelta(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		Index int             `json:"index"`
		Delta json.RawMessage `json:"delta"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("content_block_delta: %w", err)
	}
	var deltaKind struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(ev.Delta, &deltaKind); err != nil {
		return fmt.Errorf("content_block_delta kind: %w", err)
	}
	switch deltaKind.Type {
	case "text_delta":
		var d2 struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(ev.Delta, &d2); err != nil {
			return fmt.Errorf("text_delta: %w", err)
		}
		if !yield(llm.EventTextDelta{BlockIndex: ev.Index, Delta: d2.Text}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "thinking_delta":
		var d2 struct {
			Thinking string `json:"thinking"`
		}
		if err := json.Unmarshal(ev.Delta, &d2); err != nil {
			return fmt.Errorf("thinking_delta: %w", err)
		}
		if !yield(llm.EventThinkingDelta{BlockIndex: ev.Index, Delta: d2.Thinking}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "signature_delta":
		// Signature deltas arrive shortly before content_block_stop on a
		// thinking block. We accumulate into the activeBlockKinds buffer via
		// a side map: simpler to just stash in argsBuf keyed by index since
		// thinking and tool_use don't co-occupy the same index.
		var d2 struct {
			Signature string `json:"signature"`
		}
		if err := json.Unmarshal(ev.Delta, &d2); err != nil {
			return fmt.Errorf("signature_delta: %w", err)
		}
		d.argsBuf[ev.Index] = append(d.argsBuf[ev.Index], []byte(d2.Signature)...)
	case "input_json_delta":
		var d2 struct {
			PartialJSON string `json:"partial_json"`
		}
		if err := json.Unmarshal(ev.Delta, &d2); err != nil {
			return fmt.Errorf("input_json_delta: %w", err)
		}
		d.argsBuf[ev.Index] = append(d.argsBuf[ev.Index], []byte(d2.PartialJSON)...)
		if !yield(llm.EventToolCallDelta{BlockIndex: ev.Index, Delta: d2.PartialJSON}, nil) {
			*stopped = true
			return errIterationStopped
		}
	}
	return nil
}

func (d *streamDecoder) handleContentBlockStop(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		Index int `json:"index"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("content_block_stop: %w", err)
	}
	kind := d.activeBlockKinds[ev.Index]
	delete(d.activeBlockKinds, ev.Index)
	switch kind {
	case "text":
		if !yield(llm.EventTextEnd{BlockIndex: ev.Index}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "thinking", "redacted_thinking":
		sig := string(d.argsBuf[ev.Index])
		delete(d.argsBuf, ev.Index)
		if !yield(llm.EventThinkingEnd{BlockIndex: ev.Index, Signature: sig}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "tool_use":
		args := d.argsBuf[ev.Index]
		delete(d.argsBuf, ev.Index)
		// If the model emitted no input deltas at all (empty input), default
		// to an empty JSON object so downstream consumers can always
		// json.Unmarshal without special-casing.
		if len(args) == 0 {
			args = []byte("{}")
		}
		if !yield(llm.EventToolCallEnd{BlockIndex: ev.Index, Arguments: args}, nil) {
			*stopped = true
			return errIterationStopped
		}
	}
	return nil
}

func (d *streamDecoder) handleMessageDelta(data string) error {
	var ev struct {
		Delta struct {
			StopReason string `json:"stop_reason"`
		} `json:"delta"`
		Usage struct {
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("message_delta: %w", err)
	}
	if ev.Delta.StopReason != "" {
		d.stopReason = stopReasonFromAPI(ev.Delta.StopReason)
	}
	// message_delta also carries the final usage tally.
	if ev.Usage.OutputTokens != 0 {
		// We finalize and emit usage in handleMessageStop; stash for now.
		d.cachedOutputTokens = ev.Usage.OutputTokens
	}
	return nil
}

func (d *streamDecoder) handleMessageStop(yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	usage := llm.Usage{
		InputTokens:      d.inputTokens,
		OutputTokens:     d.cachedOutputTokens,
		CacheReadTokens:  d.cacheReadTokens,
		CacheWriteTokens: d.cacheWriteTokens,
	}
	usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	if !yield(llm.EventMessageEnd{StopReason: d.stopReason, Usage: usage}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleError(data string) error {
	var ev struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal([]byte(data), &ev)
	return &llm.APIError{
		Provider: "anthropic",
		Status:   500, // mid-stream errors don't carry an HTTP status
		Body:     []byte(data),
		Inner:    llm.ErrProvider,
	}
}
