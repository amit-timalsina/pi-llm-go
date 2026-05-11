package openai_responses

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/internal/sse"
)

var errIterationStopped = errors.New("iteration stopped")

// streamDecoder maps Responses API SSE events to llm.StreamEvents. The
// Responses protocol surfaces 53 distinct event types; this decoder
// handles the core subset:
//
//   - Lifecycle: response.created, response.completed, response.failed,
//     error.
//   - Text output: response.output_text.delta, response.output_text.done.
//   - Function tool calls: response.function_call_arguments.delta,
//     response.function_call_arguments.done.
//   - Output item creation: response.output_item.added — used to emit our
//     EventTextStart / EventToolCallStart at the right moment.
//   - Reasoning summary: response.reasoning_summary_text.delta /.done —
//     mapped to EventThinking{Start,Delta,End} so reasoning surfaces as
//     a ThinkingBlock in the final llm.Message.
//
// Other events (MCP, built-in web/file/code, image, audio) are ignored;
// adding them is additive and non-breaking.
type streamDecoder struct {
	// outputIndexToBlock tracks which output_index maps to which Block
	// position in the llm.Message we're building. Responses API uses
	// independent indexing per output item; we synthesize ascending block
	// indexes in emission order.
	outputIndexToBlock map[int]int
	itemKinds          map[int]string // "text" | "function_call" | "reasoning"
	nextBlockIdx       int

	// pendingToolCalls keyed by output_index — name/id captured at
	// output_item.added, args streamed via function_call_arguments.delta.
	pendingToolCalls map[int]toolCallMeta

	model        string
	emittedStart bool
}

type toolCallMeta struct {
	id   string
	name string
}

func newStreamDecoder() *streamDecoder {
	return &streamDecoder{
		outputIndexToBlock: map[int]int{},
		itemKinds:          map[int]string{},
		pendingToolCalls:   map[int]toolCallMeta{},
	}
}

func (d *streamDecoder) decode(r io.Reader, yield func(llm.StreamEvent, error) bool) error {
	stopped := false
	err := sse.Read(r, 0, func(f sse.Frame) error {
		if stopped {
			return errIterationStopped
		}
		switch f.Event {
		case "response.created":
			return d.handleCreated(f.Data, yield, &stopped)
		case "response.output_item.added":
			return d.handleOutputItemAdded(f.Data, yield, &stopped)
		case "response.output_text.delta":
			return d.handleOutputTextDelta(f.Data, yield, &stopped)
		case "response.output_text.done":
			return d.handleOutputTextDone(f.Data, yield, &stopped)
		case "response.function_call_arguments.delta":
			return d.handleFunctionCallArgsDelta(f.Data, yield, &stopped)
		case "response.function_call_arguments.done":
			return d.handleFunctionCallArgsDone(f.Data, yield, &stopped)
		case "response.reasoning_summary_part.added":
			return d.handleReasoningSummaryPartAdded(f.Data, yield, &stopped)
		case "response.reasoning_summary_text.delta":
			return d.handleReasoningSummaryDelta(f.Data, yield, &stopped)
		case "response.reasoning_summary_text.done":
			return d.handleReasoningSummaryDone(f.Data, yield, &stopped)
		case "response.completed":
			return d.handleCompleted(f.Data, yield, &stopped)
		case "response.failed", "error":
			return d.handleErrorFrame(f.Data)
		default:
			// Quietly ignore unknown / unhandled event types; adding more is
			// additive and the protocol is open to growth.
			return nil
		}
	})
	if errors.Is(err, errIterationStopped) {
		return err
	}
	return err
}

func (d *streamDecoder) handleCreated(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		Response struct {
			Model string `json:"model"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("response.created: %w", err)
	}
	d.model = ev.Response.Model
	if !yield(llm.EventMessageStart{Model: d.model}, nil) {
		*stopped = true
		return errIterationStopped
	}
	d.emittedStart = true
	return nil
}

func (d *streamDecoder) handleOutputItemAdded(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		OutputIndex int             `json:"output_index"`
		Item        json.RawMessage `json:"item"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("output_item.added: %w", err)
	}
	var kind struct {
		Type   string `json:"type"`
		ID     string `json:"id"`
		Name   string `json:"name"`
		CallID string `json:"call_id"`
	}
	if err := json.Unmarshal(ev.Item, &kind); err != nil {
		return fmt.Errorf("output_item.added kind: %w", err)
	}

	switch kind.Type {
	case "message":
		// Emit EventTextStart lazily on the first delta — Responses can
		// send empty messages; avoid creating a hanging empty text block.
		d.itemKinds[ev.OutputIndex] = "text"
		// Allocate the block index now so deltas can reach it.
		d.outputIndexToBlock[ev.OutputIndex] = d.nextBlockIdx
		d.nextBlockIdx++
		if !yield(llm.EventTextStart{BlockIndex: d.outputIndexToBlock[ev.OutputIndex]}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "function_call":
		d.itemKinds[ev.OutputIndex] = "function_call"
		d.outputIndexToBlock[ev.OutputIndex] = d.nextBlockIdx
		d.nextBlockIdx++
		callID := kind.CallID
		if callID == "" {
			callID = kind.ID
		}
		d.pendingToolCalls[ev.OutputIndex] = toolCallMeta{id: callID, name: kind.Name}
		if !yield(llm.EventToolCallStart{
			BlockIndex: d.outputIndexToBlock[ev.OutputIndex],
			ID:         callID,
			Name:       kind.Name,
		}, nil) {
			*stopped = true
			return errIterationStopped
		}
	case "reasoning":
		// Reasoning items contain summary parts streamed separately. We
		// emit EventThinkingStart now and let the summary deltas accumulate.
		d.itemKinds[ev.OutputIndex] = "reasoning"
		d.outputIndexToBlock[ev.OutputIndex] = d.nextBlockIdx
		d.nextBlockIdx++
		if !yield(llm.EventThinkingStart{BlockIndex: d.outputIndexToBlock[ev.OutputIndex]}, nil) {
			*stopped = true
			return errIterationStopped
		}
	}
	return nil
}

func (d *streamDecoder) handleOutputTextDelta(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		OutputIndex int    `json:"output_index"`
		Delta       string `json:"delta"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("output_text.delta: %w", err)
	}
	blockIdx, ok := d.outputIndexToBlock[ev.OutputIndex]
	if !ok {
		return nil
	}
	if !yield(llm.EventTextDelta{BlockIndex: blockIdx, Delta: ev.Delta}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleOutputTextDone(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		OutputIndex int `json:"output_index"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("output_text.done: %w", err)
	}
	blockIdx, ok := d.outputIndexToBlock[ev.OutputIndex]
	if !ok {
		return nil
	}
	if !yield(llm.EventTextEnd{BlockIndex: blockIdx}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleFunctionCallArgsDelta(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		OutputIndex int    `json:"output_index"`
		Delta       string `json:"delta"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("function_call_arguments.delta: %w", err)
	}
	blockIdx, ok := d.outputIndexToBlock[ev.OutputIndex]
	if !ok {
		return nil
	}
	if !yield(llm.EventToolCallDelta{BlockIndex: blockIdx, Delta: ev.Delta}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleFunctionCallArgsDone(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		OutputIndex int    `json:"output_index"`
		Arguments   string `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("function_call_arguments.done: %w", err)
	}
	blockIdx, ok := d.outputIndexToBlock[ev.OutputIndex]
	if !ok {
		return nil
	}
	args := ev.Arguments
	if args == "" {
		args = "{}"
	}
	if !yield(llm.EventToolCallEnd{BlockIndex: blockIdx, Arguments: json.RawMessage(args)}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleReasoningSummaryPartAdded(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	// A new summary part within a reasoning item. We treat each part's
	// stream as appending to the parent thinking block — no new
	// EventThinkingStart, just keep streaming deltas into the same block.
	return nil
}

func (d *streamDecoder) handleReasoningSummaryDelta(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		OutputIndex int    `json:"output_index"`
		Delta       string `json:"delta"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("reasoning_summary_text.delta: %w", err)
	}
	blockIdx, ok := d.outputIndexToBlock[ev.OutputIndex]
	if !ok {
		return nil
	}
	if !yield(llm.EventThinkingDelta{BlockIndex: blockIdx, Delta: ev.Delta}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleReasoningSummaryDone(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		OutputIndex int `json:"output_index"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("reasoning_summary_text.done: %w", err)
	}
	blockIdx, ok := d.outputIndexToBlock[ev.OutputIndex]
	if !ok {
		return nil
	}
	// Signature is empty — Responses API doesn't expose a per-thinking-block
	// continuity token in the streaming events (uses previous_response_id
	// at the request level instead).
	if !yield(llm.EventThinkingEnd{BlockIndex: blockIdx, Signature: ""}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleCompleted(data string, yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	var ev struct {
		Response struct {
			Status            string `json:"status"`
			IncompleteDetails *struct {
				Reason string `json:"reason"`
			} `json:"incomplete_details"`
			Usage *struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
				TotalTokens  int `json:"total_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(data), &ev); err != nil {
		return fmt.Errorf("response.completed: %w", err)
	}
	incompleteReason := ""
	if ev.Response.IncompleteDetails != nil {
		incompleteReason = ev.Response.IncompleteDetails.Reason
	}
	stop := stopReasonFromStatus(ev.Response.Status, incompleteReason)
	// If any tool calls were emitted, the model is waiting on results.
	if len(d.pendingToolCalls) > 0 && stop == llm.StopReasonEnd {
		stop = llm.StopReasonToolUse
	}
	usage := llm.Usage{}
	if ev.Response.Usage != nil {
		usage.InputTokens = ev.Response.Usage.InputTokens
		usage.OutputTokens = ev.Response.Usage.OutputTokens
		usage.TotalTokens = ev.Response.Usage.TotalTokens
		if usage.TotalTokens == 0 {
			usage.TotalTokens = usage.InputTokens + usage.OutputTokens
		}
	}
	if !yield(llm.EventMessageEnd{StopReason: stop, Usage: usage}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}

func (d *streamDecoder) handleErrorFrame(data string) error {
	return &llm.APIError{
		Provider: "openai_responses",
		Status:   200, // mid-stream error has no HTTP status of its own
		Body:     []byte(data),
		Inner:    llm.ErrProvider,
	}
}
