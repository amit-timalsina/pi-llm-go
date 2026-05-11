package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/internal/sse"
)

var errIterationStopped = errors.New("iteration stopped")

// chunk is the per-frame payload OpenAI emits. Most fields are optional.
type chunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// streamDecoder owns per-stream state. OpenAI streams interleave a text
// block (if any) with one or more tool_calls (each identified by index).
// We synthesize llm.BlockIndex values: text is always 0 if present, then
// tool calls follow in toolIndex order starting at 1 (or 0 if no text).
type streamDecoder struct {
	emittedStart bool
	model        string

	// textOpened becomes true on the first non-empty content chunk; we emit
	// EventTextStart lazily so messages with only tool calls don't get a
	// leading empty text block.
	textOpened   bool
	textBlockIdx int
	finishReason string

	// toolCalls tracks one entry per tool_call index. Each entry knows its
	// block index (post-text-block-or-0), accumulated arguments, and start-
	// emitted flag.
	toolCalls map[int]*toolCallState

	// nextBlockIdx is the next free pi-llm-go BlockIndex to assign.
	nextBlockIdx int

	usage llm.Usage
}

type toolCallState struct {
	blockIdx     int
	id           string
	name         string
	args         strings.Builder
	startEmitted bool
}

func newStreamDecoder() *streamDecoder {
	return &streamDecoder{
		toolCalls: map[int]*toolCallState{},
	}
}

func (d *streamDecoder) decode(r io.Reader, yield func(llm.StreamEvent, error) bool) error {
	stopped := false
	err := sse.Read(r, 0, func(f sse.Frame) error {
		if stopped {
			return errIterationStopped
		}
		if f.Data == "[DONE]" {
			return d.finalize(yield, &stopped)
		}
		var c chunk
		if err := json.Unmarshal([]byte(f.Data), &c); err != nil {
			return fmt.Errorf("decode chunk: %w", err)
		}

		if !d.emittedStart {
			d.model = c.Model
			if !yield(llm.EventMessageStart{Model: d.model}, nil) {
				stopped = true
				return errIterationStopped
			}
			d.emittedStart = true
			d.nextBlockIdx = 0
		}

		if c.Usage != nil {
			d.usage = llm.Usage{
				InputTokens:  c.Usage.PromptTokens,
				OutputTokens: c.Usage.CompletionTokens,
				TotalTokens:  c.Usage.TotalTokens,
			}
		}

		for _, choice := range c.Choices {
			if choice.Delta.Content != "" {
				if !d.textOpened {
					d.textBlockIdx = d.nextBlockIdx
					d.nextBlockIdx++
					if !yield(llm.EventTextStart{BlockIndex: d.textBlockIdx}, nil) {
						stopped = true
						return errIterationStopped
					}
					d.textOpened = true
				}
				if !yield(llm.EventTextDelta{BlockIndex: d.textBlockIdx, Delta: choice.Delta.Content}, nil) {
					stopped = true
					return errIterationStopped
				}
			}

			for _, tcDelta := range choice.Delta.ToolCalls {
				state, ok := d.toolCalls[tcDelta.Index]
				if !ok {
					state = &toolCallState{blockIdx: d.nextBlockIdx}
					d.nextBlockIdx++
					d.toolCalls[tcDelta.Index] = state
				}
				if tcDelta.ID != "" {
					state.id = tcDelta.ID
				}
				if tcDelta.Function.Name != "" {
					state.name = tcDelta.Function.Name
				}
				if !state.startEmitted && state.id != "" && state.name != "" {
					if !yield(llm.EventToolCallStart{
						BlockIndex: state.blockIdx,
						ID:         state.id,
						Name:       state.name,
					}, nil) {
						stopped = true
						return errIterationStopped
					}
					state.startEmitted = true
				}
				if tcDelta.Function.Arguments != "" {
					state.args.WriteString(tcDelta.Function.Arguments)
					if !yield(llm.EventToolCallDelta{
						BlockIndex: state.blockIdx,
						Delta:      tcDelta.Function.Arguments,
					}, nil) {
						stopped = true
						return errIterationStopped
					}
				}
			}

			if choice.FinishReason != nil && *choice.FinishReason != "" {
				d.finishReason = *choice.FinishReason
			}
		}
		return nil
	})
	if errors.Is(err, errIterationStopped) {
		return err
	}
	return err
}

// finalize is invoked on `data: [DONE]`. It closes any open text or
// tool-call blocks and emits the terminal EventMessageEnd.
func (d *streamDecoder) finalize(yield func(llm.StreamEvent, error) bool, stopped *bool) error {
	if d.textOpened {
		if !yield(llm.EventTextEnd{BlockIndex: d.textBlockIdx}, nil) {
			*stopped = true
			return errIterationStopped
		}
	}
	// Close tool calls in BlockIndex order so consumers see deterministic
	// block ordering.
	type indexed struct {
		blockIdx int
		state    *toolCallState
	}
	var ordered []indexed
	for _, state := range d.toolCalls {
		ordered = append(ordered, indexed{blockIdx: state.blockIdx, state: state})
	}
	// Simple insertion sort by blockIdx — tool calls are rare and small.
	for i := 1; i < len(ordered); i++ {
		for j := i; j > 0 && ordered[j-1].blockIdx > ordered[j].blockIdx; j-- {
			ordered[j-1], ordered[j] = ordered[j], ordered[j-1]
		}
	}
	for _, item := range ordered {
		args := strings.TrimSpace(item.state.args.String())
		if args == "" {
			args = "{}"
		}
		if !yield(llm.EventToolCallEnd{
			BlockIndex: item.state.blockIdx,
			Arguments:  json.RawMessage(args),
		}, nil) {
			*stopped = true
			return errIterationStopped
		}
	}
	stop, contentFiltered := stopReasonFromAPI(d.finishReason)
	if contentFiltered != nil {
		return &llm.APIError{
			Provider: "openai",
			Status:   200,
			Inner:    llm.ErrProvider,
		}
	}
	if !yield(llm.EventMessageEnd{StopReason: stop, Usage: d.usage}, nil) {
		*stopped = true
		return errIterationStopped
	}
	return nil
}
