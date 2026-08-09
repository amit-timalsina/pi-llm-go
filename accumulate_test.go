package llm_test

import (
	"encoding/json"
	"errors"
	"iter"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// fromEvents returns an iter.Seq2 that yields the given events in order
// followed by an optional terminal error.
func fromEvents(events []llm.StreamEvent, terminalErr error) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		for _, e := range events {
			if !yield(e, nil) {
				return
			}
		}
		if terminalErr != nil {
			yield(nil, terminalErr)
		}
	}
}

func TestAccumulateBuildsFinalMessage(t *testing.T) {
	events := []llm.StreamEvent{
		llm.EventMessageStart{Model: "claude-opus-4-7"},
		llm.EventTextStart{BlockIndex: 0},
		llm.EventTextDelta{BlockIndex: 0, Delta: "Hello "},
		llm.EventTextDelta{BlockIndex: 0, Delta: "world."},
		llm.EventTextEnd{BlockIndex: 0, Signature: "text-signature"},
		llm.EventMessageEnd{
			StopReason: llm.StopReasonEnd,
			Usage:      llm.Usage{InputTokens: 10, OutputTokens: 3, TotalTokens: 13},
		},
	}

	var final *llm.Message
	for msg, err := range llm.Accumulate(fromEvents(events, nil)) {
		if err != nil {
			t.Fatalf("Accumulate yielded error: %v", err)
		}
		final = msg
	}
	if final == nil {
		t.Fatal("final message is nil")
	}
	if final.Model != "claude-opus-4-7" {
		t.Errorf("Model=%q", final.Model)
	}
	if len(final.Content) != 1 {
		t.Fatalf("Content len=%d", len(final.Content))
	}
	tb, ok := final.Content[0].(llm.TextBlock)
	if !ok {
		t.Fatalf("Content[0] type=%T", final.Content[0])
	}
	if tb.Text != "Hello world." {
		t.Errorf("Text=%q", tb.Text)
	}
	if tb.Signature != "text-signature" {
		t.Errorf("Signature=%q", tb.Signature)
	}
	if final.StopReason != llm.StopReasonEnd {
		t.Errorf("StopReason=%v", final.StopReason)
	}
	if final.Usage.TotalTokens != 13 {
		t.Errorf("Usage=%+v", final.Usage)
	}
}

func TestAccumulateMultipleBlocks(t *testing.T) {
	events := []llm.StreamEvent{
		llm.EventMessageStart{Model: "x"},
		llm.EventTextStart{BlockIndex: 0},
		llm.EventTextDelta{BlockIndex: 0, Delta: "thinking out loud"},
		llm.EventTextEnd{BlockIndex: 0},
		llm.EventToolCallStart{BlockIndex: 1, ID: "tu_1", Name: "search"},
		llm.EventToolCallDelta{BlockIndex: 1, Delta: `{"q":`},
		llm.EventToolCallDelta{BlockIndex: 1, Delta: `"go iterators"}`},
		llm.EventToolCallEnd{BlockIndex: 1, Arguments: json.RawMessage(`{"q":"go iterators"}`), Signature: "tool-signature"},
		llm.EventMessageEnd{StopReason: llm.StopReasonToolUse, Usage: llm.Usage{}},
	}
	var final *llm.Message
	for msg, err := range llm.Accumulate(fromEvents(events, nil)) {
		if err != nil {
			t.Fatalf("Accumulate err: %v", err)
		}
		final = msg
	}
	if final == nil || len(final.Content) != 2 {
		t.Fatalf("want 2 blocks, got: %+v", final)
	}
	if _, ok := final.Content[0].(llm.TextBlock); !ok {
		t.Errorf("Content[0] type=%T", final.Content[0])
	}
	tc, ok := final.Content[1].(llm.ToolCallBlock)
	if !ok {
		t.Fatalf("Content[1] type=%T", final.Content[1])
	}
	if tc.ID != "tu_1" || tc.Name != "search" {
		t.Errorf("ToolCallBlock fields: %+v", tc)
	}
	if tc.Signature != "tool-signature" {
		t.Errorf("ToolCallBlock.Signature=%q", tc.Signature)
	}
	var args struct{ Q string }
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("Unmarshal Arguments: %v", err)
	}
	if args.Q != "go iterators" {
		t.Errorf("args.Q=%q", args.Q)
	}
}

func TestAccumulatePropagatesError(t *testing.T) {
	sentinel := errors.New("transport blew up")
	events := []llm.StreamEvent{
		llm.EventMessageStart{Model: "x"},
		llm.EventTextStart{BlockIndex: 0},
		llm.EventTextDelta{BlockIndex: 0, Delta: "partial..."},
	}
	var seenErr error
	var snapshots int
	for msg, err := range llm.Accumulate(fromEvents(events, sentinel)) {
		if err != nil {
			seenErr = err
			// On error, the final yielded msg is still meaningful (partial).
			if msg == nil {
				t.Error("expected non-nil partial snapshot on error")
			}
			break
		}
		snapshots++
	}
	if !errors.Is(seenErr, sentinel) {
		t.Errorf("err=%v, want %v", seenErr, sentinel)
	}
	if snapshots == 0 {
		t.Error("expected at least one pre-error snapshot")
	}
}

func TestAccumulateSnapshotsAreIndependent(t *testing.T) {
	// Holding onto an earlier snapshot must not be mutated by later events.
	events := []llm.StreamEvent{
		llm.EventMessageStart{Model: "x"},
		llm.EventTextStart{BlockIndex: 0},
		llm.EventTextDelta{BlockIndex: 0, Delta: "first"},
		llm.EventTextDelta{BlockIndex: 0, Delta: "-second"},
		llm.EventTextEnd{BlockIndex: 0},
		llm.EventMessageEnd{StopReason: llm.StopReasonEnd},
	}
	var snapshots []*llm.Message
	for msg, err := range llm.Accumulate(fromEvents(events, nil)) {
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		snapshots = append(snapshots, msg)
	}
	if len(snapshots) < 3 {
		t.Fatalf("want >=3 snapshots, got %d", len(snapshots))
	}
	// The snapshot right after the first delta should show only "first";
	// the one after the second should be "first-second". We retained both.
	mid := snapshots[2] // after first text_delta
	final := snapshots[len(snapshots)-1]
	midText, _ := mid.Content[0].(llm.TextBlock)
	finalText, _ := final.Content[0].(llm.TextBlock)
	if midText.Text != "first" {
		t.Errorf("mid snapshot text=%q", midText.Text)
	}
	if finalText.Text != "first-second" {
		t.Errorf("final snapshot text=%q", finalText.Text)
	}
}

// Past Content's end used to panic; in range over another kind used to
// clobber it silently.
func TestAccumulateUnmatchedToolCallEnd(t *testing.T) {
	cases := []struct {
		name   string
		events []llm.StreamEvent
	}{
		{
			name: "index past end of content",
			events: []llm.StreamEvent{
				llm.EventMessageStart{Model: "m"},
				llm.EventToolCallEnd{BlockIndex: 0, Arguments: json.RawMessage(`{"a":1}`)},
			},
		},
		{
			name: "index over a finished text block",
			events: []llm.StreamEvent{
				llm.EventMessageStart{Model: "m"},
				llm.EventTextStart{BlockIndex: 0},
				llm.EventTextDelta{BlockIndex: 0, Delta: "answer"},
				llm.EventTextEnd{BlockIndex: 0},
				llm.EventToolCallEnd{BlockIndex: 0, Arguments: json.RawMessage(`{"a":1}`)},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotErr error
			var last *llm.Message
			for msg, err := range llm.Accumulate(fromEvents(tc.events, nil)) {
				last = msg
				if err != nil {
					gotErr = err
					break
				}
			}
			if gotErr == nil {
				t.Fatalf("want ErrMalformedStream, got nil (message=%+v)", last)
			}
			if !errors.Is(gotErr, llm.ErrMalformedStream) {
				t.Errorf("errors.Is(err, ErrMalformedStream)=false: %v", gotErr)
			}
			if !errors.Is(gotErr, llm.ErrProvider) {
				t.Errorf("ErrMalformedStream must stay under ErrProvider: %v", gotErr)
			}
			if llm.IsRetriable(gotErr) {
				t.Errorf("malformed stream must not be retriable: %v", gotErr)
			}
		})
	}
}

func TestAccumulateWellFormedToolCallUnaffected(t *testing.T) {
	events := []llm.StreamEvent{
		llm.EventMessageStart{Model: "m"},
		llm.EventToolCallStart{BlockIndex: 0, ID: "call_1", Name: "add"},
		llm.EventToolCallDelta{BlockIndex: 0, Delta: `{"a":1}`},
		llm.EventToolCallEnd{BlockIndex: 0, Arguments: json.RawMessage(`{"a":1}`)},
		llm.EventMessageEnd{StopReason: llm.StopReasonToolUse},
	}
	var final *llm.Message
	for msg, err := range llm.Accumulate(fromEvents(events, nil)) {
		if err != nil {
			t.Fatalf("Accumulate: %v", err)
		}
		final = msg
	}
	tc, ok := final.Content[0].(llm.ToolCallBlock)
	if !ok {
		t.Fatalf("Content[0]=%T, want ToolCallBlock", final.Content[0])
	}
	if tc.ID != "call_1" || tc.Name != "add" || string(tc.Arguments) != `{"a":1}` {
		t.Errorf("ToolCallBlock=%+v", tc)
	}
}
