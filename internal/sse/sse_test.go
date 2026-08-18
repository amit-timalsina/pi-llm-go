package sse

import (
	"errors"
	"strings"
	"testing"

	llm "github.com/amit-timalsina/pi-llm-go"
)

func TestReadSingleFrame(t *testing.T) {
	in := "event: foo\ndata: hello\n\n"
	var got []Frame
	err := Read(strings.NewReader(in), func(f Frame) error {
		got = append(got, f)
		return nil
	})
	if err != nil {
		t.Fatalf("Read returned error: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1 frame, got %d", len(got))
	}
	if got[0].Event != "foo" || got[0].Data != "hello" {
		t.Fatalf("unexpected frame: %+v", got[0])
	}
}

func TestReadMultilineData(t *testing.T) {
	in := "event: foo\ndata: line1\ndata: line2\n\n"
	var got []Frame
	_ = Read(strings.NewReader(in), func(f Frame) error {
		got = append(got, f)
		return nil
	})
	if len(got) != 1 || got[0].Data != "line1\nline2" {
		t.Fatalf("multiline join failed: %+v", got)
	}
}

func TestReadCommentLinesIgnored(t *testing.T) {
	in := ": ping\nevent: foo\n: another comment\ndata: x\n\n"
	var got []Frame
	_ = Read(strings.NewReader(in), func(f Frame) error {
		got = append(got, f)
		return nil
	})
	if len(got) != 1 || got[0].Data != "x" {
		t.Fatalf("comment handling failed: %+v", got)
	}
}

func TestReadDataOnly(t *testing.T) {
	in := "data: hello\n\ndata: world\n\n"
	var got []Frame
	_ = Read(strings.NewReader(in), func(f Frame) error {
		got = append(got, f)
		return nil
	})
	if len(got) != 2 {
		t.Fatalf("want 2 frames, got %d", len(got))
	}
	if got[0].Event != "" || got[0].Data != "hello" {
		t.Fatalf("first frame wrong: %+v", got[0])
	}
	if got[1].Event != "" || got[1].Data != "world" {
		t.Fatalf("second frame wrong: %+v", got[1])
	}
}

func TestReadFinalFrameWithoutBlankLine(t *testing.T) {
	in := "event: foo\ndata: tail"
	var got []Frame
	_ = Read(strings.NewReader(in), func(f Frame) error {
		got = append(got, f)
		return nil
	})
	if len(got) != 1 || got[0].Data != "tail" {
		t.Fatalf("trailing frame not flushed: %+v", got)
	}
}

func TestReadCallbackErrorPropagates(t *testing.T) {
	in := "event: a\ndata: x\n\nevent: b\ndata: y\n\n"
	want := errors.New("stop")
	count := 0
	err := Read(strings.NewReader(in), func(f Frame) error {
		count++
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("want callback error, got %v", err)
	}
	if count != 1 {
		t.Fatalf("expected callback to stop after 1 frame, got %d", count)
	}
}

// The size that used to die with "bufio.Scanner: token too long", plus the
// sizes either side of the old 1 MB ceiling. A frame the client did not ask
// for the size of must not destroy the turn.
func TestReadLinesPastTheOldCeiling(t *testing.T) {
	for _, size := range []int{
		900 * 1024,       // fit under the old ceiling
		1024 * 1024,      // the exact size that failed
		4 * 1024 * 1024,  // past gemini's old explicit limit
		12 * 1024 * 1024, // a large encrypted_content blob
	} {
		blob := strings.Repeat("x", size)
		in := "event: response.completed\ndata: {\"blob\":\"" + blob + "\"}\n\n"
		var frames []Frame
		if err := Read(strings.NewReader(in), func(f Frame) error {
			frames = append(frames, f)
			return nil
		}); err != nil {
			t.Fatalf("size=%dKB: %v", size/1024, err)
		}
		if len(frames) != 1 {
			t.Fatalf("size=%dKB: frames=%d, want 1", size/1024, len(frames))
		}
		if frames[0].Event != "response.completed" {
			t.Errorf("size=%dKB: event=%q", size/1024, frames[0].Event)
		}
		if !strings.Contains(frames[0].Data, blob) {
			t.Errorf("size=%dKB: payload did not survive intact", size/1024)
		}
	}
}

// Beyond the backstop the caller gets a typed error it can distinguish from
// a transport fault, not a bare bufio one.
func TestReadBeyondMaxFrameBytesIsTyped(t *testing.T) {
	in := "data: " + strings.Repeat("y", MaxFrameBytes+1) + "\n\n"
	err := Read(strings.NewReader(in), func(Frame) error { return nil })
	if !errors.Is(err, llm.ErrFrameTooLarge) {
		t.Fatalf("want llm.ErrFrameTooLarge, got %v", err)
	}
	if !errors.Is(err, llm.ErrProvider) {
		t.Errorf("must stay under ErrProvider: %v", err)
	}
	if llm.IsRetriable(err) {
		t.Error("the same request produces the same frame; must not be retriable")
	}
}

// A long line split across many buffer fills must reassemble byte-for-byte,
// including its interior newline-free content and CRLF endings.
func TestReadLongLineWithCRLFReassembles(t *testing.T) {
	blob := strings.Repeat("abcdefgh", 200_000) // 1.6 MB, no newlines
	in := "event: e\r\ndata: " + blob + "\r\n\r\n"
	var frames []Frame
	if err := Read(strings.NewReader(in), func(f Frame) error {
		frames = append(frames, f)
		return nil
	}); err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(frames) != 1 {
		t.Fatalf("frames=%d, want 1", len(frames))
	}
	if frames[0].Data != blob {
		t.Errorf("data length=%d, want %d", len(frames[0].Data), len(blob))
	}
	if frames[0].Event != "e" {
		t.Errorf("event=%q, want e", frames[0].Event)
	}
}
