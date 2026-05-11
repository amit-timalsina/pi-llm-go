package sse

import (
	"errors"
	"strings"
	"testing"
)

func TestReadSingleFrame(t *testing.T) {
	in := "event: foo\ndata: hello\n\n"
	var got []Frame
	err := Read(strings.NewReader(in), 0, func(f Frame) error {
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
	_ = Read(strings.NewReader(in), 0, func(f Frame) error {
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
	_ = Read(strings.NewReader(in), 0, func(f Frame) error {
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
	_ = Read(strings.NewReader(in), 0, func(f Frame) error {
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
	_ = Read(strings.NewReader(in), 0, func(f Frame) error {
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
	err := Read(strings.NewReader(in), 0, func(f Frame) error {
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
