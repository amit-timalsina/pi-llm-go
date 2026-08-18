// Package sse parses Server-Sent Events frames from an io.Reader. The
// implementation covers the subset of SSE both Anthropic and OpenAI emit:
// event/data fields, comment lines starting with ':', and the [DONE]
// sentinel used by OpenAI Chat Completions.
package sse

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// MaxFrameBytes caps one SSE line. It is a memory backstop, not a tuned
// budget: providers put payloads the client does not control on a single
// line — the Responses API's terminal frame carries the whole response
// object, and reasoning.encrypted_content is not bounded by
// max_output_tokens — so a ceiling near observed sizes is a cliff waiting
// for a bigger response. Exceeding it is llm.ErrFrameTooLarge.
const MaxFrameBytes = 32 * 1024 * 1024

// readBufferSize is what a stream actually costs. Lines longer than this
// grow on demand; nothing is preallocated to MaxFrameBytes.
const readBufferSize = 64 * 1024

// Frame is one complete SSE event. Event may be empty for sources that only
// send data lines (OpenAI Chat Completions). Data is the joined contents of
// the frame's data lines, separated by newlines per the SSE spec.
type Frame struct {
	Event string
	Data  string
}

// Read parses SSE frames from r, calling fn for each complete frame in
// order. Returns nil at EOF, the first read error otherwise. If fn returns
// an error, Read returns it immediately without consuming further input.
//
// Line length is bounded by MaxFrameBytes alone — deliberately not per
// caller. A per-call-site limit is what let the provider with the largest
// frames inherit the smallest budget and lose whole turns to
// bufio.ErrTooLong.
func Read(r io.Reader, fn func(Frame) error) error {
	br := bufio.NewReaderSize(r, readBufferSize)

	var event string
	var dataLines []string

	flush := func() error {
		if event == "" && len(dataLines) == 0 {
			return nil
		}
		f := Frame{Event: event, Data: strings.Join(dataLines, "\n")}
		event = ""
		dataLines = nil
		return fn(f)
	}

	consume := func(line string) error {
		if line == "" {
			return flush()
		}
		if strings.HasPrefix(line, ":") {
			// Comment / keep-alive line.
			return nil
		}
		idx := strings.IndexByte(line, ':')
		var field, value string
		if idx < 0 {
			// Lines without a colon are treated as field-only with empty value.
			field = line
		} else {
			field = line[:idx]
			value = strings.TrimPrefix(line[idx+1:], " ")
		}
		switch field {
		case "event":
			event = value
		case "data":
			dataLines = append(dataLines, value)
		}
		// All other fields (id, retry) ignored.
		return nil
	}

	for {
		line, err := readLine(br)
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		atEOF := errors.Is(err, io.EOF)
		if !atEOF || line != "" {
			if cerr := consume(line); cerr != nil {
				return cerr
			}
		}
		if atEOF {
			// Flush a final frame that lacks a trailing blank line.
			return flush()
		}
	}
}

// readLine returns one line with its trailing CR/LF removed. A line longer
// than the read buffer accumulates across fills, so cost tracks the line
// actually received. io.EOF accompanies a final unterminated line, matching
// bufio.Reader's own convention.
func readLine(br *bufio.Reader) (string, error) {
	chunk, err := br.ReadSlice('\n')
	if !errors.Is(err, bufio.ErrBufferFull) {
		return trimEOL(chunk), err
	}
	// ReadSlice returns a view into the reader's buffer, so copy before the
	// next fill overwrites it.
	buf := append([]byte(nil), chunk...)
	for {
		chunk, err = br.ReadSlice('\n')
		if len(buf)+len(chunk) > MaxFrameBytes {
			return "", fmt.Errorf("%w: line exceeds %d bytes", llm.ErrFrameTooLarge, MaxFrameBytes)
		}
		buf = append(buf, chunk...)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return trimEOL(buf), err
	}
}

func trimEOL(b []byte) string {
	return strings.TrimSuffix(strings.TrimSuffix(string(b), "\n"), "\r")
}
