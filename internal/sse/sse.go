// Package sse parses Server-Sent Events frames from an io.Reader. The
// implementation covers the subset of SSE both Anthropic and OpenAI emit:
// event/data fields, comment lines starting with ':', and the [DONE]
// sentinel used by OpenAI Chat Completions.
package sse

import (
	"bufio"
	"io"
	"strings"
)

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
// maxLine caps the largest line bufio.Scanner will accept; SSE producers
// can emit large data lines for tool-call JSON, so callers may need to set
// this higher than the bufio default (64KB).
func Read(r io.Reader, maxLine int, fn func(Frame) error) error {
	scanner := bufio.NewScanner(r)
	if maxLine <= 0 {
		maxLine = 1024 * 1024
	}
	scanner.Buffer(make([]byte, 0, 4096), maxLine)

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

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			// Comment / keep-alive line.
			continue
		}
		idx := strings.IndexByte(line, ':')
		var field, value string
		if idx < 0 {
			// Lines without a colon are treated as field-only with empty value.
			field = line
		} else {
			field = line[:idx]
			value = line[idx+1:]
			// Per spec, single leading space is stripped.
			value = strings.TrimPrefix(value, " ")
		}
		switch field {
		case "event":
			event = value
		case "data":
			dataLines = append(dataLines, value)
		}
		// All other fields (id, retry) ignored.
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	// Flush a final frame that lacks a trailing blank line.
	return flush()
}
