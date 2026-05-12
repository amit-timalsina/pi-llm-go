// Package files implements a minimal Gemini Files API client —
// Upload, Wait (until ACTIVE), Get, Delete. The Files API lets
// callers stage large media payloads (>20 MB videos, audio, PDFs,
// images) once and reference them by URI in subsequent
// generateContent calls. Files expire after ~48 hours (the actual
// value is on FileRef.ExpirationTime).
//
// This is a separate sub-package from providers/gemini deliberately:
// the upload lifecycle is async (state PROCESSING → ACTIVE | FAILED,
// gated by polling) and has nothing to do with the synchronous
// streaming model the LLM interface exposes. Forcing the two into
// one package would warp both abstractions.
//
// Usage:
//
//	f := files.New(files.Options{APIKey: key})
//	ref, err := f.Upload(ctx, videoReader, "video/mp4", files.UploadOptions{
//	    DisplayName: "demo.mp4",
//	})
//	if err != nil { /* handle */ }
//	ref, err = f.Wait(ctx, ref) // blocks until ACTIVE
//	if err != nil { /* handle */ }
//	defer f.Delete(context.Background(), ref.Name)
//
//	// Now hand ref.URI to a VideoBlock:
//	llm.VideoBlock{URI: ref.URI}
//
// The Files API requires an API key (header x-goog-api-key); OAuth /
// Vertex AI uploads are out of scope for this sub-package. Multipart
// uploads support files up to ~2 GB; resumable uploads (for larger
// payloads) are a future addition.
package files

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strconv"
	"time"
)

// DefaultBaseURL is the standard Google AI endpoint. Override via
// Options.BaseURL for proxies, mocks, or non-standard regional
// endpoints.
const DefaultBaseURL = "https://generativelanguage.googleapis.com"

// Options configures the Files client. Mirrors gemini.Options so
// callers can share configuration; the two structs are NOT
// interchangeable (separate types keep the import graph clean).
type Options struct {
	// APIKey is the Google AI API key. Sent as the x-goog-api-key
	// header. Required.
	APIKey string

	// BaseURL overrides DefaultBaseURL.
	BaseURL string

	// HTTPClient overrides the default net/http client. Required for
	// OpenTelemetry instrumentation, custom timeouts, or test fakes.
	HTTPClient *http.Client
}

// Client is the Files API client.
type Client struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

// New constructs a Client. Returns an error if APIKey is empty.
func New(opts Options) (*Client, error) {
	if opts.APIKey == "" {
		return nil, errors.New("files: APIKey is required")
	}
	base := opts.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{
		apiKey:  opts.APIKey,
		baseURL: base,
		http:    client,
	}, nil
}

// State is the Gemini file lifecycle. New uploads start as Processing,
// transition to Active once the server has indexed the content (videos
// take seconds to minutes; small files are often Active immediately),
// or Failed if the upload was malformed.
type State string

const (
	StateProcessing State = "PROCESSING"
	StateActive     State = "ACTIVE"
	StateFailed     State = "FAILED"
)

// FileRef is the typed view of a Gemini file. The URI field is what
// callers plug into llm.VideoBlock.URI (or any other content block
// the Gemini provider's fileData path supports).
type FileRef struct {
	// Name is the file resource id (e.g. "files/abc123"). Used as
	// the path-suffix for Get / Delete.
	Name string
	// DisplayName is the optional human-readable label set on upload.
	DisplayName string
	// URI is the absolute URL the Gemini provider uses to reference
	// the file in a fileData part. Hand this to VideoBlock.URI.
	URI string
	// MimeType is what the server inferred or what the caller passed.
	MimeType string
	// SizeBytes is the file size; useful for cost / quota tracking.
	SizeBytes int64
	// State is the current lifecycle position (see State godoc).
	State State
	// CreateTime is when the upload completed server-side.
	CreateTime time.Time
	// ExpirationTime is when the server will reclaim the file
	// (typically ~48h from creation). Callers must re-upload or
	// extend before this lapses.
	ExpirationTime time.Time
	// SHA256Hash is the base64-encoded SHA-256 of the file body.
	// Useful for caching / dedup against client-side state.
	SHA256Hash string
	// Source is "UPLOADED" for caller uploads or "GENERATED" for
	// files Gemini produced (e.g. via image generation).
	Source string
}

// UploadOptions configures a single Upload call. All fields optional.
type UploadOptions struct {
	// DisplayName sets a human-readable label on the file. Defaults
	// to empty.
	DisplayName string
}

// Upload sends bytes from r to the Files API and returns the
// resulting FileRef. Multipart protocol; supports files up to ~2 GB.
// The returned ref's State may be Processing — call Wait to block
// until Active before using the URI in a generateContent call.
//
// mimeType MUST be the standard MIME identifier (e.g. "video/mp4",
// "image/png"). The server stores and round-trips it; downstream
// VideoBlock / ImageBlock parts inherit it.
//
// On non-2xx response, returns the raw body text in the error for
// debuggability.
func (c *Client) Upload(ctx context.Context, r io.Reader, mimeType string, opts UploadOptions) (*FileRef, error) {
	if mimeType == "" {
		return nil, errors.New("files.Upload: mimeType is required")
	}

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)

	// Part 1: file metadata as application/json. Without setting
	// Content-Type via CreatePart's header, multipart defaults the
	// part to application/octet-stream which Gemini rejects.
	metaHdr := textproto.MIMEHeader{}
	metaHdr.Set("Content-Type", "application/json")
	metaPart, err := mw.CreatePart(metaHdr)
	if err != nil {
		return nil, fmt.Errorf("files.Upload: create metadata part: %w", err)
	}
	meta := uploadMetadata{File: uploadMetadataFile(opts)}
	if err := json.NewEncoder(metaPart).Encode(meta); err != nil {
		return nil, fmt.Errorf("files.Upload: encode metadata: %w", err)
	}

	// Part 2: file bytes with caller-supplied MIME.
	dataHdr := textproto.MIMEHeader{}
	dataHdr.Set("Content-Type", mimeType)
	dataPart, err := mw.CreatePart(dataHdr)
	if err != nil {
		return nil, fmt.Errorf("files.Upload: create data part: %w", err)
	}
	if _, err := io.Copy(dataPart, r); err != nil {
		return nil, fmt.Errorf("files.Upload: copy file bytes: %w", err)
	}
	if err := mw.Close(); err != nil {
		return nil, fmt.Errorf("files.Upload: close multipart: %w", err)
	}

	url := c.baseURL + "/upload/v1beta/files?uploadType=multipart"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return nil, fmt.Errorf("files.Upload: new request: %w", err)
	}
	req.Header.Set("x-goog-api-key", c.apiKey)
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("files.Upload: do request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("files.Upload: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var wrap struct {
		File apiFile `json:"file"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wrap); err != nil {
		return nil, fmt.Errorf("files.Upload: decode response: %w", err)
	}
	return wrap.File.toRef(), nil
}

// WaitOptions tunes Wait. All fields optional.
type WaitOptions struct {
	// PollInterval is the wait between state polls. Defaults to 2s.
	// Set lower for short-lived uploads in tests; never below 200ms
	// (server-side rate limits).
	PollInterval time.Duration
}

const defaultWaitPollInterval = 2 * time.Second

// Wait polls the Files API until ref's state reaches Active or Failed,
// or ctx is cancelled. Returns the latest FileRef the server reported.
// Wait does NOT mutate the input ref.
//
// Use ctx with a deadline to bound the wait; videos may take minutes
// to process. A short ctx timeout is preferable to a small MaxAttempts
// option — the deadline is observable from outside, the attempt count
// isn't.
//
// On Failed state, returns (ref, error) where ref has State=Failed.
func (c *Client) Wait(ctx context.Context, ref *FileRef, opts ...WaitOptions) (*FileRef, error) {
	if ref == nil || ref.Name == "" {
		return nil, errors.New("files.Wait: ref.Name is required")
	}
	interval := defaultWaitPollInterval
	if len(opts) > 0 && opts[0].PollInterval > 0 {
		interval = opts[0].PollInterval
	}

	for {
		current, err := c.Get(ctx, ref.Name)
		if err != nil {
			return nil, fmt.Errorf("files.Wait: %w", err)
		}
		switch current.State {
		case StateActive:
			return current, nil
		case StateFailed:
			return current, fmt.Errorf("files.Wait: file %s failed processing", current.Name)
		case StateProcessing, "":
			// Keep polling. Empty state defends against a weird
			// server response — treat unknown as "keep trying."
		default:
			return current, fmt.Errorf("files.Wait: unexpected state %q on %s", current.State, current.Name)
		}

		select {
		case <-ctx.Done():
			return current, fmt.Errorf("files.Wait: %w (last state: %s)", ctx.Err(), current.State)
		case <-time.After(interval):
		}
	}
}

// Get fetches the current state of a file by name (e.g. "files/abc123").
func (c *Client) Get(ctx context.Context, name string) (*FileRef, error) {
	if name == "" {
		return nil, errors.New("files.Get: name is required")
	}
	url := c.baseURL + "/v1beta/" + name
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("files.Get: new request: %w", err)
	}
	req.Header.Set("x-goog-api-key", c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("files.Get: do request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("files.Get: HTTP %d: %s", resp.StatusCode, string(respBody))
	}
	var f apiFile
	if err := json.NewDecoder(resp.Body).Decode(&f); err != nil {
		return nil, fmt.Errorf("files.Get: decode response: %w", err)
	}
	return f.toRef(), nil
}

// Delete removes a file from Gemini's storage. Returns nil on
// success (HTTP 200, body "{}"). Idempotent: deleting an
// already-deleted file returns a 4xx that surfaces as an error;
// callers building defer-cleanup paths may want to wrap with
// errors.Is-style 404 detection in a future PR.
func (c *Client) Delete(ctx context.Context, name string) error {
	if name == "" {
		return errors.New("files.Delete: name is required")
	}
	url := c.baseURL + "/v1beta/" + name
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return fmt.Errorf("files.Delete: new request: %w", err)
	}
	req.Header.Set("x-goog-api-key", c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("files.Delete: do request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("files.Delete: HTTP %d: %s", resp.StatusCode, string(respBody))
	}
	return nil
}

// --- internal wire-format helpers ---

type uploadMetadata struct {
	File uploadMetadataFile `json:"file"`
}

type uploadMetadataFile struct {
	DisplayName string `json:"displayName,omitempty"`
}

// apiFile mirrors Gemini's File resource on the wire. SizeBytes
// arrives as a JSON string (Gemini's pattern for 64-bit ints) so we
// decode it via a string field, then parse to int64.
type apiFile struct {
	Name           string `json:"name"`
	DisplayName    string `json:"displayName,omitempty"`
	MimeType       string `json:"mimeType"`
	SizeBytes      string `json:"sizeBytes"`
	CreateTime     string `json:"createTime"`
	UpdateTime     string `json:"updateTime"`
	ExpirationTime string `json:"expirationTime"`
	SHA256Hash     string `json:"sha256Hash"`
	URI            string `json:"uri"`
	State          string `json:"state"`
	Source         string `json:"source,omitempty"`
}

func (a apiFile) toRef() *FileRef {
	size, _ := strconv.ParseInt(a.SizeBytes, 10, 64)
	create, _ := time.Parse(time.RFC3339Nano, a.CreateTime)
	expire, _ := time.Parse(time.RFC3339Nano, a.ExpirationTime)
	return &FileRef{
		Name:           a.Name,
		DisplayName:    a.DisplayName,
		URI:            a.URI,
		MimeType:       a.MimeType,
		SizeBytes:      size,
		State:          State(a.State),
		CreateTime:     create,
		ExpirationTime: expire,
		SHA256Hash:     a.SHA256Hash,
		Source:         a.Source,
	}
}
