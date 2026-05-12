package files_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/gemini/files"
)

// fakeAPI is a minimal in-memory stand-in for the Files API. Records
// every request for assertion-side inspection; lets each test set
// canned responses keyed by method+path.
type fakeAPI struct {
	t            *testing.T
	uploadRespFn http.HandlerFunc                // override for upload
	getRespFn    func(name string) (int, string) // status, body keyed by name
	deleteRespFn func(name string) (int, string)

	// captured for inspection
	uploadParts     map[string][]byte // part header key -> body
	uploadHeader    http.Header
	uploadCalled    int
	getCalledFor    []string
	deleteCalledFor []string
}

func newFakeAPI(t *testing.T) *fakeAPI {
	return &fakeAPI{t: t}
}

func (f *fakeAPI) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/upload/v1beta/files":
			f.uploadCalled++
			f.uploadHeader = r.Header.Clone()
			// Parse the multipart body so tests can assert on parts.
			contentType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
			if err != nil || !strings.HasPrefix(contentType, "multipart/") {
				http.Error(w, "expected multipart", http.StatusBadRequest)
				return
			}
			mr := multipart.NewReader(r.Body, params["boundary"])
			f.uploadParts = map[string][]byte{}
			for {
				part, err := mr.NextPart()
				if err == io.EOF {
					break
				}
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				body, _ := io.ReadAll(part)
				key := part.Header.Get("Content-Type")
				f.uploadParts[key] = body
				_ = part.Close()
			}
			if f.uploadRespFn != nil {
				f.uploadRespFn(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, defaultUploadResponse)

		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1beta/files/"):
			name := strings.TrimPrefix(r.URL.Path, "/v1beta/")
			f.getCalledFor = append(f.getCalledFor, name)
			if f.getRespFn != nil {
				status, body := f.getRespFn(name)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = io.WriteString(w, body)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, fmt.Sprintf(`{"name":%q,"state":"ACTIVE","mimeType":"text/plain","sizeBytes":"42","uri":"%s/v1beta/%s","createTime":"2026-05-12T06:00:00Z","expirationTime":"2026-05-14T06:00:00Z","sha256Hash":"abc==","source":"UPLOADED"}`,
				name, "https://fake", name))

		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/v1beta/files/"):
			name := strings.TrimPrefix(r.URL.Path, "/v1beta/")
			f.deleteCalledFor = append(f.deleteCalledFor, name)
			if f.deleteRespFn != nil {
				status, body := f.deleteRespFn(name)
				w.WriteHeader(status)
				_, _ = io.WriteString(w, body)
				return
			}
			_, _ = io.WriteString(w, "{}")

		default:
			f.t.Logf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
		}
	}
}

const defaultUploadResponse = `{"file":{
	"name":"files/abc123",
	"displayName":"smoke.txt",
	"mimeType":"text/plain",
	"sizeBytes":"42",
	"createTime":"2026-05-12T06:00:00Z",
	"updateTime":"2026-05-12T06:00:00Z",
	"expirationTime":"2026-05-14T06:00:00Z",
	"sha256Hash":"abc==",
	"uri":"https://fake/v1beta/files/abc123",
	"state":"ACTIVE",
	"source":"UPLOADED"
}}`

func newClient(t *testing.T, srv *httptest.Server) *files.Client {
	t.Helper()
	c, err := files.New(files.Options{APIKey: "test-key", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

// --- Tests ---

func TestNew_RequiresAPIKey(t *testing.T) {
	if _, err := files.New(files.Options{}); err == nil {
		t.Error("expected error when APIKey is empty")
	}
}

// TestUpload_WireShape verifies the multipart upload format:
// - x-goog-api-key header
// - Content-Type starts with multipart/related (or multipart/form-data; Go's writer)
// - Two parts: application/json metadata + caller-MIME data
// - metadata includes displayName when set
// - data body round-trips byte-for-byte
func TestUpload_WireShape(t *testing.T) {
	fs := newFakeAPI(t)
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	payload := []byte("hello multipart")
	ref, err := c.Upload(context.Background(), bytes.NewReader(payload), "text/plain", files.UploadOptions{
		DisplayName: "smoke.txt",
	})
	if err != nil {
		t.Fatalf("Upload: %v", err)
	}

	if fs.uploadCalled != 1 {
		t.Errorf("upload calls=%d, want 1", fs.uploadCalled)
	}
	if got := fs.uploadHeader.Get("x-goog-api-key"); got != "test-key" {
		t.Errorf("x-goog-api-key=%q, want test-key", got)
	}
	if ct := fs.uploadHeader.Get("Content-Type"); !strings.HasPrefix(ct, "multipart/") {
		t.Errorf("Content-Type=%q, want multipart/...", ct)
	}

	// Metadata part: application/json with displayName.
	metaBytes, ok := fs.uploadParts["application/json"]
	if !ok {
		t.Fatalf("upload missing metadata part: keys=%v", partKeys(fs.uploadParts))
	}
	var meta struct {
		File struct {
			DisplayName string `json:"displayName"`
		} `json:"file"`
	}
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		t.Fatalf("metadata JSON: %v", err)
	}
	if meta.File.DisplayName != "smoke.txt" {
		t.Errorf("metadata displayName=%q, want smoke.txt", meta.File.DisplayName)
	}

	// Data part: text/plain with our exact bytes.
	dataBytes, ok := fs.uploadParts["text/plain"]
	if !ok {
		t.Fatalf("upload missing data part: keys=%v", partKeys(fs.uploadParts))
	}
	if !bytes.Equal(dataBytes, payload) {
		t.Errorf("data part round-trip mismatch: got %q want %q", dataBytes, payload)
	}

	// Response decoded into FileRef.
	if ref.Name != "files/abc123" {
		t.Errorf("ref.Name=%q, want files/abc123", ref.Name)
	}
	if ref.State != files.StateActive {
		t.Errorf("ref.State=%v, want ACTIVE", ref.State)
	}
	if ref.URI != "https://fake/v1beta/files/abc123" {
		t.Errorf("ref.URI=%q", ref.URI)
	}
	if ref.SizeBytes != 42 {
		t.Errorf("ref.SizeBytes=%d, want 42 (string-int parse)", ref.SizeBytes)
	}
	wantCreate, _ := time.Parse(time.RFC3339, "2026-05-12T06:00:00Z")
	if !ref.CreateTime.Equal(wantCreate) {
		t.Errorf("ref.CreateTime=%v, want %v", ref.CreateTime, wantCreate)
	}
}

func TestUpload_RequiresMimeType(t *testing.T) {
	fs := newFakeAPI(t)
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	_, err := c.Upload(context.Background(), bytes.NewReader([]byte("x")), "", files.UploadOptions{})
	if err == nil {
		t.Error("expected error when mimeType is empty")
	}
	if fs.uploadCalled != 0 {
		t.Errorf("server should not be called on validation failure; called=%d", fs.uploadCalled)
	}
}

func TestUpload_HTTPErrorSurfacesBody(t *testing.T) {
	fs := newFakeAPI(t)
	fs.uploadRespFn = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"invalid mime"}}`)
	}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	_, err := c.Upload(context.Background(), bytes.NewReader([]byte("x")), "video/mp4", files.UploadOptions{})
	if err == nil {
		t.Fatal("expected HTTP-error from Upload")
	}
	if !strings.Contains(err.Error(), "status=400") {
		t.Errorf("error %q should mention status=400", err.Error())
	}
	if !strings.Contains(err.Error(), "invalid mime") {
		t.Errorf("error %q should include response body", err.Error())
	}
}

func TestGet_RoundTrip(t *testing.T) {
	fs := newFakeAPI(t)
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	ref, err := c.Get(context.Background(), "files/xyz")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ref.State != files.StateActive {
		t.Errorf("ref.State=%v, want ACTIVE", ref.State)
	}
	if ref.Name != "files/xyz" {
		t.Errorf("ref.Name=%q, want files/xyz", ref.Name)
	}
	if !strings.HasSuffix(ref.URI, "files/xyz") {
		t.Errorf("ref.URI=%q", ref.URI)
	}
	if got := fs.getCalledFor; len(got) != 1 || got[0] != "files/xyz" {
		t.Errorf("getCalledFor=%v, want [files/xyz]", got)
	}
}

func TestGet_RequiresName(t *testing.T) {
	srv := httptest.NewServer(newFakeAPI(t).handler())
	defer srv.Close()
	c := newClient(t, srv)
	if _, err := c.Get(context.Background(), ""); err == nil {
		t.Error("expected error when name is empty")
	}
}

func TestDelete_RoundTrip(t *testing.T) {
	fs := newFakeAPI(t)
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	if err := c.Delete(context.Background(), "files/xyz"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if got := fs.deleteCalledFor; len(got) != 1 || got[0] != "files/xyz" {
		t.Errorf("deleteCalledFor=%v, want [files/xyz]", got)
	}
}

func TestDelete_HTTPErrorSurfacesBody(t *testing.T) {
	fs := newFakeAPI(t)
	fs.deleteRespFn = func(_ string) (int, string) {
		return http.StatusNotFound, `{"error":{"message":"not found"}}`
	}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	err := c.Delete(context.Background(), "files/missing")
	if err == nil {
		t.Fatal("expected HTTP-error from Delete")
	}
	if !strings.Contains(err.Error(), "status=404") {
		t.Errorf("error %q should mention status=404", err.Error())
	}
}

// TestWait_PollsUntilActive verifies the poll loop: server returns
// PROCESSING for the first 2 polls, then ACTIVE on the third. Wait
// must return the ACTIVE ref without further polling.
func TestWait_PollsUntilActive(t *testing.T) {
	fs := newFakeAPI(t)
	var calls int
	fs.getRespFn = func(name string) (int, string) {
		calls++
		state := "PROCESSING"
		if calls >= 3 {
			state = "ACTIVE"
		}
		return http.StatusOK, fmt.Sprintf(`{"name":%q,"state":%q,"mimeType":"video/mp4","sizeBytes":"100","createTime":"2026-05-12T06:00:00Z","expirationTime":"2026-05-14T06:00:00Z","uri":"https://fake/v1beta/%s"}`,
			name, state, name)
	}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	ref := &files.FileRef{Name: "files/poller", State: files.StateProcessing}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	got, err := c.Wait(ctx, ref, files.WaitOptions{PollInterval: 10 * time.Millisecond})
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if got.State != files.StateActive {
		t.Errorf("Wait returned state=%v, want ACTIVE", got.State)
	}
	if calls != 3 {
		t.Errorf("Get calls=%d, want 3 (2 processing + 1 active)", calls)
	}
}

// TestWait_FailedSurfaces verifies that Wait returns the FAILED ref
// AND an error (so callers can branch on both).
func TestWait_FailedSurfaces(t *testing.T) {
	fs := newFakeAPI(t)
	fs.getRespFn = func(name string) (int, string) {
		return http.StatusOK, fmt.Sprintf(`{"name":%q,"state":"FAILED","mimeType":"video/mp4","sizeBytes":"0","createTime":"2026-05-12T06:00:00Z","expirationTime":"2026-05-14T06:00:00Z"}`, name)
	}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	ref := &files.FileRef{Name: "files/bad", State: files.StateProcessing}
	got, err := c.Wait(context.Background(), ref, files.WaitOptions{PollInterval: 10 * time.Millisecond})
	if err == nil {
		t.Fatal("expected error on FAILED state; got nil")
	}
	if got == nil || got.State != files.StateFailed {
		t.Errorf("Wait returned ref=%+v, want state=FAILED", got)
	}
	if !strings.Contains(err.Error(), "failed processing") {
		t.Errorf("error %q should mention 'failed processing'", err.Error())
	}
}

// TestWait_ContextCancellation verifies the ctx-bound poll loop
// surfaces ctx.Err() promptly when the deadline expires mid-poll.
func TestWait_ContextCancellation(t *testing.T) {
	fs := newFakeAPI(t)
	fs.getRespFn = func(name string) (int, string) {
		// Forever PROCESSING
		return http.StatusOK, fmt.Sprintf(`{"name":%q,"state":"PROCESSING","mimeType":"video/mp4","sizeBytes":"100","createTime":"2026-05-12T06:00:00Z","expirationTime":"2026-05-14T06:00:00Z"}`, name)
	}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	ref := &files.FileRef{Name: "files/slow"}
	start := time.Now()
	_, err := c.Wait(ctx, ref, files.WaitOptions{PollInterval: 30 * time.Millisecond})
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected ctx-deadline error; got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error %q should wrap context.DeadlineExceeded", err.Error())
	}
	if elapsed > 500*time.Millisecond {
		t.Errorf("Wait took %v to honor 50ms ctx deadline; should unwind <500ms", elapsed)
	}
}

func TestWait_RequiresRefName(t *testing.T) {
	srv := httptest.NewServer(newFakeAPI(t).handler())
	defer srv.Close()
	c := newClient(t, srv)
	if _, err := c.Wait(context.Background(), &files.FileRef{}, files.WaitOptions{}); err == nil {
		t.Error("expected error when ref.Name is empty")
	}
	if _, err := c.Wait(context.Background(), nil, files.WaitOptions{}); err == nil {
		t.Error("expected error when ref is nil")
	}
}

// TestWait_ShortCircuitsOnActiveRef pins the no-wasted-Get contract:
// calling Wait on a ref already in StateActive must return without
// hitting the server.
func TestWait_ShortCircuitsOnActiveRef(t *testing.T) {
	fs := newFakeAPI(t)
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	ref := &files.FileRef{Name: "files/already-active", State: files.StateActive}
	got, err := c.Wait(context.Background(), ref, files.WaitOptions{})
	if err != nil {
		t.Fatalf("Wait on ACTIVE ref: %v", err)
	}
	if got != ref {
		t.Errorf("Wait should return the same ref pointer; short-circuit means no fresh fetch")
	}
	if len(fs.getCalledFor) != 0 {
		t.Errorf("Wait on ACTIVE ref burned %d Get round-trip(s); want 0", len(fs.getCalledFor))
	}
}

// TestUpload_EmptyBodyRejectedAtBoundary verifies Upload itself
// doesn't enforce non-empty body locally — the server is the
// authority on what's an acceptable payload, and rejection comes
// through as a *llm.APIError carrying the response body. The test
// fixture returns a 400 to simulate Gemini's actual behavior for
// zero-byte uploads.
func TestUpload_EmptyBodyRejectedAtBoundary(t *testing.T) {
	fs := newFakeAPI(t)
	fs.uploadRespFn = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"empty body"}}`)
	}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	_, err := c.Upload(context.Background(), bytes.NewReader(nil), "text/plain", files.UploadOptions{})
	if err == nil {
		t.Fatal("expected error on empty-body upload (server-side rejection)")
	}
	if !strings.Contains(err.Error(), "empty body") {
		t.Errorf("error %q should include server-side rejection body", err.Error())
	}
}

// TestUpload_ContextCancellationMidFlight verifies that cancelling
// ctx during an in-flight Upload unwinds promptly, not after the
// server's full response timeline. Uses a slow server (2s sleep) and
// a tight ctx (50ms) to assert the client returns within ~500ms of
// cancel rather than waiting for the server to finish.
func TestUpload_ContextCancellationMidFlight(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Slow server: hold the response for 2s. If the client doesn't
		// honor ctx cancellation, Upload would block here.
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, defaultUploadResponse)
	}))
	defer srv.Close()
	c := newClient(t, srv)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	_, err := c.Upload(ctx, bytes.NewReader([]byte("payload")), "text/plain", files.UploadOptions{})
	elapsed := time.Since(started)

	if err == nil {
		t.Fatal("expected ctx-cancel error on Upload; got nil")
	}
	if elapsed > 1*time.Second {
		t.Errorf("Upload took %v to honor ctx cancel; want < 1s (cancel should unblock the http.Client without waiting for server)", elapsed)
	}
}

// TestUpload_HTTPErrorIsAPIError verifies error wrapping: callers can
// errors.As to *llm.APIError and branch on Status / Inner sentinels.
func TestUpload_HTTPErrorIsAPIError(t *testing.T) {
	fs := newFakeAPI(t)
	fs.uploadRespFn = func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"bad api key"}}`)
	}
	srv := httptest.NewServer(fs.handler())
	defer srv.Close()
	c := newClient(t, srv)

	_, err := c.Upload(context.Background(), bytes.NewReader([]byte("x")), "text/plain", files.UploadOptions{})
	if err == nil {
		t.Fatal("expected error")
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error %T does not unwrap to *llm.APIError", err)
	}
	if apiErr.Status != http.StatusUnauthorized {
		t.Errorf("APIError.Status=%d, want 401", apiErr.Status)
	}
	if apiErr.Provider != "gemini-files" {
		t.Errorf("APIError.Provider=%q, want gemini-files", apiErr.Provider)
	}
	if !errors.Is(err, llm.ErrAuth) {
		t.Errorf("error should wrap llm.ErrAuth for 401; got %v", err)
	}
}

func partKeys(parts map[string][]byte) []string {
	out := make([]string, 0, len(parts))
	for k := range parts {
		out = append(out, k)
	}
	return out
}
