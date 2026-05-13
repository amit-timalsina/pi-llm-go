# Changelog

All notable changes to **pi-llm-go** will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Structured error categories** for retry / escalation policy
  (closes #11). Two new sentinels split out of the previous
  catch-all `ErrProvider`:
  - `ErrServerError` — generic 5xx (excluding 529). Consumer
    policy: retry with backoff; escalate if sustained.
  - `ErrOverloaded` — Anthropic-style 529 "overloaded." Consumer
    policy: short backoff (~60s); consider provider fallback if
    sustained.
  - Both wrap `ErrProvider` via `fmt.Errorf("%w...")`, so
    existing `errors.Is(err, llm.ErrProvider)` callers keep
    matching 5xx + 529 responses unchanged. Fully backward
    compatible.
- `APIError.RetryAfter time.Duration` — populated by all four
  built-in providers (Anthropic, OpenAI Chat, OpenAI Responses,
  Gemini + Gemini Files) when the response carries a `Retry-After`
  or `retry-after-ms` header. Surfaced in `APIError.Error()` for
  debuggability.
- `llm.ParseRetryAfter(http.Header) time.Duration` — helper
  exposing the same parser. Supports RFC 7231 delta-seconds,
  RFC 7231 HTTP-date (past dates clamp to 0), and OpenAI's
  `retry-after-ms` precision form.
- Sugar helpers `llm.IsRateLimited(err)`, `llm.IsOverloaded(err)`,
  `llm.IsServerError(err)` — one-liner wrappers around
  `errors.Is` for the common consumer-side branches.
- `SentinelForStatus` updated to return `ErrOverloaded` for 529
  and `ErrServerError` for other 5xx; existing 401/403→ErrAuth,
  429→ErrRateLimit, other 4xx→ErrInvalidRequest unchanged.

## [0.5.0] - 2026-05-12

Gemini Files API helper. Closes the >20 MB video gap left by v0.4.0;
callers no longer need Google's `genai-go` SDK alongside pi-llm-go to
stage large media for `generateContent` calls.

### Added

- New `providers/gemini/files` sub-package — minimal Gemini Files API
  client. Closes the >20 MB video gap left by v0.4.0; callers no
  longer need Google's `genai-go` SDK alongside pi-llm-go to upload.
- `files.New(Options) (*Client, error)` — same Options shape as
  `gemini.New` (APIKey + BaseURL + HTTPClient).
- `Client.Upload(ctx, r, mimeType, UploadOptions) (*FileRef, error)`
  — multipart upload, supports files up to ~2 GB. Returns a
  `FileRef` with the URI to plug into `VideoBlock.URI`.
- `Client.Wait(ctx, ref, WaitOptions...) (*FileRef, error)` —
  polls `Get` until state reaches `ACTIVE` or `FAILED`. Default
  poll interval 2s; configurable via `WaitOptions.PollInterval`.
  Honors `ctx` cancellation promptly.
- `Client.Get(ctx, name) (*FileRef, error)` and `Client.Delete(ctx, name) error`
  for state inspection + cleanup.
- `FileRef` exposes `Name`, `URI`, `MimeType`, `SizeBytes`, `State`,
  `CreateTime`, `ExpirationTime` (typically ~48h), `SHA256Hash`,
  `Source`.
- `examples/multimodal_gemini` extended with `--video-upload PATH`
  flag that exercises the full upload → wait → use → delete loop.
- Live-API verified end-to-end (Upload → Wait → Get → Delete).

### Deferred

- **Resumable upload protocol** for files >2 GB. The current
  multipart path handles the typical case; resumable adds a
  separate two-step `X-Goog-Upload-Protocol: resumable` flow.

## [0.4.0] - 2026-05-12

First-party Google Gemini support + native multimodal video input via
the new `llm.VideoBlock`. Anthropic and OpenAI providers reject
`VideoBlock` at the wire boundary (video is Gemini-exclusive at
v0.4.0).

### Added

- New `providers/gemini` package — first-party Google Gemini support
  (Gemini 2.5 family + Gemini 3 family + Gemini Robotics ER 1.6). Same
  `llm.LLM` interface as the existing providers; SSE streaming via
  `:streamGenerateContent?alt=sse`. Constants for canonical model IDs.
  Auth via `x-goog-api-key` header (Google AI direct; Vertex AI is a
  future addition).
- `llm.VideoBlock` — sealed `Block` extension for multimodal video
  input. Today only the Gemini provider accepts it natively; Anthropic
  and OpenAI providers reject `VideoBlock` at the wire boundary with a
  clear error pointing to the frame-extraction workaround. Two
  emission shapes:
  - **Inline base64** via `Data` + `MimeType` for files under ~20 MB.
  - **URI reference** via `URI` for YouTube URLs (free-tier 8h/day cap)
    or pre-uploaded Files API handles.
  - Optional `StartOffset` / `EndOffset` / `FPS` for clipping + frame-
    rate override (Gemini defaults to 1 FPS).
  - `Validate()` enforces "exactly one of Data or URI", rejects a
    leading `"data:"` URI prefix, and requires `MimeType` when `Data`
    is set.
- Gemini provider features:
  - Text + image (ImageBlock) + video (VideoBlock) input.
  - Tool calling via function declarations; the loop folds RoleTool
    messages into the prior user turn as `functionResponse` parts
    since Gemini has no separate tool role.
  - Extended thinking via `generationConfig.thinkingConfig` (translated
    from `Request.Thinking`). `thoughtsTokenCount` rolls into
    `Usage.OutputTokens` so cost accounting stays accurate.
  - `systemInstruction` for system prompts (Gemini's dedicated
    top-level field, not a role-system content).
- `examples/multimodal_gemini` — text / image / video / video-URI
  demos against any Gemini model via flags. Live-API verified against
  Gemini 2.5 Flash (text + image) and Gemini 3 Flash Preview (10-min
  YouTube video → 54k input tokens, correct content identification).

### Deferred

- **Files API helper** (`providers/gemini/files.Upload/.Wait/.Delete`)
  is planned for v0.5.0. Callers who need to upload >20 MB videos today
  can use Google's official `google.golang.org/genai` SDK to upload,
  then pass the resulting `https://generativelanguage.googleapis.com/v1beta/files/...`
  URI to `VideoBlock.URI` — pi-llm-go is URI-agnostic, no special
  handling required.
- **Vertex AI backend** — different endpoint scheme + OAuth instead of
  API key. Future Backend option on `gemini.Options`.

## [0.3.0] - 2026-05-11

Multimodal image input across all three providers. WWMD-aligned with
Mario Zechner's pi-ai. Input-only; assistant image output deferred.

### Added

- `llm.ImageBlock` — sealed `Block` extension for multimodal image
  **input** (user-role messages only — assistant/tool ImageBlocks are
  rejected at the wire boundary). Shape: `{Data: <base64>, MimeType: <mime>}`.
  Base64-only at the API surface (caller fetches their own URLs first);
  `MimeType` is required. `ImageBlock.Validate()` rejects empty fields
  and a leading `"data:"` URI prefix in `Data` (common foot-gun).
  Portable across providers when the MIME is one of `image/jpeg`,
  `image/png`, `image/gif`, `image/webp`.
  Assistant image **output** is a separate, future feature; v0.3.0 is
  input-only.
- Anthropic provider: emits the standard `{type:"image", source:{type:"base64",
  media_type, data}}` wire shape. Image-only user messages get a
  synthetic "(see attached image)" text block prepended, matching the
  upstream pi-ai placeholder convention (Anthropic prefers some
  accompanying text).
- OpenAI Chat Completions provider: switches user-message `content`
  from a plain string to the array form `[{type:"text"}, {type:"image_url",
  image_url:{url:"data:<mime>;base64,..."}}]` when any `ImageBlock` is
  present. Text-only user messages stay on the legacy string shape for
  back-compat with hosts that don't accept the array form.
- OpenAI Responses API provider: emits the `{type:"input_image",
  image_url:"data:<mime>;base64,..."}` shape (image_url is a flat
  string here, unlike Chat Completions which wraps it in an object).
- `examples/multimodal` — generates a small red-square-on-white PNG at
  runtime, asks the model to describe it; flags `--image` (use your own
  file) and `--openai` (use OpenAI Chat Completions instead of Anthropic).

## [0.2.0] - 2026-05-11

First breaking change since v0.1.x. Prompt-caching API is now a single
retention knob instead of explicit per-block markers; see migration block
below. WWMD-aligned with Mario Zechner's pi-ai `cacheRetention`.

### Added

- `CacheRetention` enum on `llm.Request` for Anthropic prompt caching.
  Values: `CacheRetentionNone` (default, no markers), `CacheRetentionShort`
  (ephemeral, ~5 min), `CacheRetentionLong` (ephemeral, 1h TTL — auto-
  attaches the `extended-cache-ttl-2025-04-11` beta header). The Anthropic
  provider auto-places markers at the static prefix boundary: the system
  prompt's trailing block, the final tool in `Request.Tools`, and the last
  text block of the most recent user message. OpenAI providers silently
  ignore `CacheRetention` (their cache is automatic and opaque).

### Removed (breaking)

- `llm.CacheControl` type and `Ephemeral()` / `EphemeralLong()` helpers.
- `CacheControl *CacheControl` field on `TextBlock`, `ThinkingBlock`,
  `ToolCallBlock`, `ToolResultBlock`, and `Tool`.
- `SystemCacheControl` and `ToolsCacheControl` fields on `Request`.

  These were introduced unreleased on `main` (PR #6, never tagged) as
  explicit per-block markers. WWMD audit against Mario Zechner's upstream
  pi-ai found the explicit-marker API has been rejected twice in the
  upstream issue tracker as a footgun: it leaks Anthropic's 4-breakpoint
  limit into caller code, encourages bad placement, and proliferates
  fragile invalidation. The single retention knob is the upstream-aligned
  shape — closes #7.

  Migration:

  ```go
  // before
  req := llm.Request{
      System:             "...",
      SystemCacheControl: llm.Ephemeral(),
      ToolsCacheControl:  llm.EphemeralLong(),
  }
  // after
  req := llm.Request{
      System:         "...",
      CacheRetention: llm.CacheRetentionLong,
  }
  ```

## [0.1.1] - 2026-05-11

CI + lint cleanup. No behavioral change vs v0.1.0.

### Added

- Dependabot config for `gomod` + `github-actions` ecosystems (weekly).
- README badges: CI status, Go Reference (pkg.go.dev), Go Report Card, MIT license.

### Changed

- Pinned `golangci-lint-action` to v8 and the linter binary to v2.12.2
  (was v2.1.6, which panicked on Go 1.26 toolchain).
- Removed the unused `errStreamCanceled` sentinel from `llm.go` and the
  unused `contentFilter` field from `providers/openai/stream.go`.
- Tightened error message capitalization (Go convention: lowercase first
  letter) on three "model is required" build errors.
- Wrapped three `defer resp.Body.Close()` calls so errcheck is satisfied.
- Various godoc and gofmt -s normalizations.

## [0.1.0] - 2026-05-11

Initial public release. Real-API verified against Anthropic (streaming,
tool calling, extended thinking) and Azure OpenAI (Chat Completions
endpoint with GPT-5.4-mini, plus the Responses API with reasoning
summaries).

### Added

- Initial release skeleton: `LLM` interface, `Request`, `Message`, `Block`
  sum type (`TextBlock`, `ThinkingBlock`, `ToolCallBlock`, `ToolResultBlock`),
  `StreamEvent` sum type, `Tool`, `Usage`, `StopReason`, `APIError` + sentinels.
- `Stream` / `Complete` / `Accumulate` top-level helpers.
- Anthropic Messages provider (`providers/anthropic`): text streaming,
  tool calling, extended thinking, normalized stop reasons, redacted-
  thinking pass-through.
- OpenAI-compatible Chat Completions provider (`providers/openai`):
  text streaming, tool calling, normalized stop reasons, multi-tool-result
  message expansion at the boundary.
- OpenAI Responses API provider (`providers/openai_responses`):
  /v1/responses endpoint covering text, function tool calls, reasoning
  summaries (mapped to llm.ThinkingBlock), response lifecycle envelope.
  Required for GPT-5-family server-side state, reasoning summaries, and
  the built-in tool stack. Supports OpenAI directly and Azure OpenAI /
  Azure AI Services via URL + Headers options. Includes ReasoningEffort
  and IncludeReasoningSummary options.
- `openai.Options.URL` field: full chat-completions endpoint URL override.
  Required for Azure OpenAI, whose endpoint embeds a deployment name and
  api-version query.
- `openai.Options.Headers` map: merged into outgoing requests, user values
  win over defaults. Required for Azure's `api-key:` auth header instead
  of `Authorization: Bearer`.
- `internal/sse` parser shared by both providers.
- Examples (all verified end-to-end against real APIs):
  - `examples/streaming` — basic streaming completion.
  - `examples/tool_calling` — hand-rolled tool-call loop.
  - `examples/multi_turn` — manual transcript management across turns.
  - `examples/thinking` — extended thinking with ANSI-styled output.
  - `examples/azure_openai` — Azure OpenAI via data-plane key or AAD token.
  - `examples/openai_responses` — Responses API with optional reasoning
    summary streaming, against OpenAI or Azure.
- Model-id convenience constants in each provider package
  (Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5; GPT-5.5 / 5.4 / 5.4-mini /
  5.4-nano / 4.1).
- Test coverage: SSE parser, Accumulate, error mapping, both providers'
  text + tool-call paths against httptest fakes, context cancellation,
  HTTP error propagation.
- `ThinkingConfig` godoc documents the Anthropic constraint
  `MaxTokens > BudgetTokens`.

### Fixed

- OpenAI wire format now uses `max_completion_tokens` instead of the
  deprecated `max_tokens`. GPT-5, o1, and other modern reasoning models
  reject `max_tokens` outright; the new field is accepted by all current
  OpenAI-compatible hosts. Caught via Azure OpenAI smoke-testing against
  gpt-5.4-mini.

[Unreleased]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/amit-timalsina/pi-llm-go/releases/tag/v0.1.0
