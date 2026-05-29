# Changelog

All notable changes to **pi-llm-go** will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1] - 2026-05-29

Adds slog telemetry on the retry loop so long-running consumers can
see backoff windows that were previously invisible. No new public API
surface; slog field names are locked as part of the v1 contract.
Closes #29 (surfaced from noumenal-ai/noumenal_agent#105 — silent
30-second-to-30-minute retries during DSA runs).

### Added

- **Structured slog telemetry on `RunWithRetry`** (closes #29). Each
  retriable failure that gets retried emits a `slog.DebugContext`
  record with stable structured fields; budget exhaustion emits a
  separate exhaustion record. No new public API, no new dependencies
  — `log/slog` is stdlib. Default slog level is INFO so consumers see
  nothing unless they configure DEBUG handling. Otel consumers can
  route via the `otelslog` bridge.
  - Records: `"llm.retry.attempt"` (per retried failure),
    `"llm.retry.exhausted"` (once on budget exhaustion).
  - Attempt fields: `attempt` (1-indexed), `max_attempts`, `delay_ms`,
    `cause` ∈ {`rate_limit` / `overloaded` / `server_error` /
    `net_error` / `unknown`}, `retry_after_ms` (when server-hinted),
    `error` (truncated to 256 chars). The `error` key follows
    structured-log shipper convention (Datadog / ECS / Loki all
    auto-extract on that name) rather than the Go-variable
    convention `err`.
  - Exhaustion fields: `max_attempts`, `last_cause`, `error`.
  - No record on first-attempt success or non-retriable errors —
    those paths don't need narration.
  - **Field names are part of the v1 public contract.** Consumers
    writing custom slog handlers (prometheus counter, dashboard
    enrichment) can rely on the field set not churning post-v1.
    Surfaced from noumenal-ai/noumenal_agent#105 (silent retries
    during long DSA runs).

## [0.11.0] - 2026-05-20

Strict tool use + ToolChoice across all four providers. Closes #26.
Two related additions that unblock guaranteed-schema tool-call
sampling and forced tool dispatch — load-bearing for tool-heavy
agents (e.g. Actioning Agent rule generation in noumenal_product).
Live-smoked against Anthropic + Azure OpenAI (Chat Completions AND
Responses API surfaces) to verify the divergent wire shapes.

### Added

- **Strict tool use** (`Tool.Strict bool`) — opts a tool into
  grammar-constrained sampling. The model's token sampler is
  constrained to schema-valid tokens, so the emitted input is
  guaranteed to match `InputSchema` (no app-side validate-and-retry
  loop). Closes #26.
  - Anthropic Messages: serialized as top-level `strict` on the tool
    definition (peer of name/description/input_schema). Mirrored on
    `count_tokens` so input-token counts stay accurate.
  - OpenAI Chat Completions: serialized as `function.strict` (nested
    under the function object).
  - OpenAI Responses API: top-level `strict` on the function tool
    (flatter Responses wire shape).
  - Gemini: silently ignored (no per-tool strict equivalent; Gemini
    uses `response_schema` for a different surface).
  - `omitempty` on the wire field — non-strict tools emit no `strict`
    key, keeping the body lean and cross-host-compatible.
- **`Request.ToolChoice`** — controls whether/which tool the model
  must call. Four modes via the neutral `llm.ToolChoiceType` enum:
  `Auto` (model decides), `Any` (must call some tool), `Tool` (must
  call this named tool, via `ToolChoice.Name`), `None` (disable tools).
  - Anthropic: forwarded as `tool_choice: {"type": <type>, "name": <name>}`.
    Keywords match pi-llm-go's enum 1:1.
  - OpenAI Chat Completions: keyword remap — `Any` → `"required"`.
    `Tool` becomes `{"type":"function","function":{"name":"..."}}`.
  - OpenAI Responses API: same keywords; `Tool` uses the flatter
    `{"type":"function","name":"..."}` shape (no nested `function`).
  - Gemini: forwarded as `toolConfig.functionCallingConfig` with
    `mode` mapping `Auto`→AUTO, `Any`→ANY, `None`→NONE; `Tool` becomes
    `mode=ANY` + `allowedFunctionNames=[Name]` (Gemini has no dedicated
    force-this-exact-tool mode).
  - Build-time validation: `Type=Tool` without `Name` returns an error
    BEFORE the request leaves the client (provider would 400 otherwise).
  - **Anthropic + extended thinking caveat**: `Any` and `Tool` are
    incompatible with `ThinkingConfig` on Anthropic (only `Auto`/`None`
    work). Surfaces as a 400 at request time. Documented on
    `ToolChoice` godoc.
- `examples/strict_tool_use/` — cookbook demonstrating both features:
  enum-constrained input + forced named tool call against Anthropic.

## [0.10.2] - 2026-05-14

Republishes v0.10.1's thinking-block fix with internal product
identifiers scrubbed from comments + CHANGELOG. v0.10.1 is retracted
via `retract` in go.mod. No code-behavior change between v0.10.1
and v0.10.2.

### Changed

- **Scrubbed internal product identifiers from public docs**. The
  bug surface + repro context (multi-iteration Opus 4.7 agent runs
  with adaptive thinking) remains documented; only proprietary
  identifiers drop.

### Retracted

- v0.10.1 — shipped with internal product identifiers in CHANGELOG +
  test-file comments that should not appear in this OSS repo. Use
  v0.10.2 instead.

## [0.10.1] - 2026-05-14 [RETRACTED]

Retracted via `retract v0.10.1` in go.mod — see v0.10.2.

Hotfix for v0.10.0's adaptive-thinking rollout: empty-content
ThinkingBlocks were elided on round-trip, breaking multi-iteration
agent runs.

### Fixed

- **Anthropic round-trip of empty-thinking blocks**. v0.10.0's
  `apiBlock.Thinking` had `omitempty`, which elided the field on
  ThinkingBlocks where the model emitted a signed continuation token
  but minimal/empty thinking summary text. Anthropic's content-block
  validator requires the field on type=thinking and returned HTTP 400
  with path `messages.N.content.M.thinking.thinking: Field required`,
  breaking any multi-iteration agent run with thinking enabled
  (reported 2026-05-14 on a multi-iteration Opus 4.7 agent run with
  adaptive thinking).
  - Fix: added `apiBlock.MarshalJSON` that special-cases the
    "thinking" type to force the `thinking` field through (even when
    empty) while preserving omitempty behavior for every other block
    type — text / tool_use / tool_result / image regress-tested.

## [0.10.0] - 2026-05-13

Anthropic Opus 4.7 adaptive thinking support. Closes issue #20:
v0.9.0's `thinking.type=enabled` shape returns 400 against Opus 4.7+
models, blocking any Noumenal use of extended thinking on the new
flagship. New `Effort` field on `ThinkingConfig` plus a provider-side
dispatch routes to the right wire shape per model family.

### Added

- **`llm.Effort` type + `ThinkingConfig.Effort` field** for Anthropic
  adaptive extended thinking (Opus 4.6+, **required** on Opus 4.7+).
  Closes issue #20.
  - Constants: `llm.EffortLow`, `llm.EffortMedium`, `llm.EffortHigh`.
  - Anthropic provider dispatches on the field that's set:
    - `Effort` set → adaptive shape: `{"thinking":{"type":"adaptive"},
      "output_config":{"effort":"<level>"}}`.
    - `BudgetTokens > 0` (Effort empty) → legacy manual shape:
      `{"thinking":{"type":"enabled","budget_tokens":N}}`.
    - Both set → Effort wins (adaptive is the future, manual is
      deprecated). Lets callers pre-set both during a migration.
  - The `output_config` field is at the request TOP LEVEL (not nested
    under thinking) per Anthropic's wire contract.
  - `apiThinkingConfig.budget_tokens` now has `omitempty` so the
    adaptive shape doesn't leak a `budget_tokens: 0` that Opus 4.7
    rejects.
  - Per-provider behavior documented on `ThinkingConfig` godoc:
    Anthropic dispatches, Gemini honors BudgetTokens only (no Effort
    mapping yet), OpenAI Chat ignores entirely, OpenAI Responses
    uses its own ReasoningEffort.

### Fixed

- **Opus 4.7 returned 400** for any request using `ThinkingConfig`
  (legacy `thinking.type=enabled` shape rejected by all 4.7+ models).
  Live-smoke-confirmed: `BudgetTokens=2048` → `"thinking.type.enabled"
  is not supported for this model. Use \"thinking.type.adaptive\" and
  \"output_config.effort\"...`. The new `Effort` path is now the
  correct surface for Opus 4.7. Originally reported by an internal
  consumer 2026-05-13.

## [0.9.0] - 2026-05-13

Retry middleware + finer-grained 4xx sentinels. v0.6.0's structured
error categories and `RetryAfter` surfacing are now backed by a
first-party retry loop; v0.6.0 callers needing a one-liner default
policy can drop the boilerplate.

### Added

- **`llm.RetryPolicy` + `Options.Retry` on every provider.** Retries
  retriable errors (429 / 529 / 5xx / transient network) with
  exponential backoff + full jitter. Server-supplied `Retry-After`
  hints dominate the exponential schedule, capped at `MaxDelay`.
  `ErrAuth` / `ErrInvalidRequest` / `context.Canceled` are NOT
  retried.
  - Scope: only the initial HTTP attempt is retriable. Mid-stream
    connection breaks terminate the iterator — resuming would replay
    events the consumer already saw.
  - `llm.DefaultRetryPolicy()` returns 4 attempts / 1s base / 30s cap.
  - `llm.RunWithRetry[T]` is exported so callers can build
    cross-provider fallback / circuit-breaker logic on top.
- **`llm.ErrContextLength` and `llm.ErrPolicyViolation` sentinels**
  wrap `ErrInvalidRequest` (4xx subtree). Detected by inspecting the
  response body for known message patterns across Anthropic / OpenAI /
  Gemini schemas — provider error envelopes don't carry canonical
  machine-readable categories for these cases.
- **`llm.ClassifyInvalidRequest(body)`** returns the most-specific
  4xx sentinel for an error body.
- **`llm.SentinelFor(status, body)`** combines status-code mapping with
  body-pattern inspection — providers use this when constructing
  `APIError` so the finer sentinels reach callers automatically.
- **`llm.IsContextLength` / `llm.IsPolicyViolation`** sugar helpers
  for `errors.Is`.

### Changed

- Every provider's APIError construction now calls `SentinelFor` instead
  of `SentinelForStatus`, so consumers using `errors.Is(err, ErrContextLength)`
  or `errors.Is(err, ErrPolicyViolation)` match without changes elsewhere.
- Backward compatibility: `errors.Is(err, ErrInvalidRequest)` continues to
  match the new child sentinels (they wrap it via `%w`).

## [0.8.1] - 2026-05-13

Bug fix for Gemini `CountTokens`: the v0.8.0 implementation posted
`systemInstruction` and `tools` at the top level of the count_tokens
body, but Gemini's `:countTokens` endpoint rejects that shape with
`"Unknown name 'systemInstruction': Cannot find field."`. The body
must be wrapped in `{"generateContentRequest": {model, contents, ...}}`
so the system prompt and tool definitions contribute to the count.

### Fixed

- **`gemini.Provider.CountTokens` now sends the wrapped body shape.**
  Without the wrapper, any v0.8.0 caller passing a non-empty `System`
  hit a 400 from Gemini. With the wrapper, the inner `model` field is
  set to the fully-qualified resource name (`models/<id>`) per the
  endpoint contract. Discovered via live-API smoke against v1beta;
  the httptest fakes accepted the bad shape without complaint.
- Test asserts the wrapped shape + the inner `model` field on
  every count_tokens body to guard against regression.

## [0.8.0] - 2026-05-13

Two roadmap items shipped together: pre-flight token counting (closes
Mario's "no count helper" gap from the WWMD audit) and first-party
cost projection from a `Usage` record. The new helpers consume the
v0.7.0 TTL breakdown directly, so silent 5min fallback flows into
the cost number with zero caller-side branching.

### Added

- **`TokenCounter` interface** for pre-flight token counting without
  spending an inference call. Implemented by the Anthropic and Gemini
  providers against their dedicated `/v1/messages/count_tokens` and
  `:countTokens` endpoints respectively. Use via type assertion:
  ```go
  if c, ok := provider.(llm.TokenCounter); ok {
      n, err := c.CountTokens(ctx, req)
  }
  ```
  The OpenAI Chat Completions and Responses providers do NOT implement
  `TokenCounter` — OpenAI's tokenization is local-only via tiktoken,
  which pi-llm-go does not bundle.
- **`llm.Cost` + `llm.Pricing` + `ComputeCost` / `ApplyPricing` /
  `RegisterPricing`** for dollar-cost projection from a `Usage` record.
  Ships a hand-curated seed table covering the Claude 4, GPT-5, and
  Gemini 2.5/3.1 families (verified 2026-05-13 against provider docs);
  callers register custom `Pricing` for any model not in the seed (older
  snapshots, regional or batch tiers).
- The TTL breakdown on `Usage` (v0.7.0) flows through `ApplyPricing`:
  when `CacheWrite5mTokens` and `CacheWrite1hTokens` are populated, they
  price against their respective tier rates independently, so silent
  5min fallback (Issue #12 diagnostic) is reflected in cost projections
  without any caller-side branching.

## [0.7.0] - 2026-05-13

Per-TTL cache-write breakdown on `Usage`, providing the structured
signal needed to detect silent 5min fallback when `CacheRetention=long`
is requested but the model downgrades the hold. Closes Noumenal issue
#12 — multi-iteration cost budgeting can now adjust on observed TTL
rather than assuming the requested tier landed. Fully backward
compatible.

### Added

- **Per-TTL cache-write breakdown** on `Usage` (closes Noumenal
  issue #12). Two new Anthropic-specific fields:
  - `Usage.CacheWrite5mTokens` — tokens cached at the default ~5
    minute TTL.
  - `Usage.CacheWrite1hTokens` — tokens cached at the extended 1
    hour TTL (the `extended-cache-ttl-2025-04-11` beta tier).
  - Decoded from Anthropic's `cache_creation.ephemeral_5m_input_tokens`
    / `ephemeral_1h_input_tokens` response fields. Other providers
    leave both at 0.
  - The diagnostic signal: when `CacheRetention=long` was requested
    AND `CacheWrite5mTokens > 0` AND `CacheWrite1hTokens == 0` on the
    response, the model silently fell back to 5min — cost projections
    that assumed a 1h-cached prefix need to adjust. Noumenal Decision 20's
    multi-iteration cost budget depends on this signal.
  - `Usage.CacheWriteTokens` semantics unchanged: still the total
    across all TTL tiers. Backward compatible.

### Changed

- `cache.go` godoc on `CacheRetention` now documents the 1h-TTL
  model availability (all currently-shipped Claude 4 family models
  support it; older Claude 3.x may silently fall back), Anthropic's
  March 2026 regression of the default ephemeral TTL from 60 min to
  5 min, and the structured-signal diagnostic for detecting silent
  5min fallback via the Usage breakdown fields.

## [0.6.0] - 2026-05-13

Structured error categories + `retry-after` surfacing. Closes Noumenal
issue #11 — consumers can now implement category-distinct retry /
escalation policies (429 / 529 / 5xx) without parsing error strings.
Fully backward compatible.

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

### Changed

- `APIError.Error()` now appends ` retry_after=<duration>` to its
  rendered string when `RetryAfter > 0`. Pre-1.0 cosmetic change;
  callers that string-match the error format will need to widen
  their pattern. The wrapped sentinel + Status + Body remain in
  the same positions for stable parsing.

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

[Unreleased]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.11.1...HEAD
[0.11.1]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.10.0...v0.10.2
[0.10.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/amit-timalsina/pi-llm-go/releases/tag/v0.1.0
