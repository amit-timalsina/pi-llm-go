# pi-llm-go roadmap

This is the maintainer's working plan. Items aren't promises — they're
ranked by user-value-per-LOC and informed by WWMD audits against Mario
Zechner's [pi-ai](https://github.com/badlogic/pi-mono/tree/main/packages/ai).
Reordering happens when reality changes.

## Status

- **v1.0.0** shipped 2026-06-06 — First stable release. The public API
  has been additive-only since v0.2.0 (no breaking churn) and dogfooded
  in production at Noumenal (DSA + AA) for ~4 weeks. From here the
  package follows semver strictly: no breaking change without a
  major-version (module-path) bump. No new code vs v0.11.2 — this tag
  is the stability commitment plus a documentation feature-coverage
  pass (strict tool use, `ToolChoice`, adaptive `Effort`, full examples
  list). The self-imposed "≥1 external Go reviewer" gate is **waived**:
  internal production dogfooding across two agents is treated as the
  adoption signal.
- **v0.11.2** shipped 2026-05-30 — Pricing seed entries for
  `claude-opus-4-6` + `claude-opus-4-5` + `claude-sonnet-4-5` +
  `gemini-robotics-er-1.6-preview`. Closes #32 — noumenal AA + DSA
  VLM previously hit `ErrUnknownModel`. Added `ClaudeOpus4_6`
  provider constant. Surfaced from noumenal-ai/noumenal_agent#108.
- **v0.11.1** shipped 2026-05-29 — Structured slog telemetry on
  `RunWithRetry`. Closes #29 — silent backoff windows during long
  agent runs are now observable. No new public API; slog field
  names locked as v1 contract. Surfaced from
  noumenal-ai/noumenal_agent#105 (consumer-driven).
- **v0.11.0** shipped 2026-05-20 — Strict tool use (`Tool.Strict`) +
  `Request.ToolChoice` across all four providers. Closes #26 — the
  Noumenal Actioning Agent's rule-generator needed enum-constrained
  controllable-name fields with parenthetical units that
  non-strict-mode models strip ~100% of the time. Live-smoked
  against Anthropic + Azure OpenAI (Chat + Responses).
- **v0.10.2** shipped 2026-05-14 — Republishes v0.10.1's thinking-
  block fix with internal product identifiers scrubbed from comments
  + CHANGELOG. v0.10.1 is retracted (Go proxy had cached it before
  the scrub PR landed). No code-behavior change between v0.10.1 and
  v0.10.2.
- **v0.10.1** shipped 2026-05-14 [RETRACTED] — Hotfix for v0.10.0's adaptive-
  thinking rollout: empty-content ThinkingBlocks were elided on
  round-trip via `apiBlock.Thinking`'s `omitempty` tag, which broke
  multi-iteration agent runs (Anthropic rejected with HTTP 400
  `messages.N.content.M.thinking.thinking: Field required`).
  Reported 2026-05-14 on a multi-iteration Opus 4.7 agent run with
  adaptive thinking. `apiBlock.MarshalJSON` now special-cases the
  thinking type to force the field through.
- **v0.10.0** shipped 2026-05-13 — Anthropic adaptive thinking
  support for Opus 4.7+. New `llm.Effort` enum + `ThinkingConfig.Effort`
  field; provider routes adaptive vs manual wire shapes per the
  caller-set field. Closes issue #20 (Noumenal team hit v0.9.0's
  legacy shape returning 400 on Opus 4.7). Live-smoke verified.
- **v0.9.0** shipped 2026-05-13 — Retry middleware (`llm.RetryPolicy`,
  `Options.Retry` on every provider, `llm.RunWithRetry[T]` exported)
  + finer 4xx sentinels (`ErrContextLength`, `ErrPolicyViolation`)
  detected by body-pattern matching. v0.6.0's categorical errors +
  `RetryAfter` surfacing are now backed by a first-party retry loop.
- **v0.8.1** shipped 2026-05-13 — Fix Gemini `CountTokens` wire shape
  (`generateContentRequest` wrapper required for system + tools to
  contribute to count). Live-smoke regression catch.
- **v0.8.0** shipped 2026-05-13 — `TokenCounter` interface (Anthropic +
  Gemini, against their dedicated count endpoints) + first-party
  `Cost` / `Pricing` / `ComputeCost` helpers with a hand-curated seed
  table for Claude 4 / GPT-5 / Gemini 2.5/3.1. The v0.7.0 TTL
  breakdown flows through `ApplyPricing` automatically.
- **v0.7.0** shipped 2026-05-13 — Per-TTL cache-write breakdown on
  `Usage` (`CacheWrite5mTokens` / `CacheWrite1hTokens`) decoded from
  Anthropic's `cache_creation.ephemeral_*_input_tokens` response
  fields. Closes Noumenal issue #12 — multi-iteration cost budgeting
  can now detect silent 5min fallback when `CacheRetention=long` was
  requested but the model downgraded the hold.
- **v0.6.0** shipped 2026-05-13 — Structured error categories
  (`ErrServerError`, `ErrOverloaded`) split out of `ErrProvider` plus
  `APIError.RetryAfter` populated by all four providers. Closes
  Noumenal issue #11.
- **v0.5.0** shipped 2026-05-12 — Gemini Files API helper
  (`providers/gemini/files`): Upload / Wait / Get / Delete with
  multipart upload, ACTIVE-state polling (short-circuits on
  already-ACTIVE refs), and ~2 GB file ceiling. Closes the >20 MB
  video gap left by v0.4.0.
- **v0.4.0** shipped 2026-05-12 — Gemini provider + `llm.VideoBlock` for
  native multimodal video input (Gemini-exclusive; Anthropic and OpenAI
  providers reject at the wire boundary).
- **v0.3.0** shipped 2026-05-11 — `ImageBlock` multimodal image input
  across the three pre-Gemini providers.
- **v0.2.0** shipped 2026-05-11 — CacheRetention knob (WWMD convergence).
- **v1.0 — shipped 2026-06-06.** Next: the mid-term additive surface
  below, each released as a v1.x minor.

## Near-term (next 1–3 minor releases)

### Open near-term slots

(v0.9.0 closed the retry + finer-errors slot. Next near-term candidates
are driven by consumer asks. Highest signal: a real Go consumer
review of the public API before tagging v1.0.)

## Mid-term

- **Batch API** (Anthropic + OpenAI both ship async batch — ~50% off
  list). New `Batch` interface alongside `LLM`. Different lifecycle so
  it stays its own surface.
- **Citations** (Anthropic): pass through `citations` arrays on text
  blocks when the provider returns them.
- **Web search / built-in tools**: surface Anthropic's
  `web_search_20250305` and OpenAI Responses' built-in tools as a new
  block type.
- **PDF input**: Anthropic accepts PDFs as document blocks; ours could
  too. Lower priority than images.
- **More provider compat**: AWS Bedrock direct, Mistral direct — both
  currently reachable via OpenAI-compat with caveats.

## Observability

Observability is **first-class but external**: the `StreamEvent`
iterator is already the per-token telemetry stream, and `Usage` carries
token + cache stats per call. The framework adds no observability deps.

Planned (no version pin yet):

- `examples/observability/` — wires OpenTelemetry spans and `slog`
  structured logs by ranging over `Stream()` events and tagging
  attributes from `Usage`. Zero new framework deps; consumer
  copy-and-tweak. Pairs with the agent-side example in pi-agent-go.
- OTel HTTP propagation works out-of-the-box today: pass an OTel-
  instrumented `*http.Client` via the provider's `Options.HTTPClient`
  and `traceparent` headers flow to the provider unchanged.

A first-party `pi-llm-go/otel` sub-package is **deferred** until the
example pattern proves insufficient for real consumers. Same for any
baked-in `Observer` interface — events already are the observer surface.

## Out of scope (intentionally)

- **Computer use** (Anthropic). Too narrow; consumers wire it via tool
  blocks if they need it.
- **Built-in agent loop.** That's pi-agent-go's job. pi-llm-go stays
  one-shot.
- **Embeddings / fine-tuning / file management endpoints.** Different
  shape, different package — would dilute the streaming-completions
  focus. Could spin out a sibling later.
- **Model registry / dynamic capability detection.** Caller knows their
  model. Typed constants per provider are the limit.

## v1.0 readiness checklist — closed 2026-06-06

- [x] Multimodal input shipped (image v0.3.0, video v0.4.0, Files API v0.5.0).
- [x] Four providers + Azure compat verified in production.
- [x] Public API surface additive-only (no breaking churn) since v0.2.0.
- [~] External Go reviewer — **waived**; internal production dogfooding
      across DSA + AA is the adoption signal.
- [x] `pkg.go.dev` `Example_*` tests (`example_test.go`).
- [x] CONTRIBUTING.md present (provider-addition checklist in CLAUDE.md).
- [ ] `examples/observability/` — deferred post-1.0 (OTel/slog pattern
      already works via `Options.HTTPClient` + the `StreamEvent` iterator;
      see Observability section).

## Convergence work — closed

WWMD audits of `cache_control` and PromptBuilder drove the v0.2.0
rewrite. No open WWMD divergence as of 2026-05-11.
