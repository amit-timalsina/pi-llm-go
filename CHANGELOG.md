# Changelog

All notable changes to **pi-llm-go** will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/amit-timalsina/pi-llm-go/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/amit-timalsina/pi-llm-go/releases/tag/v0.1.0
