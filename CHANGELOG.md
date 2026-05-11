# Changelog

All notable changes to **pi-llm-go** will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `internal/sse` parser shared by both providers.
- Examples: `examples/streaming`, `examples/tool_calling`.
- Model-id convenience constants in each provider package
  (Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5; GPT-5.5 / 5.4 / 5.4-mini /
  5.4-nano / 4.1).
- Test coverage: SSE parser, Accumulate, error mapping, both providers'
  text + tool-call paths against httptest fakes, context cancellation,
  HTTP error propagation.

[Unreleased]: https://github.com/amittimalsina/pi-llm-go/compare/v0.0.0...HEAD
