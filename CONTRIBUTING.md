# Contributing to pi-llm-go

Thanks for considering a contribution. This package is maintained by a single person; the bar is "things one maintainer can sustain forever." Contributions are welcome with that constraint in mind.

## Quick orientation

- The `LLM` interface, message types, stream events, and helpers live at the module root.
- Providers (Anthropic, OpenAI Chat Completions, OpenAI Responses) live under `providers/`.
- The shared SSE parser is in `internal/sse/` (not part of the public surface).
- Examples are runnable `package main` programs under `examples/`.

For deeper context on the design decisions, see the per-repo `CLAUDE.md`.

## Before opening an issue

- **Bug report**: minimal reproduction (env, code, observed vs. expected) helps a lot. If a real provider response is involved, redact secrets and include the raw HTTP body.
- **Feature request**: describe the use case first, then the proposed shape. We resist additions that don't have a concrete consumer.

## Before opening a PR

Required:
- Tests for new behavior. Provider tests use `httptest.NewServer` with canned SSE responses (see `providers/*/openai*_test.go`). Live API tests are gated on env vars and not run in CI.
- `CHANGELOG.md` entry under `[Unreleased]`, classified `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`.
- `go test -race ./...` green.
- `go vet ./...` green.
- For new public API: a paragraph in the PR description explaining the use case and any rejected alternatives.

Style:
- `gofmt` enforced via CI. No bikeshedding on naming in review unless misleading.
- Comments only when the WHY is non-obvious; let well-named identifiers do the WHAT.

## Adding a new provider

1. New package under `providers/<name>/`.
2. Implement `llm.LLM.Stream` via raw `net/http` (we deliberately avoid vendor SDKs to control streaming behavior and dep surface).
3. Use `internal/sse.Read` for the response stream.
4. Map provider events to existing `llm.StreamEvent` types where possible; only widen the public events if your provider exposes a meaningfully different concept.
5. Tests against `httptest.NewServer` covering: text streaming, tool-call streaming, request-body shape, HTTP error wrapping. **No live API in CI.**
6. End-to-end smoke test against the real provider before submission (documented in the PR description).
7. Update `examples/` if your provider needs a flag in an existing example or merits its own example.

## Review cadence

- Issues acknowledged within 7 days.
- PRs reviewed within 14 days for an initial response. Larger PRs may take longer for full review.
- Provider PRs get fast-tracked once the test suite is green and end-to-end verification is shown.

## Releasing (maintainer)

1. Confirm `CHANGELOG.md` `[Unreleased]` is accurate.
2. Rename to `[X.Y.Z] - YYYY-MM-DD`, start a fresh `[Unreleased]` block, commit.
3. `git tag -a vX.Y.Z -m "vX.Y.Z"` then `git push origin vX.Y.Z`.
4. `gh release create vX.Y.Z --title "..." --notes "..."` (pull the notes from the matching CHANGELOG section).

## License

By contributing, you agree your contributions will be licensed under the project's [MIT License](LICENSE).
