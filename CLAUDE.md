# pi-llm-go — maintainer working agreement

This file is the **maintainer's working agreement** for `pi-llm-go`. Keep it short. When something grows past a few lines, move it to a dedicated doc and link from here.

## What this repo is

A minimal, Go-native LLM adapter with streaming, tool calling, and extended thinking. Provider-agnostic `LLM` interface; built-in Anthropic Messages and OpenAI-compatible Chat Completions providers. ~1.5kLoC.

Sibling repo: [`pi-agent-go`](https://github.com/amittimalsina/pi-agent-go) — the single-loop agent layer built on top of this package.

## Stability

- **Pre-1.0 today.** API may change between minor versions; CHANGELOG documents every change.
- v1.0 lands once we've used both repos in production for ≥4 weeks without API churn and at least one external Go user has reviewed the surface.
- Post-1.0: strict semver. Breaking changes require a `vN+1` module path bump per Go's major-version policy.

## Hard rules

- **Atomic commits.** Each logical unit of work = one commit. Conventional commits with scope (`feat(provider): ...`). HEREDOC bodies, `Co-Authored-By: Claude Opus 4.7 (1M context)` trailer when AI-assisted. Each commit must build + test green.
- **Push, force-push, repo-creation, opening PRs require explicit human OK.** Atomic commits are local-by-default.
- **Every public type / func has a godoc comment** starting with the identifier name. `gofmt` enforced via CI.
- **Provider additions require: implementation + test against `httptest` fake + example update.** No live-API tests in CI; gated on env-var-driven workflows only.
- **No panics in public API.** Validate at boundaries; return `error`. Generic constructors like `tools.Typed[I,O]` may panic on schema-reflection failures since they're program-start operations, not per-call.
- **Sealed sum types stay sealed.** `Block` and `StreamEvent` have unexported marker methods. New block / event types ship as additive package-internal additions, not external implementations.

## Code conventions

- `context.Context` is the first parameter of every call that does I/O or could be cancelled.
- Streaming is `iter.Seq2[T, error]` — never callbacks, never async-iterator-plus-Promise hybrids.
- Errors expose sentinels (`ErrAuth`, `ErrRateLimit`, …) wrapped in `*APIError`. Callers branch via `errors.Is` / `errors.As`.
- No env-var reads inside library code. Callers pass keys explicitly via constructor options.
- No model registry. Provider packages expose typed string constants for canonical model IDs as IDE-autocomplete convenience.

## Currency you must keep current

- **Model-id constants in each provider package.** OpenAI and Anthropic churn names faster than Go's training cutoff. When in doubt, verify against the official docs (`platform.openai.com/docs/models`, `platform.claude.com/docs/en/about-claude/models/overview`) before updating.
- **Go floor.** Currently `go 1.23` (iter.Seq2 floor). Bump only when we adopt a feature above 1.23.
- **CI Go matrix.** Currently 1.23 + 1.24. Add the new stable when Go ships, drop the oldest when 1.23 becomes 1.21-old (Go supports 2 minor versions at a time).

## Adding a new provider checklist

1. New package under `providers/<name>/`.
2. Provider struct implementing `LLM.Stream`. Constructor `New(Options) (*Provider, error)`.
3. Wire-level converter for `Request` → provider payload, `Message` round-trip.
4. SSE decoder mapping provider events → `llm.StreamEvent`.
5. Tests against `httptest.NewServer` covering: text streaming, tool-call streaming, request-body shape, HTTP error wrapping. No live API in CI.
6. Update `examples/streaming/main.go` if the new provider needs a flag.
7. `CHANGELOG.md` entry under `[Unreleased]` → `Added`.

## Releases

Tag `vX.Y.Z` (signed). GitHub Actions workflow `release.yml` picks up the tag, copies the matching CHANGELOG section into the release notes, and publishes.

Bumping the pi-agent-go dependency on pi-llm-go is a separate PR in that repo with its own changelog entry.

## License

MIT. Attribution to upstream `pi-ai` (Mario Zechner, MIT) in README and LICENSE.

## See also

- [README.md](README.md) — user-facing intro and quickstart.
- [CHANGELOG.md](CHANGELOG.md) — per-release changes.
- Parent monorepo (private during dogfooding) tracks the broader Noumenal context.
