# pi-llm-go roadmap

This is the maintainer's working plan. Items aren't promises — they're
ranked by user-value-per-LOC and informed by WWMD audits against Mario
Zechner's [pi-ai](https://github.com/badlogic/pi-mono/tree/main/packages/ai).
Reordering happens when reality changes.

## Status

- **v0.5.0** landing — Gemini Files API helper
  (`providers/gemini/files`): Upload / Wait / Get / Delete with
  multipart upload, ACTIVE-state polling, and ~2 GB file ceiling.
  Closes the >20 MB video gap left by v0.4.0. Tag stamped on merge.
- **v0.4.0** shipped 2026-05-12 — Gemini provider + `llm.VideoBlock` for
  native multimodal video input (Gemini-exclusive; Anthropic and OpenAI
  providers reject at the wire boundary).
- **v0.3.0** shipped 2026-05-11 — `ImageBlock` multimodal image input
  across the three pre-Gemini providers.
- **v0.2.0** shipped 2026-05-11 — CacheRetention knob (WWMD convergence).
- **v1.0 ETA:** unknown. v1.0 requires ≥4 weeks production use without
  API churn + ≥1 external Go reviewer of the public surface.

## Near-term (next 1–3 minor releases)

### v0.6.0 — token counting + cost helpers

- `Provider.CountTokens(ctx, Request) (int, error)` — Anthropic, OpenAI,
  and Gemini all expose this without spending an inference call.
  Useful for cost guardrails before the request flies.
- Optional `Cost(usage Usage, model string) (input, output, total float64)`
  per provider with a maintained pricing table. Mario does not ship
  this; Go consumers keep reinventing it; ours can.

### v0.7.0 — retry middleware + finer-grained errors

- Provider-side retry on rate limits + 5xx (configurable base/max
  delay; default off; opt-in via `Options.Retry`). Keeps the no-magic
  default while making the common ask one-liner cheap.
- More granular error sentinels: `ErrContextLength`, `ErrPolicyViolation`,
  `ErrContentFilter`, `ErrServerOverloaded`. Currently all collapse
  into `ErrProvider`, forcing callers to string-match.

## Mid-term (v0.7+)

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

## v1.0 readiness checklist

- [ ] Multimodal input shipped (most common pre-1.0 ask).
- [ ] Three providers + Azure compat verified in production.
- [ ] Public API surface frozen for ≥4 weeks of real use.
- [ ] At least one external Go reviewer has read the API and not
      requested breaking changes.
- [ ] `pkg.go.dev` `Example_*` tests for every exported type.
- [ ] `examples/observability/` shipped and referenced from the README.
- [ ] CONTRIBUTING.md walks a contributor through adding a new provider.

## Convergence work — closed

WWMD audits of `cache_control` and PromptBuilder drove the v0.2.0
rewrite. No open WWMD divergence as of 2026-05-11.
