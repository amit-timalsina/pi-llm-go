# pi-llm-go

[![CI](https://github.com/amit-timalsina/pi-llm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/amit-timalsina/pi-llm-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/amit-timalsina/pi-llm-go.svg)](https://pkg.go.dev/github.com/amit-timalsina/pi-llm-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/amit-timalsina/pi-llm-go)](https://goreportcard.com/report/github.com/amit-timalsina/pi-llm-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Minimal, provider-agnostic LLM client for Go.** Streaming completions, tool calling, prompt caching, token counting, cost projection, and retry middleware for **Anthropic Claude** (Messages API), **OpenAI** (Chat Completions + Responses API, including GPT-5 family), **Google Gemini** (text / image / video), and any **OpenAI-compatible** endpoint (Azure OpenAI, Groq, Together, vLLM, OpenRouter, Ollama). Idiomatic Go: `iter.Seq2` streaming, sealed sum types, `errors.Is` sentinels, `context.Context` cancellation.

> Status: **v0.9.0, pre-1.0.** API may change between minor versions; see [CHANGELOG.md](CHANGELOG.md). Used internally at [Noumenal](https://noumenalai.com).

## Install

```bash
go get github.com/amit-timalsina/pi-llm-go
```

Requires Go 1.23+ (for `iter.Seq2`).

## Capability matrix

| Capability | Anthropic | OpenAI Chat | OpenAI Responses | Gemini |
|---|---|---|---|---|
| Streaming text | ✅ | ✅ | ✅ | ✅ |
| Tool calling | ✅ | ✅ | ✅ | ✅ |
| Image input | ✅ | ✅ | ✅ | ✅ |
| Video input | reject at wire | reject at wire | reject at wire | ✅ native |
| Extended thinking | ✅ | — | ✅ (reasoning summaries) | ✅ |
| Prompt caching (5m + 1h tier) | ✅ | automatic | automatic | single-TTL |
| Per-TTL cache-write Usage breakdown | ✅ | — | — | — |
| `CountTokens` (no-spend pre-flight) | ✅ | — (no API) | — (no API) | ✅ |
| Cost projection helper | ✅ | ✅ | ✅ | ✅ |
| Retry middleware (`Options.Retry`) | ✅ | ✅ | ✅ | ✅ |
| Files API helper (>20 MB inputs) | — | — | — | ✅ |

## Quickstart — one-shot Claude

```go
package main

import (
    "context"
    "fmt"
    "os"

    llm "github.com/amit-timalsina/pi-llm-go"
    "github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

func main() {
    p, _ := anthropic.New(anthropic.Options{APIKey: os.Getenv("ANTHROPIC_API_KEY")})

    msg, err := llm.Complete(context.Background(), p, llm.Request{
        Model:     anthropic.ClaudeSonnet4_6,
        MaxTokens: 1024,
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "Reply with one word: hello"}}},
        },
    })
    if err != nil { panic(err) }

    for _, block := range msg.Content {
        if tb, ok := block.(llm.TextBlock); ok {
            fmt.Println(tb.Text)
        }
    }
}
```

## Quickstart — streaming with `iter.Seq2`

```go
for event, err := range p.Stream(ctx, req) {
    if err != nil { /* handle */ break }
    if d, ok := event.(llm.EventTextDelta); ok {
        fmt.Print(d.Delta)
    }
}
```

`Stream` returns `iter.Seq2[llm.StreamEvent, error]`. `Complete` is a synchronous helper that drains the stream and returns the final assistant `Message`.

## Quickstart — tool calling

```go
req := llm.Request{
    Model:     anthropic.ClaudeSonnet4_6,
    MaxTokens: 1024,
    Tools: []llm.Tool{{
        Name:        "get_weather",
        Description: "Get the weather for a city",
        InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
    }},
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "What's the weather in Tokyo?"}}},
    },
}
msg, _ := llm.Complete(ctx, p, req)
for _, b := range msg.Content {
    if tc, ok := b.(llm.ToolCallBlock); ok {
        fmt.Printf("tool=%s args=%s\n", tc.Name, tc.Arguments)
        // Execute the tool yourself, then send a ToolResultBlock on the next turn.
        // For a built-in execution loop, use https://github.com/amit-timalsina/pi-agent-go
    }
}
```

## When to pick `pi-llm-go`

| You want | Pick |
|---|---|
| One Go interface across Anthropic / OpenAI / Gemini, streaming-first | **pi-llm-go** |
| Only OpenAI, no streaming abstraction needed | [sashabaranov/go-openai](https://github.com/sashabaranov/go-openai) |
| Only Anthropic, vendor SDK guarantees | [anthropics/anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go) |
| A heavyweight framework with chains, agents, memory, embeddings, vector stores | [tmc/langchaingo](https://github.com/tmc/langchaingo) |
| Built-in agent loop on top of a provider-agnostic LLM client | **pi-llm-go + [pi-agent-go](https://github.com/amit-timalsina/pi-agent-go)** |

`pi-llm-go` is the **streaming-completions** layer. ~1.5kLoC of plain Go: one interface, four providers, no model registries, no plugin systems, no abstractions beyond what the LLM wire format demands.

## Features

- **Streaming-first.** `Stream()` returns `iter.Seq2[StreamEvent, error]` — Go 1.23 iterators, no callbacks, no goroutine leaks. `Complete()` is the synchronous helper for one-shot use.
- **Sealed sum types.** `Block` and `StreamEvent` are interfaces with package-private marker methods. Type-switch exhaustively; the compiler tells you if you miss a case.
- **Tool calling.** Declare tools on `Request.Tools`; receive `ToolCallBlock`s on the response; send `ToolResultBlock`s back. Pi-llm-go does not execute tools — that's [pi-agent-go](https://github.com/amit-timalsina/pi-agent-go)'s job.
- **Extended thinking.** `ThinkingConfig{BudgetTokens: N}` on requests, surfaced as `ThinkingBlock` content. Anthropic-only at v1.
- **Open-closed providers.** Implement `LLM.Stream` to add custom providers; no plugin registry needed.
- **Errors that branch cleanly.** `errors.Is(err, llm.ErrRateLimit)` works through `*APIError` wraps; `errors.As(err, &apiErr)` gives you status + body.
- **Cancellation = `context.Context`.** No bespoke abort signal types.

## Providers

### Anthropic

```go
import "github.com/amit-timalsina/pi-llm-go/providers/anthropic"

p, _ := anthropic.New(anthropic.Options{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    // BaseURL: defaults to https://api.anthropic.com
    // Version: defaults to "2023-06-01"
    // Beta:    optional anthropic-beta header values
})
```

Honors `Request.Thinking`. Surfaces all content-block types (text, thinking, tool_use).

### OpenAI-compatible

```go
import "github.com/amit-timalsina/pi-llm-go/providers/openai"

p, _ := openai.New(openai.Options{
    APIKey:  os.Getenv("OPENAI_API_KEY"),
    BaseURL: "https://api.openai.com/v1",  // or "https://api.groq.com/openai/v1", etc.
})
```

Talks the Chat Completions wire format, so the same provider works against any compatible host. `Request.Thinking` is ignored at v1 — reasoning-effort dialects vary too much across compatible hosts to map portably.

### Gemini

```go
import "github.com/amit-timalsina/pi-llm-go/providers/gemini"

p, _ := gemini.New(gemini.Options{APIKey: os.Getenv("GEMINI_API_KEY")})
```

Native support for the Gemini 2.5 / 3 / Robotics ER 1.6 families. Same `LLM` interface as the other providers, plus **`llm.VideoBlock` for native video input** (Gemini is the only provider that accepts video natively; Anthropic and OpenAI reject `VideoBlock` at the wire boundary with a clear pointer to the frame-extraction workaround). YouTube URLs work directly:

```go
llm.VideoBlock{URI: "https://www.youtube.com/watch?v=..."}
```

For files larger than ~20 MB, the sibling `providers/gemini/files` sub-package handles the multipart upload + ACTIVE-state polling:

```go
import "github.com/amit-timalsina/pi-llm-go/providers/gemini/files"

fc, _ := files.New(files.Options{APIKey: os.Getenv("GEMINI_API_KEY")})
ref, _ := fc.Upload(ctx, mp4Reader, "video/mp4", files.UploadOptions{DisplayName: "demo.mp4"})
ref, _ = fc.Wait(ctx, ref, files.WaitOptions{}) // polls until ACTIVE
defer fc.Delete(context.Background(), ref.Name) // ~48h server-side TTL if you forget

// Now use the URI in a generateContent call:
content := []llm.Block{llm.TextBlock{Text: "describe"}, llm.VideoBlock{URI: ref.URI}}
```

Vertex AI (gs:// URIs + OAuth) is a planned future addition; v0.5 only supports the Google AI direct endpoint.

## Error handling

Non-2xx HTTP responses surface as `*llm.APIError` wrapping one of the typed sentinels — `errors.Is` works through the wrapping so consumers branch on category, not status code:

```
ErrAuth             // 401, 403
ErrRateLimit        // 429
ErrInvalidRequest   // other 4xx (parent of the next two)
├─ ErrContextLength // prompt / max_tokens exceeds the model's window
└─ ErrPolicyViolation // input rejected by content / safety policy
ErrProvider         // generic provider problem (parent of the next two)
├─ ErrServerError   // 5xx (excluding 529)
└─ ErrOverloaded    // 529 (Anthropic infra overload)
```

`ErrServerError` and `ErrOverloaded` both wrap `ErrProvider`; `ErrContextLength` and `ErrPolicyViolation` both wrap `ErrInvalidRequest`. Legacy `errors.Is(err, ErrProvider)` and `errors.Is(err, ErrInvalidRequest)` callers keep matching the full subtree.

The finer 4xx sentinels are derived by inspecting the response body — provider error schemas don't carry a canonical machine-readable category for "context too long" vs "policy violation," so pi-llm-go pattern-matches the message text. Patterns cover Anthropic / OpenAI / Gemini current shapes. For structured per-provider decoding, branch on `apiErr.Body` directly.

Sugar helpers and the parsed `Retry-After`:

```go
for ev, err := range provider.Stream(ctx, req) {
    if err == nil { /* consume ev */ continue }

    var apiErr *llm.APIError
    if errors.As(err, &apiErr) {
        switch {
        case llm.IsRateLimited(err):     // 429 → respect apiErr.RetryAfter
        case llm.IsOverloaded(err):      // 529 → short backoff, consider failover
        case llm.IsServerError(err):     // 5xx → retry, escalate if sustained
        case errors.Is(err, llm.ErrAuth):
        }
    }
    return err
}
```

`APIError.RetryAfter` is populated by all four built-in providers when the response carries a `Retry-After` (RFC 7231 delta-seconds or HTTP-date) or `retry-after-ms` (OpenAI's millisecond form, which wins when both are present).

### Retry middleware

Set `Options.Retry` on any provider to retry retriable errors (429 / 529 / 5xx / transient network failures) with exponential backoff + full jitter:

```go
p, _ := anthropic.New(anthropic.Options{
    APIKey: key,
    Retry:  &llm.RetryPolicy{MaxAttempts: 4, BaseDelay: time.Second, MaxDelay: 30 * time.Second},
})
// or use the sane defaults:
p2, _ := anthropic.New(anthropic.Options{APIKey: key, Retry: ptr(llm.DefaultRetryPolicy())})
```

Server-supplied `Retry-After` hints dominate the exponential schedule (capped at `MaxDelay`). `ErrAuth`, `ErrInvalidRequest`, and `context.Canceled`/`DeadlineExceeded` are NOT retried — caller intent always wins.

**Scope:** retry covers only the initial HTTP attempt. Once a 200 OK lands and streaming begins, the run is committed — a mid-stream connection break terminates the iterator rather than retrying, because resuming would replay events the consumer already saw. Callers needing at-least-once streaming should wrap pi-llm-go in their own idempotent replay layer.

`llm.RunWithRetry[T]` is exported so callers can build cross-provider fallback / circuit-breaker logic on top.

**Telemetry:** every retry emits a `slog.DebugContext` record so silent backoff windows are visible to ops:

```
DEBUG llm.retry.attempt   attempt=1 max_attempts=4 delay_ms=820 cause=overloaded retry_after_ms=500 error="..."
DEBUG llm.retry.exhausted max_attempts=4 last_cause=overloaded error="..."
```

Default slog level is INFO so these are silent unless you configure DEBUG. Field names are part of the v1 contract — write a custom `slog.Handler` to route them to Prometheus counters, dashboards, or wherever else. Otel users can bridge via [`otelslog`](https://pkg.go.dev/go.opentelemetry.io/contrib/bridges/otelslog).

## Cost telemetry

Every completion surfaces a typed `Usage` value on the final `EventMessageEnd` (and on the `*llm.Message` returned by `Complete` / `Accumulate`):

```go
msg, _ := llm.Complete(ctx, provider, req)
fmt.Printf("in=%d out=%d cache_write=%d (5m=%d, 1h=%d) cache_read=%d total=%d\n",
    msg.Usage.InputTokens, msg.Usage.OutputTokens,
    msg.Usage.CacheWriteTokens, msg.Usage.CacheWrite5mTokens, msg.Usage.CacheWrite1hTokens,
    msg.Usage.CacheReadTokens, msg.Usage.TotalTokens)
```

`CacheWrite5mTokens` and `CacheWrite1hTokens` are Anthropic-specific — they break `CacheWriteTokens` down by TTL tier so consumers can detect silent 5min fallback when `CacheRetention=long` was requested but the model didn't honor the extended TTL:

```go
if req.CacheRetention == llm.CacheRetentionLong &&
    msg.Usage.CacheWrite5mTokens > 0 && msg.Usage.CacheWrite1hTokens == 0 {
    // Model downgraded the cache hold to 5min. Cost projections that
    // assumed a 1h-cached prefix across iterations need to adjust.
}
```

OpenAI and Gemini leave the TTL-breakdown fields at zero (their cache surfaces are opaque or single-TTL).

`llm.ComputeCost` applies a built-in pricing table to a `Usage` value and returns a dollar breakdown:

```go
cost, err := llm.ComputeCost(msg.Usage, req.Model)
if err == nil {
    fmt.Printf("$%.4f total (in=$%.4f out=$%.4f cache_read=$%.4f cache_write_1h=$%.4f)\n",
        cost.Total(), cost.Input, cost.Output, cost.CacheRead, cost.CacheWrite1h)
}
```

The seed table covers the Claude 4, GPT-5, and Gemini 2.5/3.1 families (verified 2026-05-13). For any model not in the table — older snapshots, deprecated IDs, regional/batch tiers — call `llm.RegisterPricing(modelID, llm.Pricing{...})` at startup with rates pulled from your own source. Registered entries override the seed.

The TTL-breakdown fields from `Usage` carry through automatically: when `CacheWrite5mTokens` and `CacheWrite1hTokens` are populated, they price against their respective tiers independently — so silent 5min fallback is reflected in the cost projection without any caller-side branching.

## Token counting

Anthropic and Gemini expose a count-tokens endpoint that returns the input-token count without spending an inference call. Providers implement `llm.TokenCounter` when they support this:

```go
if c, ok := provider.(llm.TokenCounter); ok {
    n, err := c.CountTokens(ctx, req)
    if err == nil {
        fmt.Printf("request would consume %d input tokens\n", n)
    }
}
```

OpenAI's Chat Completions and Responses providers do NOT implement `TokenCounter` — OpenAI's tokenization is local-only via tiktoken, which pi-llm-go does not bundle. Callers needing a pre-flight count for an OpenAI-hosted model should run tiktoken themselves.

## Examples

Runnable examples in `examples/`:

- `examples/streaming` — basic streaming of a text response.
- `examples/tool_calling` — hand-rolled tool-call loop against `get_current_time`.
- `examples/multimodal` — image input on Anthropic + OpenAI.
- `examples/multimodal_gemini` — text / image / video against Gemini; `--video-upload PATH` exercises the Files API end-to-end.
- `examples/prompt_caching` — `CacheRetention` knob driving Anthropic prompt-cache hits.

Run them with `go run ./examples/streaming` (set `ANTHROPIC_API_KEY` first).

## Versioning

This package is pre-1.0. Anything can change between minor versions; refer to [CHANGELOG.md](CHANGELOG.md) for each release.

v1.0 lands when the API surface has been used in production for ≥4 weeks without churn. Post-1.0 follows semver strictly.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Designed and named after [pi-ai](https://github.com/earendil-works/pi/tree/main/packages/ai) by Mario Zechner (TypeScript, MIT). The wire-level vocabulary and event types follow the upstream's lead; the Go-native API surface (interface-based providers, iterator streaming, sealed sum types) is a from-scratch redesign for Go idioms.

Built and maintained by Amit Timalsina with Claude Code assistance — all design decisions and release calls are human-owned.
