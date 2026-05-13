# pi-llm-go

[![CI](https://github.com/amit-timalsina/pi-llm-go/actions/workflows/ci.yml/badge.svg)](https://github.com/amit-timalsina/pi-llm-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/amit-timalsina/pi-llm-go.svg)](https://pkg.go.dev/github.com/amit-timalsina/pi-llm-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/amit-timalsina/pi-llm-go)](https://goreportcard.com/report/github.com/amit-timalsina/pi-llm-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A minimal, Go-native LLM adapter with streaming, tool calling, and extended thinking. Provider-agnostic interface with built-in support for **Anthropic Messages**, **OpenAI Chat Completions** (covering OpenAI, Azure OpenAI, Groq, Together, vLLM, OpenRouter, Ollama), and the **OpenAI Responses API** (GPT-5-family server-side state + reasoning summaries).

> Status: **v0.x — pre-1.0**. API may change between minor versions; see [CHANGELOG.md](CHANGELOG.md).

## Why

The Go LLM library landscape forces you to pick between heavy vendor SDKs that don't compose, code-generated client surfaces that don't track new providers, or "framework" libraries that ship more concepts than you want. `pi-llm-go` is the opposite: ~1.5kLoc of plain Go that gives you an interface, two providers, and a streaming model that uses `iter.Seq2` like a normal iterator. No HTTP wrappers, no model registries, no provider-specific magic.

## Installation

```bash
go get github.com/amit-timalsina/pi-llm-go
```

Requires Go 1.23 or later (for `iter.Seq2`).

## Quickstart

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
            {Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hello"}}},
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

Streaming is `iter.Seq2[llm.StreamEvent, error]` — range over it:

```go
for event, err := range p.Stream(ctx, req) {
    if err != nil { /* handle */ }
    if d, ok := event.(llm.EventTextDelta); ok {
        fmt.Print(d.Delta)
    }
}
```

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
ErrAuth           // 401, 403
ErrRateLimit      // 429
ErrInvalidRequest // other 4xx
ErrProvider       // generic provider problem (parent of the next two)
├─ ErrServerError // 5xx (excluding 529)
└─ ErrOverloaded  // 529 (Anthropic infra overload)
```

`ErrServerError` and `ErrOverloaded` both wrap `ErrProvider` via `%w`, so legacy `errors.Is(err, llm.ErrProvider)` keeps matching 5xx + 529 unchanged.

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

OpenAI and Gemini leave the TTL-breakdown fields at zero (their cache surfaces are opaque or single-TTL). A first-party `Cost(usage, model)` helper with pricing tables is on the roadmap; until then, multiply per-token rates yourself.

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
