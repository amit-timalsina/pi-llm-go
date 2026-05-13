---
title: Choosing a Go LLM library in 2026
published: false
description: A technical comparison of pi-llm-go, sashabaranov/go-openai, anthropics/anthropic-sdk-go, langchaingo, and google/genai for production Go services.
tags: go, llm, anthropic, openai
canonical_url: https://github.com/amit-timalsina/pi-llm-go/blob/main/docs/blog/choosing-a-go-llm-library-2026.md
---

# Choosing a Go LLM library in 2026

If you're building anything in Go that talks to an LLM today, you have five real options. This post is a technical comparison — what each library does well, what it doesn't do at all, and which to pick for which use case. No marketing speak, no "and this is why ours is better." Just the tradeoffs.

Quick orientation: I maintain [`pi-llm-go`](https://github.com/amit-timalsina/pi-llm-go), one of the five. I'll be explicit about where it wins and where it doesn't.

## The five options

| Library | First commit | Provider scope | Style |
|---|---|---|---|
| [sashabaranov/go-openai](https://github.com/sashabaranov/go-openai) | 2023 | OpenAI Chat Completions + any OpenAI-compatible host | hand-written client, generated types |
| [anthropics/anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go) | 2024 | Anthropic Messages only | code-generated from OpenAPI spec |
| [google/genai](https://github.com/googleapis/go-genai) | 2024 | Google Gemini only | hand-written, official |
| [tmc/langchaingo](https://github.com/tmc/langchaingo) | 2023 | Multi-provider via adapter pattern | framework: chains + memory + vector stores + retrievers + agents |
| [pi-llm-go](https://github.com/amit-timalsina/pi-llm-go) | 2026 | Anthropic + OpenAI Chat + OpenAI Responses + Gemini + OpenAI-compatible | minimal interface, idiomatic Go iterators |

There are smaller community libraries too — `liushuangls/go-anthropic`, `openrouter/openrouter-go`, the `ollama` Go client. They're all narrower than the five above; I'll skip them for this post.

## Decision tree

```
Do you need just one provider, and is it OpenAI?
├── Yes → sashabaranov/go-openai
└── No
    │
    Do you need just one provider, and is it Anthropic?
    ├── Yes → anthropics/anthropic-sdk-go
    └── No
        │
        Do you need just one provider, and is it Gemini?
        ├── Yes → google/genai
        └── No (multi-provider, switchable at runtime)
            │
            Do you need chains, memory, vector stores, retrievers?
            ├── Yes → langchaingo
            └── No → pi-llm-go
```

Plain English: vendor SDKs win when you want first-party guarantees and don't need to switch providers. langchaingo wins when you're building a full RAG / chatbot stack with multiple components. pi-llm-go wins when you want the LLM-client layer alone, multi-provider, with idiomatic Go ergonomics.

## What each one feels like

### sashabaranov/go-openai

```go
client := openai.NewClient("sk-...")
resp, err := client.CreateChatCompletion(
    context.Background(),
    openai.ChatCompletionRequest{
        Model: openai.GPT4o,
        Messages: []openai.ChatCompletionMessage{
            {Role: openai.ChatMessageRoleUser, Content: "hello"},
        },
    },
)
```

The OpenAI Chat Completions object model, translated faithfully to Go. Active maintenance, large user base, works against any host that speaks the OpenAI wire format (Azure, Groq, Together, vLLM, OpenRouter, Ollama). Streaming is callback-based via a stream.Recv() loop — pre-Go-1.23 design.

**Strengths:**
- Single best Go OpenAI client by adoption.
- Compatible with everything that speaks the OpenAI wire format.
- Stable API across many releases.

**Limitations:**
- OpenAI Chat Completions only. The Responses API isn't supported (no GPT-5 server-side state, no reasoning summaries). No Anthropic, no Gemini.
- Streaming uses `stream.Recv()` polling, not iterators. Idiomatic in 2023; less so post-Go-1.23.
- No first-party cost helpers, no token-counting wrappers.

### anthropics/anthropic-sdk-go

```go
client := anthropic.NewClient(option.WithAPIKey(key))
msg, err := client.Messages.New(ctx, anthropic.MessageNewParams{
    Model:     anthropic.F(anthropic.ModelClaudeSonnet4_6),
    MaxTokens: anthropic.F(int64(1024)),
    Messages: anthropic.F([]anthropic.MessageParam{
        anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
    }),
})
```

Anthropic's official Go SDK, auto-generated from their OpenAPI spec. Every parameter is an `anthropic.F(...)` wrapper (the "Field" type) to disambiguate "unset" from "zero value." Streaming returns a `*ssestream.Stream[MessageStreamEventUnion]` that you `.Next()` through.

**Strengths:**
- Official. Tracks the API in lockstep with each new release.
- Comprehensive: every Anthropic feature surfaces.
- Strict typing on every field via the `F()` wrapper.

**Limitations:**
- Anthropic only.
- Auto-generated style: lots of long type names, `F()` wrappers everywhere. Idiomatic for the codegen, less so for hand-written Go.
- `ssestream.Stream` is its own thing — not `iter.Seq2`, not a channel.

### google/genai

```go
client, _ := genai.NewClient(ctx, &genai.ClientConfig{APIKey: key})
resp, err := client.Models.GenerateContent(ctx, "gemini-2.5-flash", []*genai.Content{
    genai.NewContentFromText("hello", genai.RoleUser),
}, nil)
```

Google's official Gemini SDK. Streaming via an iterator. Native multimodal: image, video, audio. Tracks the Generative Language API closely.

**Strengths:**
- Official. Best-in-class for Gemini-specific features (video, audio, file API).
- Active development, will track Gemini 4 / 5 / etc.

**Limitations:**
- Gemini only.
- The API surface mirrors the protobuf shape; not always the cleanest Go.

### tmc/langchaingo

```go
llm, _ := openai.New()
completion, _ := llms.GenerateFromSinglePrompt(ctx, llm, "hello")
```

The Go port of langchain. Full framework: chains, prompt templates, memory, vector stores, retrievers, agents, document loaders, embeddings. Multi-provider via the `llms.LLM` interface.

**Strengths:**
- Largest feature surface of any Go LLM library.
- Multi-provider out of the box.
- If you want a "Go RAG app in one repo," this is it.

**Limitations:**
- Big surface. The `llms` package is the small part; everything else (vectorstores, agents, chains, memory) brings a lot of concepts. If you only want completions, it's overkill.
- Streaming uses a callback (`streaming.Func`), not an iterator.
- Different release cadence than the vendor SDKs — sometimes lags new Anthropic / OpenAI features by weeks.

### pi-llm-go

```go
p, _ := anthropic.New(anthropic.Options{APIKey: key})
for event, err := range p.Stream(ctx, llm.Request{
    Model:     anthropic.ClaudeSonnet4_6,
    MaxTokens: 1024,
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: "hello"}}},
    },
}) {
    if err != nil { return err }
    if d, ok := event.(llm.EventTextDelta); ok {
        fmt.Print(d.Delta)
    }
}
```

A minimal provider-agnostic interface (`Stream` returning `iter.Seq2[StreamEvent, error]`), four built-in providers, and helpers for the things every consumer needs: token counting, cost projection, retry middleware, prompt caching.

**Strengths:**
- Idiomatic Go 1.23+: iterators, sealed sum types via type-switch, `context.Context` cancellation, `errors.Is` sentinels.
- Multi-provider with a single interface. Switch backends by changing the provider import.
- Compact: ~1.5kLoC. Readable in one sitting.
- First-party cost helpers + token counters that consume the cache-write TTL breakdown (Anthropic) so silent fallback shows up in projections.
- Retry middleware that honors server-supplied `Retry-After` hints, with full jitter and exponential backoff, gated behind `Options.Retry` so it's off by default.

**Limitations:**
- Pre-1.0. API can change between minor versions. v1.0 lands once the API has settled in production use for ≥4 weeks.
- Smaller community than the alternatives — newer project.
- Not a framework: no chains, no vector stores, no agents. The companion library [pi-agent-go](https://github.com/amit-timalsina/pi-agent-go) provides a single-loop agent on top, but that's a separate decision.

## Streaming style: callback vs iterator vs polling

This is the biggest day-to-day usability difference between the libraries. Each represents a different vintage of "streaming in Go."

**Callback (langchaingo):**

```go
streaming := func(ctx context.Context, chunk []byte) error {
    fmt.Print(string(chunk))
    return nil
}
llms.GenerateFromSinglePrompt(ctx, llm, prompt, llms.WithStreamingFunc(streaming))
```

Wraps everything in a closure. Hard to combine with other control flow (early return, accumulator pattern, parallel fan-out).

**Polling (sashabaranov/go-openai, anthropic-sdk-go):**

```go
stream, _ := client.CreateChatCompletionStream(ctx, req)
defer stream.Close()
for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) { break }
    if err != nil { return err }
    fmt.Print(chunk.Choices[0].Delta.Content)
}
```

Manual `for-Recv()-EOF` loop. Works but feels like 2018-era Go.

**Iterator (pi-llm-go, google/genai):**

```go
for event, err := range p.Stream(ctx, req) {
    if err != nil { return err }
    if d, ok := event.(llm.EventTextDelta); ok {
        fmt.Print(d.Delta)
    }
}
```

`range`-loop with native error tuple. The shape Go 1.23 added iterators for. Composes with all the same control flow as any other range loop.

## Cost and token-counting helpers

This is genuinely underprovided in the Go ecosystem. Every team building serious LLM apps eventually needs:

- Pre-flight token counting (cost guardrails before the request flies).
- Dollar-cost projection from a `Usage` record.
- Per-TTL cache-write breakdown (Anthropic-specific, but loadbearing for cost projections when you use prompt caching).

| Library | Pre-flight count | Cost helper | TTL breakdown |
|---|---|---|---|
| sashabaranov/go-openai | no | no | n/a |
| anthropics/anthropic-sdk-go | yes (via Messages.CountTokens) | no | no |
| google/genai | yes (via Models.CountTokens) | no | n/a |
| langchaingo | no | no | n/a |
| pi-llm-go | yes (Anthropic + Gemini) | yes (Claude 4 / GPT-5 / Gemini 2.5+3.1 seeded; `RegisterPricing` for the rest) | yes |

If you don't care about this category, pick on streaming style + provider coverage. If you do care, pi-llm-go and the vendor SDKs are your options; langchaingo and go-openai leave it to you.

## Retry middleware

| Library | Built-in retry | Honors Retry-After | Categorical errors |
|---|---|---|---|
| sashabaranov/go-openai | no (caller's job) | n/a | string-match on error |
| anthropics/anthropic-sdk-go | yes (RetryShouldRetry option) | yes | strongly typed |
| google/genai | yes (basic) | partial | partial |
| langchaingo | varies by provider adapter | varies | varies |
| pi-llm-go | yes (`Options.Retry`) | yes | `errors.Is(err, llm.ErrRateLimit)` etc. |

This is mostly a "do you want a one-liner default or do you want to write a retry loop yourself" choice. If you're rolling your own retry, every library above is workable — they all surface enough information to write a retry loop. If you want a default that just works, pi-llm-go and anthropic-sdk-go are the two with reasonable retry built in.

## API stability

| Library | Stability story |
|---|---|
| sashabaranov/go-openai | Stable. Few breaking changes across many releases. |
| anthropics/anthropic-sdk-go | Tracks Anthropic API; breaking changes when the API breaks. |
| google/genai | Tracks Gemini API. |
| langchaingo | Pre-1.0; breaking changes happen. |
| pi-llm-go | Pre-1.0 (v0.x). v1.0 gated on ≥4 weeks production use without churn. |

If you can't tolerate breaking changes between minor versions, the vendor SDKs are the safest bet. If you can — or you're prepared to pin to a specific minor — the multi-provider libraries (langchaingo, pi-llm-go) save you re-writing the same provider-switching glue.

## Decision summary

- **OpenAI-only, stable:** sashabaranov/go-openai.
- **Anthropic-only, official:** anthropics/anthropic-sdk-go.
- **Gemini-only, official:** google/genai.
- **Full RAG / agents / vector store stack in Go:** langchaingo.
- **Multi-provider streaming completions with first-party token-count + cost + retry, no framework baggage:** pi-llm-go.

If you're building an agent on top of multi-provider completions, pi-llm-go + [pi-agent-go](https://github.com/amit-timalsina/pi-agent-go) gives you both layers separately so you can swap either.

## Honest tradeoffs for pi-llm-go

If you're considering pi-llm-go specifically, here are the things I'd want to know before adopting it:

1. **It's new.** May 2026. Smaller ecosystem than the alternatives. Used internally at [Noumenal](https://noumenalai.com) — that's the consumer-adoption signal driving v1.0 timing, but it's one signal.
2. **It's pre-1.0.** Breaking changes between minor versions are documented in `CHANGELOG.md`, but they happen.
3. **It's not a framework.** No retrievers, no vector stores, no chains, no prompt templates. If you want those, langchaingo or DIY are your options.
4. **Streaming-only.** No batch API support yet (planned mid-term).
5. **No automatic fallback between providers.** You instantiate one provider; switching providers means a new `provider := ...` call. Building a "fall back to Claude when OpenAI 5xx" requires layering `llm.RunWithRetry` + your own dispatch on top.

If those are dealbreakers, one of the other four is the right pick. If they're not, give it a try and open an issue.

---

*If you want to dig in: [pi-llm-go on GitHub](https://github.com/amit-timalsina/pi-llm-go) · [pi-agent-go on GitHub](https://github.com/amit-timalsina/pi-agent-go) · [pkg.go.dev/pi-llm-go](https://pkg.go.dev/github.com/amit-timalsina/pi-llm-go).*
