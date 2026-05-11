// prompt_caching: measures Anthropic prompt-cache hit on iteration 2.
//
// Two completions back-to-back with the same long stable prefix (system
// prompt + tool registry) and slightly different dynamic suffix. The first
// iteration warms the cache (you'll see CacheWriteTokens > 0); the second
// iteration hits it (CacheReadTokens > 0, with total input billed at
// cache-read rates instead of full input rates).
//
// Roughly 6x cost reduction on the cached portion, per Anthropic's pricing.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/prompt_caching
//	go run ./examples/prompt_caching -long   # use 1h TTL (auto-applies beta header)
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
)

// stableSystem must be long enough that Anthropic considers the prefix
// cacheable — minimum is ~1024 tokens at the cache_control marker point.
// We pad with a multi-section style guide so the system block alone clears
// the bar without depending on the tools section. Real-world prefixes
// (large few-shot rosters, schema docs, domain knowledge) hit this
// naturally; this synthetic example pads to make the win measurable.
const stableSystem = `You are a careful technical assistant operating in a long-running session that benefits from prompt caching.

# Procedure

When asked a question, follow this procedure:
1. Identify the core ask in one sentence.
2. List the key constraints you are working under.
3. Walk through your reasoning in numbered steps.
4. State the conclusion plainly, then a one-line justification.
5. If you used any external information (tool results, prior context), cite which one informed the conclusion.

# Tool use

When using tools:
- Prefer one tool call at a time over batching, unless the tool calls are independent.
- Always state in plain language what you expect the tool to return BEFORE calling it.
- After receiving a tool result, briefly confirm whether the result matches your expectation. If it does not, name the gap and propose the next investigation step.
- Do not call the same tool twice with identical arguments unless an explicit retry is justified (e.g. transient error). Otherwise, prefer reading the prior result.
- For tools that produce large structured outputs, ask for a summary first; reach for the full payload only when the summary is insufficient for the decision you need to make.

# Uncertainty

When uncertain:
- Say so explicitly. Phrases like "I am not sure" or "I do not have enough information to commit to X" are preferred over confident-sounding hedges.
- Identify exactly what additional information would resolve the uncertainty.
- Suggest the next step, with an explicit prediction of what each branch would teach you.
- Distinguish between aleatory uncertainty (irreducible randomness) and epistemic uncertainty (resolvable by more information). Treat them differently: epistemic warrants investigation; aleatory warrants probabilistic reasoning.

# Answer length

When the answer is short:
- Do not pad. Brevity is a feature.
- A one-line answer is fine when one line suffices.
- Resist the urge to add unsolicited context.

When the answer is long:
- Use numbered sections with descriptive headings.
- Lead with the conclusion; expand the reasoning afterward.
- Repeat the question only if the answer would otherwise be ambiguous.

# Style

Style notes:
- No emojis.
- No "great question!" preambles.
- No restating the user's prompt back at them.
- Code blocks for code; prose for prose.
- Tables for tabular data; bullets for lists; numbered lists when order or step-count matters.

# Reasoning standards

When reasoning:
- Distinguish facts from inferences. Mark inferences as such.
- When you reason from analogy, name the source and target domain, and state at least one disanalogy.
- When you cite probabilities, be honest about whether they are computed, calibrated, or vibes.
- When you make a counterfactual argument, state the counterfactual explicitly.

# Tone

When discussing trade-offs:
- Lay out the alternatives with their costs and benefits.
- Make a recommendation, with the most important reason that drives it.
- Note when the trade-off is genuinely close (the alternatives are within a small factor on the load-bearing axes).

When pushing back on a request:
- Acknowledge what the user is trying to accomplish.
- Name the friction or risk.
- Propose at least one alternative path that addresses the same underlying need.

# Failure modes to avoid

- Do not over-hedge to the point of saying nothing.
- Do not under-hedge to the point of asserting beyond what you can support.
- Do not paste boilerplate that is unrelated to the question.
- Do not invent tool calls. Use only tools that have been declared.
- Do not invent citations or sources. If you do not know, say so.

# When you produce code

When producing code:
- Default to small, readable functions over clever one-liners.
- Use the language's idiomatic patterns rather than transliterating from another language.
- Include error handling where errors can actually occur; don't pad with empty try/catch blocks.
- Type signatures and function names are documentation; spend effort on naming.

# When you produce design recommendations

When producing design recommendations:
- Lead with the principle the recommendation is grounded in.
- Identify at least one alternative considered and why it was rejected.
- Surface assumptions explicitly so they can be checked.

# When evaluating prior work

When evaluating prior work:
- Steelman the original intent first.
- Identify what is load-bearing vs. cosmetic.
- Propose changes in the order of impact-to-effort ratio, with the highest-ratio change first.`

func main() {
	useLongTTL := flag.Bool("long", false, "use 1h TTL (auto-applies extended-cache-ttl beta header)")
	flag.Parse()

	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		fmt.Fprintln(os.Stderr, "ANTHROPIC_API_KEY is required")
		os.Exit(2)
	}
	p, err := anthropic.New(anthropic.Options{APIKey: key})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	retention := llm.CacheRetentionShort
	if *useLongTTL {
		retention = llm.CacheRetentionLong
	}

	// Cacheable prefix: long system prompt + tool registry. Setting
	// CacheRetention asks the provider to auto-place ephemeral breakpoints
	// at the static prefix boundary (system trailing block, last tool, last
	// user text block). Anthropic returns a cache hit on byte-identical
	// prefixes in subsequent requests.
	baseReq := llm.Request{
		Model:          anthropic.ClaudeSonnet4_6,
		System:         stableSystem,
		CacheRetention: retention,
		MaxTokens:      512,
	}

	// Make the prompt long enough to be cacheable. Anthropic's minimum
	// cacheable size is ~1024 tokens; we pad with a tool registry to clear
	// that bar even with the modest system prompt above.
	for i := 0; i < 6; i++ {
		baseReq.Tools = append(baseReq.Tools, llm.Tool{
			Name:        fmt.Sprintf("noop_%d", i),
			Description: strings.Repeat("This tool does nothing in this example; it exists to push us past Anthropic's minimum cacheable prefix size. ", 4),
			InputSchema: []byte(`{"type":"object","properties":{},"additionalProperties":false}`),
		})
	}

	ctx := context.Background()
	// Two self-contained prompts — each iteration is an independent
	// completion call (no shared message history). The only thing the
	// cache sees as identical between iterations is the prefix (system +
	// tools), which is exactly what we want to measure.
	prompts := []string{
		"What is 17 * 23?",
		"What is the prime factorization of 391?",
	}

	for i, prompt := range prompts {
		fmt.Printf("\n--- iteration %d ---\n", i+1)
		fmt.Printf("user: %s\n", prompt)

		req := baseReq
		req.Messages = []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock{Text: prompt}}},
		}

		msg, err := llm.Complete(ctx, p, req)
		if err != nil {
			fmt.Fprintln(os.Stderr, "complete error:", err)
			os.Exit(1)
		}

		// Render assistant text.
		fmt.Print("assistant: ")
		for _, block := range msg.Content {
			if tb, ok := block.(llm.TextBlock); ok {
				fmt.Print(tb.Text)
			}
		}
		fmt.Println()

		// Cache telemetry — the headline metric.
		fmt.Printf("usage: in=%d out=%d cache_write=%d cache_read=%d\n",
			msg.Usage.InputTokens,
			msg.Usage.OutputTokens,
			msg.Usage.CacheWriteTokens,
			msg.Usage.CacheReadTokens,
		)
	}

	fmt.Println("\nIteration 1 should show cache_write > 0 (prefix warms the cache).")
	fmt.Println("Iteration 2 should show cache_read > 0 (prefix hit on the same content).")
	if *useLongTTL {
		fmt.Println("With -long, the cache survives ~1h instead of ~5min.")
	}
}
