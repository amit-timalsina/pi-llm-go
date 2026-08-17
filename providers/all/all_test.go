package all_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/amit-timalsina/pi-llm-go/providers/all"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
	"github.com/amit-timalsina/pi-llm-go/providers/gemini"
	"github.com/amit-timalsina/pi-llm-go/providers/openai"
	openai_responses "github.com/amit-timalsina/pi-llm-go/providers/openai_responses"
)

// Every name resolves to its own provider — the switch consumers were
// writing by hand, pinned here instead.
func TestOpenResolvesEveryName(t *testing.T) {
	cases := []struct {
		name all.Name
		want any
	}{
		{all.Anthropic, &anthropic.Provider{}},
		{all.OpenAI, &openai.Provider{}},
		{all.OpenAIResponses, &openai_responses.Provider{}},
		{all.Gemini, &gemini.Provider{}},
	}
	for _, tc := range cases {
		t.Run(string(tc.name), func(t *testing.T) {
			p, err := all.Open(all.Spec{Provider: tc.name, APIKey: "k"})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			if got, want := typeString(p), typeString(tc.want); got != want {
				t.Errorf("Open(%q) built %s, want %s", tc.name, got, want)
			}
		})
	}
}

// Names() must stay in step with what Open accepts, or config validation
// built on it lies.
func TestNamesMatchOpen(t *testing.T) {
	for _, n := range all.Names() {
		if _, err := all.Open(all.Spec{Provider: n, APIKey: "k"}); err != nil {
			t.Errorf("Names() lists %q but Open rejects it: %v", n, err)
		}
	}
}

func TestOpenUnknownProviderNamesTheValidOnes(t *testing.T) {
	_, err := all.Open(all.Spec{Provider: "claude", APIKey: "k"})
	if err == nil {
		t.Fatal("want ErrUnknownProvider, got nil")
	}
	if !errors.Is(err, all.ErrUnknownProvider) {
		t.Fatalf("errors.Is(err, ErrUnknownProvider)=false: %v", err)
	}
	for _, n := range all.Names() {
		if !strings.Contains(err.Error(), string(n)) {
			t.Errorf("error should list %q so a typo is self-correcting: %v", n, err)
		}
	}
}

// A field the provider cannot honour is reported, not dropped — the same
// rule the request path follows for Thinking.
func TestOpenRejectsOptionsTheProviderCannotHonour(t *testing.T) {
	cases := []struct {
		name string
		spec all.Spec
	}{
		{"anthropic URL", all.Spec{Provider: all.Anthropic, APIKey: "k", URL: "https://example.test/v1/messages"}},
		{"anthropic Headers", all.Spec{Provider: all.Anthropic, APIKey: "k", Headers: map[string]string{"api-key": "k"}}},
		{"gemini URL", all.Spec{Provider: all.Gemini, APIKey: "k", URL: "https://example.test"}},
		{"gemini Headers", all.Spec{Provider: all.Gemini, APIKey: "k", Headers: map[string]string{"x": "y"}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := all.Open(tc.spec); !errors.Is(err, all.ErrUnsupportedOption) {
				t.Errorf("want ErrUnsupportedOption, got %v", err)
			}
		})
	}
}

// The OpenAI family does accept them — that is why Azure works.
func TestOpenAcceptsURLAndHeadersOnOpenAIFamily(t *testing.T) {
	for _, n := range []all.Name{all.OpenAI, all.OpenAIResponses} {
		if _, err := all.Open(all.Spec{
			Provider: n,
			APIKey:   "k",
			URL:      "https://example.openai.azure.com/openai/v1/responses?api-version=preview",
			Headers:  map[string]string{"api-key": "k"},
		}); err != nil {
			t.Errorf("Open(%q) with Azure-shaped config: %v", n, err)
		}
	}
}

// Construction errors from the underlying provider reach the caller intact.
func TestOpenPropagatesProviderValidation(t *testing.T) {
	for _, n := range all.Names() {
		if _, err := all.Open(all.Spec{Provider: n}); err == nil {
			t.Errorf("Open(%q) with no APIKey should fail", n)
		}
	}
}

func typeString(v any) string {
	switch v.(type) {
	case *anthropic.Provider:
		return "*anthropic.Provider"
	case *openai.Provider:
		return "*openai.Provider"
	case *openai_responses.Provider:
		return "*openai_responses.Provider"
	case *gemini.Provider:
		return "*gemini.Provider"
	default:
		return "unknown"
	}
}

func TestParseModelSplitsProviderAndBareID(t *testing.T) {
	cases := []struct {
		qualified string
		wantName  all.Name
		wantModel string
	}{
		{"anthropic:claude-opus-4-7", all.Anthropic, "claude-opus-4-7"},
		{"openai_responses:gpt-5.6-sol", all.OpenAIResponses, "gpt-5.6-sol"},
		{"openai:o3", all.OpenAI, "o3"},
		{"gemini:gemini-3.1-pro-preview", all.Gemini, "gemini-3.1-pro-preview"},
		// A gateway model whose own id contains a slash.
		{"openai:anthropic/claude-opus-5", all.OpenAI, "anthropic/claude-opus-5"},
		// An OpenAI fine-tune id contains colons of its own; the split is on
		// the first one, so the model half survives intact.
		{"openai:ft:gpt-4o-mini:acme::abc123", all.OpenAI, "ft:gpt-4o-mini:acme::abc123"},
	}
	for _, tc := range cases {
		t.Run(tc.qualified, func(t *testing.T) {
			name, model, err := all.ParseModel(tc.qualified)
			if err != nil {
				t.Fatalf("ParseModel: %v", err)
			}
			if name != tc.wantName {
				t.Errorf("provider=%q, want %q", name, tc.wantName)
			}
			if model != tc.wantModel {
				t.Errorf("model=%q, want %q", model, tc.wantModel)
			}
		})
	}
}

// A bare name is refused rather than guessed at, and the error teaches the
// qualified form.
func TestParseModelRejectsUnqualified(t *testing.T) {
	for _, bare := range []string{"gpt-5.6-sol", "claude-opus-4-7", "", ":gpt-4o", "openai:"} {
		t.Run(bare, func(t *testing.T) {
			_, _, err := all.ParseModel(bare)
			if !errors.Is(err, all.ErrUnqualifiedModel) {
				t.Fatalf("want ErrUnqualifiedModel, got %v", err)
			}
			for _, n := range all.Names() {
				if !strings.Contains(err.Error(), string(n)) {
					t.Errorf("error should list %q: %v", n, err)
				}
			}
		})
	}
}

// An unknown prefix is an unknown provider, not an unqualified id — the
// caller can tell "you forgot the prefix" from "that provider isn't real".
func TestParseModelRejectsUnknownPrefix(t *testing.T) {
	_, _, err := all.ParseModel("claude:claude-opus-4-7")
	if !errors.Is(err, all.ErrUnknownProvider) {
		t.Fatalf("want ErrUnknownProvider, got %v", err)
	}
	if errors.Is(err, all.ErrUnqualifiedModel) {
		t.Error("an unknown prefix must not also report as unqualified")
	}
}

func TestOpenModelReturnsProviderAndBareID(t *testing.T) {
	p, model, err := all.OpenModel("openai_responses:gpt-5.6-sol", all.Spec{APIKey: "k"})
	if err != nil {
		t.Fatalf("OpenModel: %v", err)
	}
	if got := typeString(p); got != "*openai_responses.Provider" {
		t.Errorf("built %s", got)
	}
	// The qualified form must never reach llm.Request.Model.
	if model != "gpt-5.6-sol" {
		t.Errorf("model=%q, want the bare id", model)
	}
}

// Azure: same model name, different endpoint and auth — the case a name
// prefix cannot express.
func TestOpenModelCarriesAzureShape(t *testing.T) {
	p, model, err := all.OpenModel("openai_responses:gpt-4o", all.Spec{
		APIKey:  "k",
		URL:     "https://example.openai.azure.com/openai/v1/responses?api-version=preview",
		Headers: map[string]string{"api-key": "k"},
	})
	if err != nil {
		t.Fatalf("OpenModel: %v", err)
	}
	if model != "gpt-4o" || typeString(p) != "*openai_responses.Provider" {
		t.Errorf("model=%q provider=%s", model, typeString(p))
	}
}

func TestOpenModelRejectsContradictingSpecProvider(t *testing.T) {
	_, _, err := all.OpenModel("anthropic:claude-opus-4-7", all.Spec{Provider: all.Gemini, APIKey: "k"})
	if !errors.Is(err, all.ErrConflictingProvider) {
		t.Fatalf("want ErrConflictingProvider, got %v", err)
	}
}

func TestOpenModelAllowsAgreeingSpecProvider(t *testing.T) {
	if _, _, err := all.OpenModel("gemini:gemini-2.5-flash", all.Spec{Provider: all.Gemini, APIKey: "k"}); err != nil {
		t.Fatalf("agreeing Spec.Provider should be fine: %v", err)
	}
}
