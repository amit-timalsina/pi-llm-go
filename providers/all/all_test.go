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
