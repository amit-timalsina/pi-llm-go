// Package all constructs any built-in provider from a configuration value,
// so choosing a provider at runtime is a supported operation rather than a
// switch each consumer rewrites.
//
//	p, err := all.Open(all.Spec{Provider: all.Name(cfg.Provider), APIKey: key, Retry: &retry})
//
// It lives here, rather than as llm.Open, because provider packages import
// the root package: the root cannot import them back without a cycle. This
// package is a leaf that depends on all four, so importing it links all
// four into the binary. Code that needs only one provider should keep
// calling that provider's New directly.
//
// Reasoning effort and thinking summaries are NOT configured here. Both ride
// on llm.Request.Thinking per request (see llm.ThinkingConfig), so the
// posture belongs to the call rather than to the construction — which also
// means a provider opened here can vary effort per request.
package all

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
	"github.com/amit-timalsina/pi-llm-go/providers/gemini"
	"github.com/amit-timalsina/pi-llm-go/providers/openai"
	openai_responses "github.com/amit-timalsina/pi-llm-go/providers/openai_responses"
)

// Name identifies a built-in provider. The values match the provider
// package names, so a config string maps to the import a reader expects.
type Name string

const (
	Anthropic       Name = "anthropic"
	OpenAI          Name = "openai"
	OpenAIResponses Name = "openai_responses"
	Gemini          Name = "gemini"
)

// ErrUnknownProvider is returned when Spec.Provider names no built-in
// provider. Branch on it with errors.Is.
var ErrUnknownProvider = errors.New("all: unknown provider")

// ErrUnsupportedOption is returned when a Spec field has no counterpart on
// the chosen provider — Headers against Anthropic, say. Dropping the field
// silently would hand back a provider configured differently than the
// config asked for.
var ErrUnsupportedOption = errors.New("all: unsupported option for provider")

// Spec is the portable subset of provider configuration: what a config file
// realistically carries. Provider-specific knobs outside it (Anthropic's
// Beta headers, OpenAI's OrgID) stay on the individual New functions.
type Spec struct {
	Provider Name
	APIKey   string

	// BaseURL overrides the provider's default host, for OpenAI-compatible
	// endpoints and regional deployments.
	BaseURL string

	// URL, when set, is used verbatim as the endpoint instead of being
	// derived from BaseURL. OpenAI family only — Azure needs it.
	URL string

	// Headers are merged into every outgoing request. OpenAI family only —
	// Azure authenticates with an "api-key" header rather than a bearer.
	Headers map[string]string

	HTTPClient *http.Client
	Retry      *llm.RetryPolicy
}

// Names returns the built-in provider names, in a stable order, for
// validating config and building help text.
func Names() []Name {
	return []Name{Anthropic, OpenAI, OpenAIResponses, Gemini}
}

// Open constructs the named provider. The returned llm.LLM behaves exactly
// as the provider's own New would; nothing is wrapped.
func Open(spec Spec) (llm.LLM, error) {
	switch spec.Provider {
	case Anthropic:
		if err := spec.rejectOpenAIOnly(); err != nil {
			return nil, err
		}
		return anthropic.New(anthropic.Options{
			APIKey:     spec.APIKey,
			BaseURL:    spec.BaseURL,
			HTTPClient: spec.HTTPClient,
			Retry:      spec.Retry,
		})

	case OpenAI:
		return openai.New(openai.Options{
			APIKey:     spec.APIKey,
			BaseURL:    spec.BaseURL,
			URL:        spec.URL,
			Headers:    spec.Headers,
			HTTPClient: spec.HTTPClient,
			Retry:      spec.Retry,
		})

	case OpenAIResponses:
		return openai_responses.New(openai_responses.Options{
			APIKey:     spec.APIKey,
			BaseURL:    spec.BaseURL,
			URL:        spec.URL,
			Headers:    spec.Headers,
			HTTPClient: spec.HTTPClient,
			Retry:      spec.Retry,
		})

	case Gemini:
		if err := spec.rejectOpenAIOnly(); err != nil {
			return nil, err
		}
		return gemini.New(gemini.Options{
			APIKey:     spec.APIKey,
			BaseURL:    spec.BaseURL,
			HTTPClient: spec.HTTPClient,
			Retry:      spec.Retry,
		})

	default:
		return nil, fmt.Errorf("%w: %q (want one of %s)", ErrUnknownProvider, spec.Provider, joinNames())
	}
}

func (s Spec) rejectOpenAIOnly() error {
	var unsupported []string
	if s.URL != "" {
		unsupported = append(unsupported, "URL")
	}
	if len(s.Headers) > 0 {
		unsupported = append(unsupported, "Headers")
	}
	if len(unsupported) == 0 {
		return nil
	}
	return fmt.Errorf("%w: %s does not accept %s; use BaseURL, or construct the provider directly",
		ErrUnsupportedOption, s.Provider, strings.Join(unsupported, " / "))
}

func joinNames() string {
	out := make([]string, 0, len(Names()))
	for _, n := range Names() {
		out = append(out, string(n))
	}
	return strings.Join(out, ", ")
}
