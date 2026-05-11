// azure_openai: stream a completion from Azure OpenAI / Azure AI Services.
//
// The vanilla openai-compatible URL shape (BaseURL + "/chat/completions") does
// not fit Azure, whose chat endpoint embeds a deployment name and an
// api-version query parameter:
//
//	https://<resource>.cognitiveservices.azure.com/openai/deployments/<deployment>/chat/completions?api-version=...
//
// openai.Options.URL is the escape hatch: when set, it's used verbatim and
// BaseURL is ignored.
//
// Azure also uses an "api-key: <value>" header for data-plane auth instead of
// "Authorization: Bearer ..." that vanilla OpenAI uses. The Headers map on
// Options is the escape hatch for that.
//
// Usage:
//
//	# Either set AZURE_OPENAI_KEY directly (data-plane key from the portal /
//	# `az cognitiveservices account keys list`)...
//	export AZURE_OPENAI_KEY=...
//	go run ./examples/azure_openai
//
//	# ...or rely on `az login` and have this example fetch a Microsoft Entra
//	# bearer token for you (requires the principal to have
//	# "Cognitive Services OpenAI User" or similar role on the resource).
//	az login
//	go run ./examples/azure_openai --aad
//
// Env:
//
//	AZURE_OPENAI_URL  full chat-completions URL incl. api-version query
//	                  (default: anthropicgenesis / gpt-5.4-mini in eastus)
//	AZURE_OPENAI_KEY  data-plane key for api-key header (default auth mode)
//	AZURE_AD_TOKEN    pre-fetched AAD JWT; only consulted when --aad is set
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"

	llm "github.com/amittimalsina/pi-llm-go"
	"github.com/amittimalsina/pi-llm-go/providers/openai"
)

const defaultURL = "https://anthropicgenesis.cognitiveservices.azure.com/openai/deployments/gpt-5.4-mini/chat/completions?api-version=2025-02-01-preview"

// fetchAzureToken returns a Microsoft Entra access token for the
// cognitive-services audience. Uses AZURE_AD_TOKEN if set; otherwise
// shells out to `az account get-access-token`. The az CLI must be
// logged in (`az login`) and pointed at the right subscription. The
// signed-in principal must have a role granting the data action
// `Microsoft.CognitiveServices/accounts/OpenAI/deployments/chat/completions/action`
// (e.g. the "Cognitive Services OpenAI User" built-in role).
func fetchAzureToken() (string, error) {
	if t := os.Getenv("AZURE_AD_TOKEN"); t != "" {
		return t, nil
	}
	cmd := exec.Command(
		"az", "account", "get-access-token",
		"--resource", "https://cognitiveservices.azure.com",
		"--query", "accessToken",
		"-o", "tsv",
	)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("az account get-access-token: %w (run `az login` first?)", err)
	}
	return strings.TrimSpace(string(out)), nil
}

func main() {
	useAAD := flag.Bool("aad", false, "use Microsoft Entra (AAD) bearer auth via az CLI instead of the api-key header")
	flag.Parse()

	url := os.Getenv("AZURE_OPENAI_URL")
	if url == "" {
		url = defaultURL
	}

	opts := openai.Options{URL: url}
	if *useAAD {
		token, err := fetchAzureToken()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		// AAD JWT -> Authorization: Bearer <token> (the openai provider's default).
		opts.APIKey = token
	} else {
		key := os.Getenv("AZURE_OPENAI_KEY")
		if key == "" {
			fmt.Fprintln(os.Stderr, "AZURE_OPENAI_KEY is required (or pass --aad for AAD auth via az CLI)")
			os.Exit(2)
		}
		// Data-plane key -> api-key: <value> header.
		opts.Headers = map[string]string{"api-key": key}
	}

	p, err := openai.New(opts)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// The "model" field is required by the wire format but ignored by Azure
	// (the deployment name in the URL determines which model is invoked).
	// We pass the deployment name purely for completeness.
	req := llm.Request{
		Model: "gpt-5.4-mini",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: "In one sentence, what is a Go iterator?"},
			}},
		},
		MaxTokens: 256,
	}

	for event, err := range p.Stream(context.Background(), req) {
		if err != nil {
			fmt.Fprintln(os.Stderr, "\nstream error:", err)
			os.Exit(1)
		}
		if d, ok := event.(llm.EventTextDelta); ok {
			fmt.Print(d.Delta)
		}
		if e, ok := event.(llm.EventMessageEnd); ok {
			fmt.Printf("\n\n[stop=%s tokens in/out=%d/%d]\n",
				e.StopReason, e.Usage.InputTokens, e.Usage.OutputTokens)
		}
	}
}
