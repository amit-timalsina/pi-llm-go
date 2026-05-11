// multimodal: send an image with a text question, print the model's
// description.
//
//	export ANTHROPIC_API_KEY=...
//	go run ./examples/multimodal
//	go run ./examples/multimodal --openai           # use OpenAI Chat Completions
//	go run ./examples/multimodal --image path.png   # use your own image
//	go run ./examples/multimodal --prompt "what color is the dot?"
//
// When --image is unset, the example generates a small recognizable
// red-square-on-white PNG at runtime and saves a copy alongside so you
// can confirm the model is seeing what you think.
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/anthropic"
	"github.com/amit-timalsina/pi-llm-go/providers/openai"
)

func main() {
	useOpenAI := flag.Bool("openai", false, "use OpenAI Chat Completions instead of Anthropic")
	imagePath := flag.String("image", "", "path to a local image file (jpeg/png/gif/webp); generated PNG if unset")
	prompt := flag.String("prompt", "Describe this image in one sentence.", "user text prompt")
	flag.Parse()

	imgBytes, mime, savedPath, err := loadOrGenerate(*imagePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "image:", err)
		os.Exit(1)
	}
	if savedPath != "" {
		fmt.Fprintf(os.Stderr, "[generated reference image at %s]\n", savedPath)
	}

	provider, model, err := buildProvider(*useOpenAI)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	req := llm.Request{
		Model:     model,
		MaxTokens: 256,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock{Text: *prompt},
				llm.ImageBlock{
					Data:     base64.StdEncoding.EncodeToString(imgBytes),
					MimeType: mime,
				},
			}},
		},
	}

	msg, err := llm.Complete(context.Background(), provider, req)
	if err != nil {
		fmt.Fprintln(os.Stderr, "complete:", err)
		os.Exit(1)
	}
	for _, block := range msg.Content {
		if tb, ok := block.(llm.TextBlock); ok {
			fmt.Println(tb.Text)
		}
	}
	fmt.Fprintf(os.Stderr, "\nusage: in=%d out=%d total=%d\n",
		msg.Usage.InputTokens, msg.Usage.OutputTokens, msg.Usage.TotalTokens)
}

func loadOrGenerate(path string) (data []byte, mime, savedPath string, err error) {
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, "", "", fmt.Errorf("read %s: %w", path, err)
		}
		return b, mimeFromPath(path), "", nil
	}
	// Generate a 256x256 image: white background with a centered 128x128
	// red square. 256x256 sits above the minimum resolution most vision
	// models need to reliably distinguish foreground from background
	// (anything much smaller and models start hallucinating colors).
	const side = 256
	const dot = 128
	img := image.NewRGBA(image.Rect(0, 0, side, side))
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			img.Set(x, y, color.White)
		}
	}
	off := (side - dot) / 2
	for y := off; y < off+dot; y++ {
		for x := off; x < off+dot; x++ {
			img.Set(x, y, color.RGBA{R: 220, G: 30, B: 30, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, "", "", fmt.Errorf("encode png: %w", err)
	}
	f, err := os.CreateTemp("", "pi-llm-multimodal-*.png")
	if err == nil {
		_, _ = f.Write(buf.Bytes())
		_ = f.Close()
		savedPath = f.Name()
	}
	return buf.Bytes(), "image/png", savedPath, nil
}

func mimeFromPath(p string) string {
	switch strings.ToLower(filepath.Ext(p)) {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	}
	return "image/png"
}

func buildProvider(useOpenAI bool) (llm.LLM, string, error) {
	if useOpenAI {
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			return nil, "", fmt.Errorf("OPENAI_API_KEY is required when --openai is set")
		}
		p, err := openai.New(openai.Options{APIKey: key})
		if err != nil {
			return nil, "", err
		}
		return p, openai.GPT5_5, nil
	}
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return nil, "", fmt.Errorf("ANTHROPIC_API_KEY is required (or pass --openai to use OpenAI)")
	}
	p, err := anthropic.New(anthropic.Options{APIKey: key})
	if err != nil {
		return nil, "", err
	}
	return p, anthropic.ClaudeSonnet4_6, nil
}
