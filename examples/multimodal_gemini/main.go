// multimodal_gemini: text / image / video understanding against Gemini.
//
//	export GEMINI_API_KEY=...
//	go run ./examples/multimodal_gemini                       # text-only
//	go run ./examples/multimodal_gemini --image path.png      # vision
//	go run ./examples/multimodal_gemini --video path.mp4      # native video (inline if <20MB)
//	go run ./examples/multimodal_gemini --prompt "..."        # custom prompt
//	go run ./examples/multimodal_gemini --model gemini-3-flash-preview  # pick a different model
//
// When --video is set on a file larger than ~20MB you should upload it
// via the providers/gemini/files helper first and pass the file URI
// via Go code — this example only inlines the smaller case.
package main

import (
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/amit-timalsina/pi-llm-go"
	"github.com/amit-timalsina/pi-llm-go/providers/gemini"
)

func main() {
	model := flag.String("model", gemini.Gemini2_5Flash, "Gemini model id")
	imagePath := flag.String("image", "", "path to an image file (jpeg/png/gif/webp)")
	videoPath := flag.String("video", "", "path to a video file (mp4/mov/webm); inline-only at this size")
	videoURI := flag.String("video-uri", "", "URI of an uploaded or YouTube video (e.g. https://www.youtube.com/watch?v=...)")
	prompt := flag.String("prompt", "", "user text prompt (defaults vary by mode)")
	flag.Parse()

	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		fmt.Fprintln(os.Stderr, "GEMINI_API_KEY required")
		os.Exit(2)
	}
	p, err := gemini.New(gemini.Options{APIKey: key})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	userText := *prompt
	if userText == "" {
		switch {
		case *videoPath != "" || *videoURI != "":
			userText = "Describe what happens in this video in one short paragraph."
		case *imagePath != "":
			userText = "Describe this image in one sentence."
		default:
			userText = "Reply with exactly: HELLO from Gemini."
		}
	}

	content := []llm.Block{llm.TextBlock{Text: userText}}
	if *imagePath != "" {
		b, err := os.ReadFile(*imagePath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "read image:", err)
			os.Exit(1)
		}
		content = append(content, llm.ImageBlock{
			Data:     base64.StdEncoding.EncodeToString(b),
			MimeType: mimeFromExt(*imagePath, "image/png"),
		})
	}
	if *videoPath != "" {
		b, err := os.ReadFile(*videoPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "read video:", err)
			os.Exit(1)
		}
		content = append(content, llm.VideoBlock{
			Data:     base64.StdEncoding.EncodeToString(b),
			MimeType: mimeFromExt(*videoPath, "video/mp4"),
		})
	}
	if *videoURI != "" {
		// URI path covers both pre-uploaded Files-API handles and
		// YouTube URLs. MimeType is optional when URI is set; the
		// server infers from the URI.
		content = append(content, llm.VideoBlock{URI: *videoURI})
	}

	msg, err := llm.Complete(context.Background(), p, llm.Request{
		Model:     *model,
		MaxTokens: 512,
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: content}},
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "complete:", err)
		os.Exit(1)
	}
	for _, b := range msg.Content {
		if tb, ok := b.(llm.TextBlock); ok {
			fmt.Println(tb.Text)
		}
	}
	fmt.Fprintf(os.Stderr, "\nusage: in=%d out=%d total=%d stop=%s\n",
		msg.Usage.InputTokens, msg.Usage.OutputTokens, msg.Usage.TotalTokens, msg.StopReason)
}

func mimeFromExt(p, fallback string) string {
	switch strings.ToLower(filepath.Ext(p)) {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".mp4":
		return "video/mp4"
	case ".mov":
		return "video/quicktime"
	case ".webm":
		return "video/webm"
	case ".mpeg", ".mpg":
		return "video/mpeg"
	}
	return fallback
}
