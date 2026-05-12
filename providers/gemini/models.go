package gemini

// Canonical Gemini model IDs available on Google AI as of 2026-05-12.
// IDE-autocomplete-friendly constants; callers can also pass a raw
// string for models pi-llm-go doesn't (yet) know about.
//
// All listed models support multimodal input (text + image + video +
// audio). Verify pricing and feature flags against
// https://ai.google.dev/gemini-api/docs/models before relying on the
// constants in production.
const (
	// Gemini 3 family — frontier multimodal.
	Gemini3FlashPreview = "gemini-3-flash-preview"
	Gemini3_1Pro        = "gemini-3.1-pro"
	Gemini3_1FlashLite  = "gemini-3.1-flash-lite"

	// Gemini Robotics ER — vision-language model with strong video
	// reasoning over object tracking, scene change, action segmentation.
	// Same generateContent surface as the rest of the family.
	GeminiRoboticsER1_6Preview = "gemini-robotics-er-1.6-preview"

	// Gemini 2.5 family — stable, cheaper tier.
	Gemini2_5Flash     = "gemini-2.5-flash"
	Gemini2_5FlashLite = "gemini-2.5-flash-lite"
	Gemini2_5Pro       = "gemini-2.5-pro"
)
