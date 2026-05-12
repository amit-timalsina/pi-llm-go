package gemini

// Canonical Gemini model IDs verified against the live /v1beta/models
// listing on 2026-05-12. IDE-autocomplete-friendly constants; callers
// can also pass a raw string for models pi-llm-go doesn't yet know
// about (the provider doesn't validate model IDs locally).
//
// All listed models support multimodal input (text + image + video +
// audio). Verify pricing and feature flags against
// https://ai.google.dev/gemini-api/docs/models before relying on a
// specific constant in production.
const (
	// Gemini 3 family — frontier multimodal. Preview-tier model IDs
	// have unstable behavior; Google may rename or rev them without
	// the formal deprecation cycle.
	Gemini3FlashPreview       = "gemini-3-flash-preview"
	Gemini3ProPreview         = "gemini-3-pro-preview"
	Gemini3_1ProPreview       = "gemini-3.1-pro-preview"
	Gemini3_1FlashLitePreview = "gemini-3.1-flash-lite-preview"
	Gemini3_1FlashLite        = "gemini-3.1-flash-lite"

	// Gemini Robotics ER — vision-language model with strong video
	// reasoning over object tracking, scene change, action segmentation.
	// Same generateContent surface as the rest of the family.
	GeminiRoboticsER1_6Preview = "gemini-robotics-er-1.6-preview"

	// Gemini 2.5 family — stable, cheaper tier.
	Gemini2_5Flash     = "gemini-2.5-flash"
	Gemini2_5FlashLite = "gemini-2.5-flash-lite"
	Gemini2_5Pro       = "gemini-2.5-pro"

	// "Latest" aliases — track whatever Google currently considers the
	// flagship for that tier. Convenient for examples; less suitable
	// for production where you want to pin a version.
	GeminiFlashLatest     = "gemini-flash-latest"
	GeminiFlashLiteLatest = "gemini-flash-lite-latest"
	GeminiProLatest       = "gemini-pro-latest"
)
