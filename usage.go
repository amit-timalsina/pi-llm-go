package llm

// Usage records token accounting returned by a provider for a single
// completion request. Cache fields are zero when the provider doesn't bill
// or report cache reads/writes separately.
type Usage struct {
	InputTokens      int
	OutputTokens     int
	CacheReadTokens  int
	CacheWriteTokens int
	TotalTokens      int
}
