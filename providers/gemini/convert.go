package gemini

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"time"

	llm "github.com/amit-timalsina/pi-llm-go"
)

// requestBody is the JSON body posted to :streamGenerateContent.
// Subset of the Gemini API spec — covers what pi-llm-go currently
// surfaces.
type requestBody struct {
	Contents          []apiContent      `json:"contents"`
	SystemInstruction *apiSystem        `json:"systemInstruction,omitempty"`
	Tools             []apiTool         `json:"tools,omitempty"`
	ToolConfig        *apiToolConfig    `json:"toolConfig,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

// apiToolConfig is Gemini's top-level tool_choice analog. function_calling_config
// holds the mode enum (AUTO / ANY / NONE) plus an optional allowedFunctionNames
// list — used by the ANY mode to constrain which tool the model can pick. For
// the "force a specific tool by name" case we emit ANY + a single-element
// allowedFunctionNames list (Gemini has no TOOL-singular mode like Anthropic).
type apiToolConfig struct {
	FunctionCallingConfig apiFunctionCallingConfig `json:"functionCallingConfig"`
}

type apiFunctionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// toAPIToolConfig maps llm.ToolChoice to Gemini's tool_config shape.
//
// Keyword mapping:
//
//	llm.ToolChoiceAuto → mode=AUTO        (Gemini default)
//	llm.ToolChoiceAny  → mode=ANY         (must call some tool)
//	llm.ToolChoiceNone → mode=NONE        (disable tools for this turn)
//	llm.ToolChoiceTool → mode=ANY + allowedFunctionNames=[Name]
//	                     (Gemini has no dedicated "force this exact tool"
//	                      mode; the closest semantic is ANY with a 1-element
//	                      allowlist, which prevents the model from picking
//	                      any other tool).
func toAPIToolConfig(t *llm.ToolChoice) (*apiToolConfig, error) {
	if t == nil {
		return nil, nil
	}
	switch t.Type {
	case llm.ToolChoiceAuto:
		return &apiToolConfig{FunctionCallingConfig: apiFunctionCallingConfig{Mode: "AUTO"}}, nil
	case llm.ToolChoiceAny:
		return &apiToolConfig{FunctionCallingConfig: apiFunctionCallingConfig{Mode: "ANY"}}, nil
	case llm.ToolChoiceNone:
		return &apiToolConfig{FunctionCallingConfig: apiFunctionCallingConfig{Mode: "NONE"}}, nil
	case llm.ToolChoiceTool:
		if t.Name == "" {
			return nil, fmt.Errorf("tool_choice: Type=Tool requires Name")
		}
		return &apiToolConfig{FunctionCallingConfig: apiFunctionCallingConfig{
			Mode:                 "ANY",
			AllowedFunctionNames: []string{t.Name},
		}}, nil
	default:
		return nil, fmt.Errorf("tool_choice: unknown Type %q", t.Type)
	}
}

// apiContent is one turn on the wire. Role is "user" or "model" — Gemini
// has no separate "tool" role; tool results live in user turns with
// functionResponse parts.
type apiContent struct {
	Role  string    `json:"role"`
	Parts []apiPart `json:"parts,omitempty"`
}

// apiSystem is the systemInstruction shape: a content with no role.
type apiSystem struct {
	Parts []apiPart `json:"parts"`
}

// apiPart is the discriminated union of part types Gemini accepts in
// a content. We use one struct with omitempty across the variant
// fields rather than three separate types because parts share a
// uniform JSON shape (no explicit "type" tag — variant is implied by
// which field is non-zero).
type apiPart struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *apiBlob          `json:"inlineData,omitempty"`
	FileData         *apiFileData      `json:"fileData,omitempty"`
	VideoMetadata    *apiVideoMetadata `json:"videoMetadata,omitempty"`
	FunctionCall     *apiFunctionCall  `json:"functionCall,omitempty"`
	FunctionResponse *apiFunctionResp  `json:"functionResponse,omitempty"`
	// Thought marks a thinking-only part on the response side; we never
	// emit thought parts on outgoing messages (Gemini round-trips
	// thoughts via server-side state, not by replaying them in
	// contents).
	Thought bool `json:"thought,omitempty"`
}

type apiBlob struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64-encoded
}

type apiFileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

// apiVideoMetadata is Gemini's per-part clipping + frame-rate override.
// All fields are independently optional; the server applies defaults
// for any omitted field.
type apiVideoMetadata struct {
	StartOffset string  `json:"startOffset,omitempty"` // duration like "3.5s" / "1m30s"
	EndOffset   string  `json:"endOffset,omitempty"`
	FPS         float64 `json:"fps,omitempty"`
}

// apiFunctionCall is the on-wire shape of an assistant-issued tool call.
// Id is Gemini 3+ specific: the model emits a unique id per call so that
// parallel calls to the same function can be disambiguated when the
// response comes back. Gemini 2.x doesn't emit Id; we just leave it
// empty in that case and fall back to matching by Name.
type apiFunctionCall struct {
	Id   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

// apiFunctionResp is the on-wire shape of a tool result going back to
// the model. Name MUST be the function name (matching the original
// functionDeclaration) — not a tool-call id. Id, when present, echoes
// the id from the originating functionCall so Gemini 3+ can route
// parallel responses correctly.
type apiFunctionResp struct {
	Id       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type apiTool struct {
	FunctionDeclarations []apiFunctionDecl `json:"functionDeclarations"`
}

type apiFunctionDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// generationConfig collects all "tunables" Gemini groups together —
// temperature, max output tokens, stop sequences, thinking config.
type generationConfig struct {
	Temperature     *float64     `json:"temperature,omitempty"`
	MaxOutputTokens int          `json:"maxOutputTokens,omitempty"`
	StopSequences   []string     `json:"stopSequences,omitempty"`
	ThinkingConfig  *thinkingCfg `json:"thinkingConfig,omitempty"`
}

type thinkingCfg struct {
	// ThinkingBudget is in tokens. -1 = dynamic (server chooses), 0 =
	// disable thinking. Gemini 2.5 models think by default; setting to
	// 0 is the only way to fully disable.
	ThinkingBudget  int  `json:"thinkingBudget"`
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
}

// buildRequestBody serializes a llm.Request into Gemini's wire format.
// Tool results in RoleTool messages get folded into the prior user
// turn as functionResponse parts (Gemini has no separate tool role).
func buildRequestBody(req llm.Request) (io.Reader, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	body := requestBody{}

	if req.System != "" {
		body.SystemInstruction = &apiSystem{
			Parts: []apiPart{{Text: req.System}},
		}
	}

	// Generation config — only emit when non-default to keep the wire
	// body tight.
	gc := &generationConfig{
		Temperature:     req.Temperature,
		MaxOutputTokens: req.MaxTokens,
		StopSequences:   req.StopReasons,
	}
	if req.Thinking != nil {
		gc.ThinkingConfig = &thinkingCfg{
			ThinkingBudget:  req.Thinking.BudgetTokens,
			IncludeThoughts: true,
		}
	}
	if hasGenConfig(gc) {
		body.GenerationConfig = gc
	}

	if len(req.Tools) > 0 {
		decls := make([]apiFunctionDecl, 0, len(req.Tools))
		for _, t := range req.Tools {
			decls = append(decls, apiFunctionDecl{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			})
		}
		body.Tools = []apiTool{{FunctionDeclarations: decls}}
	}

	tc, err := toAPIToolConfig(req.ToolChoice)
	if err != nil {
		return nil, err
	}
	body.ToolConfig = tc

	// Pre-walk to build a tool-call-id -> function-name index. Gemini's
	// functionResponse.name MUST be the function name (matching the
	// original functionDeclaration); pi-llm-go's ToolResultBlock only
	// carries ToolCallID, so we look the name up from the originating
	// assistant turn's ToolCallBlock. Without this, parallel calls to
	// the same tool would still produce correct names, and unknown
	// ids would emit name="" which Gemini rejects server-side.
	toolNameByID := map[string]string{}
	for _, m := range req.Messages {
		if m.Role != llm.RoleAssistant {
			continue
		}
		for _, b := range m.Content {
			if tc, ok := b.(llm.ToolCallBlock); ok && tc.ID != "" {
				toolNameByID[tc.ID] = tc.Name
			}
		}
	}

	// Walk the messages, folding tool-result messages into the prior
	// user turn's parts. This sustains Gemini's expectation that
	// functionResponse parts share a role-user turn with whatever text
	// accompanies them.
	for _, m := range req.Messages {
		converted, err := convertOutgoingMessage(m, toolNameByID)
		if err != nil {
			return nil, fmt.Errorf("convert message: %w", err)
		}
		if m.Role == llm.RoleTool && len(body.Contents) > 0 {
			// Append the functionResponse parts to the prior user turn
			// (which Gemini canonically created when the assistant's
			// functionCall was preceded by a user message).
			last := &body.Contents[len(body.Contents)-1]
			if last.Role == "user" {
				last.Parts = append(last.Parts, converted.Parts...)
				continue
			}
		}
		body.Contents = append(body.Contents, converted)
	}

	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(body); err != nil {
		return nil, fmt.Errorf("encode body: %w", err)
	}
	return buf, nil
}

func hasGenConfig(g *generationConfig) bool {
	return g.Temperature != nil ||
		g.MaxOutputTokens > 0 ||
		len(g.StopSequences) > 0 ||
		g.ThinkingConfig != nil
}

// convertOutgoingMessage maps a llm.Message into Gemini's apiContent
// shape. ImageBlock + VideoBlock are user-role-only (rejected
// otherwise — same as the OpenAI / Anthropic boundary).
//
// toolNameByID resolves ToolResultBlock.ToolCallID to the original
// function name so functionResponse.name matches Gemini's contract.
// Built by the caller via a pre-walk of req.Messages.
func convertOutgoingMessage(m llm.Message, toolNameByID map[string]string) (apiContent, error) {
	// Role-guard: media blocks are user-only.
	if m.Role != llm.RoleUser {
		for _, b := range m.Content {
			switch b.(type) {
			case llm.ImageBlock:
				return apiContent{}, fmt.Errorf("ImageBlock is only valid on user-role messages (got role %q)", m.Role)
			case llm.VideoBlock:
				return apiContent{}, fmt.Errorf("VideoBlock is only valid on user-role messages (got role %q)", m.Role)
			}
		}
	}

	role := geminiRole(m.Role)
	out := apiContent{Role: role}

	for _, block := range m.Content {
		part, drop, err := convertOutgoingBlock(block, toolNameByID)
		if err != nil {
			return apiContent{}, err
		}
		if drop {
			// e.g. ThinkingBlock on an outgoing message — Gemini doesn't
			// accept thought parts as input; emitting an empty apiPart
			// would put "{}" on the wire and the server would reject.
			continue
		}
		out.Parts = append(out.Parts, part)
	}
	return out, nil
}

// geminiRole maps pi-llm-go's Role enum onto Gemini's wire-level
// "user" | "model" two-value alphabet. RoleTool messages will be
// folded into a user turn upstream; here we still emit "user" for
// safety in case the fold-into-prior-turn shortcut doesn't fire.
func geminiRole(r llm.Role) string {
	switch r {
	case llm.RoleAssistant:
		return "model"
	default:
		return "user"
	}
}

// convertOutgoingBlock translates one llm.Block into a Gemini apiPart.
// Returns (part, drop, err). drop=true means the block has no wire
// representation (e.g. ThinkingBlock is never replayed outbound) and
// the caller should skip the empty result instead of appending it.
func convertOutgoingBlock(b llm.Block, toolNameByID map[string]string) (apiPart, bool, error) {
	switch v := b.(type) {
	case llm.TextBlock:
		return apiPart{Text: v.Text}, false, nil

	case llm.ThinkingBlock:
		// Don't replay thinking on outgoing messages — Gemini doesn't
		// accept thought parts as input, and round-tripping them via
		// server-side state would require us to track a session id we
		// don't expose at v0.4.0. Drop on send; the caller filters.
		return apiPart{}, true, nil

	case llm.ToolCallBlock:
		args := v.Arguments
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		// v.ID carries Gemini 3's wire-level id when round-tripping a
		// prior assistant turn back to the model. On Gemini 2.x it's
		// pi-llm-go's synthesized fallback (the function name) which
		// the server tolerates.
		return apiPart{
			FunctionCall: &apiFunctionCall{
				Id:   v.ID,
				Name: v.Name,
				Args: args,
			},
		}, false, nil

	case llm.ToolResultBlock:
		// Gemini expects functionResponse.response to be a JSON object,
		// not a free-form string. Wrap the content under a single
		// "result" key so the wire stays valid regardless of whether
		// the caller's tool emits structured or free-form output. The
		// model receives `{"result": "<content>"}`.
		respObj := map[string]string{"result": v.Content}
		respBytes, err := json.Marshal(respObj)
		if err != nil {
			return apiPart{}, false, fmt.Errorf("marshal tool result: %w", err)
		}
		// functionResponse.name MUST match a declared function name.
		// Look it up from the assistant turn that issued the original
		// call (toolNameByID built by buildRequestBody's pre-walk).
		// If unresolvable, emit a clear error rather than letting the
		// server reject the request opaquely.
		fnName := toolNameByID[v.ToolCallID]
		if fnName == "" {
			return apiPart{}, false, fmt.Errorf(
				"gemini: ToolResultBlock.ToolCallID %q has no matching ToolCallBlock in the prior assistant turn; functionResponse.name cannot be resolved",
				v.ToolCallID,
			)
		}
		return apiPart{
			FunctionResponse: &apiFunctionResp{
				Id:       v.ToolCallID, // echoes Gemini 3's id; harmless empty on 2.x
				Name:     fnName,
				Response: respBytes,
			},
		}, false, nil

	case llm.ImageBlock:
		if err := v.Validate(); err != nil {
			return apiPart{}, false, fmt.Errorf("gemini: %w", err)
		}
		return apiPart{
			InlineData: &apiBlob{
				MimeType: v.MimeType,
				Data:     v.Data,
			},
		}, false, nil

	case llm.VideoBlock:
		if err := v.Validate(); err != nil {
			return apiPart{}, false, fmt.Errorf("gemini: %w", err)
		}
		part := apiPart{}
		switch {
		case v.Data != "":
			part.InlineData = &apiBlob{
				MimeType: v.MimeType,
				Data:     v.Data,
			}
		case v.URI != "":
			part.FileData = &apiFileData{
				MimeType: v.MimeType, // optional when URI is set; server infers
				FileURI:  v.URI,
			}
		}
		if v.StartOffset != nil || v.EndOffset != nil || v.FPS != nil {
			meta := &apiVideoMetadata{}
			if v.StartOffset != nil {
				meta.StartOffset = formatDuration(*v.StartOffset)
			}
			if v.EndOffset != nil {
				meta.EndOffset = formatDuration(*v.EndOffset)
			}
			if v.FPS != nil {
				meta.FPS = *v.FPS
			}
			part.VideoMetadata = meta
		}
		return part, false, nil

	default:
		return apiPart{}, false, fmt.Errorf("unsupported block type %T", b)
	}
}

// formatDuration emits a Gemini-compatible duration string. Gemini's
// videoMetadata field accepts the suffix forms Go's
// time.Duration.String produces — "3.5s", "1m30s", "500ms" etc. We
// delegate rather than reformat to avoid accidentally narrowing the
// supported representation.
func formatDuration(d time.Duration) string {
	return d.String()
}
