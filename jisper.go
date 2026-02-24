// Package main provides the Jisper CLI for model-driven code edits.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/pterm/pterm"
	"go.yaml.in/yaml/v4"
)

const (
	DefaultPromptFile             = "prompt.yaml"
	DefaultTemplatePromptFile     = "default_prompt.yaml"
	DefaultAPIKeyEnvVar           = "OPENAI_API_KEY"
	DefaultURL                    = "https://api.openai.com/v1/chat/completions"
	DefaultFallbackInputUSDPer1M  = 5.0
	DefaultFallbackOutputUSDPer1M = 15.0
	DefaultMaxRetries             = 3
	DefaultRetryInitialDelayMs    = 1000
	DefaultRetryMaxDelayMs        = 30000
)

var DefaultOutputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"edit": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"explanation": map[string]any{
					"type":        "string",
					"description": "An short 1-2 sentence explanation of the changes you are making",
				},
				"commit_message": map[string]any{
					"type":        "string",
					"description": "The commit message to use for the changes you are making",
				},
				"replacements": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"filename": map[string]any{
								"type":        "string",
								"description": "The file in which to apply the edit",
							},
							"old_string": map[string]any{
								"type":        "string",
								"description": "The old string to replace in the file",
							},
							"new_string": map[string]any{
								"type":        "string",
								"description": "The new string to add",
							},
						},
						"required":             []string{"filename", "old_string", "new_string"},
						"additionalProperties": false,
					},
				},
			},
			"required":             []string{"explanation", "commit_message", "replacements"},
			"additionalProperties": false,
		},
	},
	"required": []string{"edit"},
}

func resolveOnePath(v string, baseDir string) []string {
	s := strings.TrimSpace(v)
	if s == "" {
		return []string{}
	}
	p := filepath.Join(baseDir, s)
	st, err := os.Stat(p)
	if err == nil && st.IsDir() {
		ents, _ := os.ReadDir(p)
		out := make([]string, 0)
		for _, e := range ents {
			if !e.IsDir() {
				out = append(out, filepath.Join(s, e.Name()))
			}
		}
		sort.Strings(out)
		return out
	}
	if err == nil && !st.IsDir() {
		return []string{s}
	}
	matches, _ := filepath.Glob(p)
	out := make([]string, 0)
	for _, m := range matches {
		if mst, _ := os.Stat(m); mst != nil && !mst.IsDir() {
			out = append(out, toRel(baseDir, m))
		}
	}
	sort.Strings(out)
	return out
}

func resolvePathsAndGlobs(values []string, baseDir string) []string {
	return dedupeKeepOrder(flatMapStr(values, func(v string) []string {
		return resolveOnePath(v, baseDir)
	}))
}

type Replacement struct {
	Filename  string `json:"filename"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

type Edit struct {
	Explanation   string        `json:"explanation"`
	CommitMessage string        `json:"commit_message"`
	Replacements  []Replacement `json:"replacements"`
}

type ModelResponse struct {
	Edit Edit `json:"edit"`
}

type IncludedFiles struct {
	FullFiles            []string
	StructuralLevelFiles []string
	InputLevelFiles      []string
	SourceFiles          []string
}

type Pos struct {
	Filename string `json:"Filename"`
	Offset   int    `json:"Offset"`
	Line     int    `json:"Line"`
	Column   int    `json:"Column"`
}

type Issue struct {
	Pos        Pos    `json:"Pos"`
	Text       string `json:"Text"`
	FromLinter string `json:"FromLinter"`
}

type IssuesFile struct {
	Issues []Issue `json:"Issues"`
}

type Usage struct {
	PromptTokens     *int `json:"prompt_tokens"`
	CompletionTokens *int `json:"completion_tokens"`
	TotalTokens      *int `json:"total_tokens"`
}

type Prices struct {
	InUSDPer1M  float64
	OutUSDPer1M float64
}

var ModelPricesUSDPer1M = map[string]Prices{
	"gpt-5.2":                   {InUSDPer1M: 5.0, OutUSDPer1M: 15.0},
	"gpt-5-mini":                {InUSDPer1M: 1.0, OutUSDPer1M: 3.0},
	"qwen/qwen3-coder:exacto":   {InUSDPer1M: 0.22, OutUSDPer1M: 1.8},
	"moonshotai/kimi-k2.5":      {InUSDPer1M: 0.25, OutUSDPer1M: 2.25},
	"z-ai/glm-5":                {InUSDPer1M: 1.0, OutUSDPer1M: 3.2},
	"openai/gpt-oss-120b:nitro": {InUSDPer1M: 0.35, OutUSDPer1M: 0.95},
	"minimax/minimax-m2.5":      {InUSDPer1M: 0.30, OutUSDPer1M: 1.10},
}

func responseFormatFromConfig(config map[string]any) map[string]any {
	if custom, ok := config["output_schema"].(map[string]any); ok {
		return map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"strict": true, "schema": custom},
		}
	}
	return map[string]any{
		"type":        "json_schema",
		"json_schema": map[string]any{"strict": true, "schema": DefaultOutputSchema},
	}
}

func getModelCode(config map[string]any) string {
	model, ok := asNonEmptyStr(config["model"])
	if !ok {
		fmt.Fprintln(os.Stderr, "Missing required config key: 'model'. Add a model identifier to your prompt config, e.g.:")
		fmt.Fprintln(os.Stderr, "  model: gpt-5.2")
		os.Exit(2)
	}
	return model
}

func resolveIncludedFiles(config map[string]any, baseDir string) IncludedFiles {
	if os.Getenv("DEBUG_JISPER") != "" {
		fmt.Printf("DEBUG: Resolving files in baseDir: %s\n", baseDir)
	}
	fullRaw := asListOfNonEmptyStr(config["full_files"])
	structRaw := asListOfNonEmptyStr(config["structural_level_files"])
	inputRaw := asListOfNonEmptyStr(config["input_level_files"])

	fullFiles := resolvePathsAndGlobs(fullRaw, baseDir)
	structFiles := resolvePathsAndGlobs(structRaw, baseDir)
	inputFiles := resolvePathsAndGlobs(inputRaw, baseDir)

	fullSet := map[string]bool{}
	for _, f := range fullFiles {
		fullSet[f] = true
	}
	structOut := make([]string, 0, len(structFiles))
	for _, f := range structFiles {
		if !fullSet[f] {
			structOut = append(structOut, f)
		}
	}
	structSet := map[string]bool{}
	for _, f := range structOut {
		structSet[f] = true
	}
	inputOut := make([]string, 0, len(inputFiles))
	for _, f := range inputFiles {
		if !fullSet[f] && !structSet[f] {
			inputOut = append(inputOut, f)
		}
	}

	sourceFiles := dedupeKeepOrder(append(append([]string{}, fullFiles...), append(structOut, inputOut...)...))
	return IncludedFiles{
		FullFiles:            fullFiles,
		StructuralLevelFiles: structOut,
		InputLevelFiles:      inputOut,
		SourceFiles:          sourceFiles,
	}
}

func loadPromptFile(path string) (map[string]any, error) {
	ext := strings.ToLower(filepath.Ext(path))
	b, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{}, err
	}
	var cfg map[string]any
	if ext == ".yaml" || ext == ".yml" {
		err = yaml.Unmarshal(b, &cfg)
	} else {
		err = json.Unmarshal(b, &cfg)
	}
	if err != nil {
		return map[string]any{}, err
	}
	if cfg == nil {
		cfg = map[string]any{}
	}
	return cfg, nil
}

func readAndConcatenateFiles(fileList []string, baseDir string, ctx map[string]any) string {
	var res []string
	for _, f := range fileList {
		txt, ok := readFileContent(baseDir, f)
		if !ok {
			continue
		}
		if ctx != nil {
			txt = render(txt, ctx)
		}
		res = append(res, fmt.Sprintf("--- FILENAME: %s ---\n%s", f, txt))
	}
	return strings.Join(res, "\n\n")
}

func extractFileSummaryYAML(text string) (map[string]any, bool) {
	start := strings.Index(text, "[FILE SUMMARY]")
	if start < 0 {
		return nil, false
	}
	end := strings.Index(text[start:], "[/FILE SUMMARY]")
	if end < 0 {
		return nil, false
	}
	inner := strings.TrimSpace(text[start+len("[FILE SUMMARY]") : start+end])
	if inner == "" {
		return nil, false
	}
	var loaded map[string]any
	if err := yaml.Unmarshal([]byte(inner), &loaded); err != nil {
		return nil, false
	}
	return loaded, true
}

func selectContextFields(summary map[string]any, intentOnly bool) (map[string]any, bool) {
	ctxAny, ok := summary["context"]
	if !ok {
		return nil, false
	}
	ctx, ok := ctxAny.(map[string]any)
	if !ok {
		return nil, false
	}
	intent := ctx["INTENT"]
	structural := ctx["STRUCTURAL"]
	if intentOnly {
		if intent == nil {
			return nil, false
		}
		return map[string]any{"context": map[string]any{"INTENT": intent}}, true
	}
	out := map[string]any{}
	if intent != nil {
		out["INTENT"] = intent
	}
	if structural != nil {
		out["STRUCTURAL"] = structural
	}
	if len(out) == 0 {
		return nil, false
	}
	return map[string]any{"context": out}, true
}

func buildFileSummariesSection(files []string, baseDir string, intentOnly bool, _ map[string]any) string {
	parts := []string{}
	for _, filename := range files {
		txt, ok := readFileContent(baseDir, filename)
		if !ok {
			continue
		}
		summary, ok := extractFileSummaryYAML(txt)
		if !ok {
			continue
		}
		selected, ok := selectContextFields(summary, intentOnly)
		if !ok {
			continue
		}
		dumped, _ := yaml.Marshal(selected)
		if len(dumped) > 0 {
			parts = append(parts, fmt.Sprintf("--- FILENAME: %s ---\n%s", filename, strings.TrimSpace(string(dumped))))
		}
	}
	return strings.Join(parts, "\n\n")
}

func buildSourceMaterial(promptConfig map[string]any, baseDir string, jinjaContext map[string]any) string {
	includes := resolveIncludedFiles(promptConfig, baseDir)

	fullText := strings.TrimSpace(readAndConcatenateFiles(includes.FullFiles, baseDir, jinjaContext))
	structText := strings.TrimSpace(buildFileSummariesSection(includes.StructuralLevelFiles, baseDir, false, jinjaContext))
	inputText := strings.TrimSpace(buildFileSummariesSection(includes.InputLevelFiles, baseDir, true, jinjaContext))

	parts := []string{}
	if fullText != "" {
		parts = append(parts, fullText)
	}
	if structText != "" {
		parts = append(parts, "STRUCTURAL_LEVEL_FILES (SUMMARIES ONLY):\n"+structText)
	}
	if inputText != "" {
		parts = append(parts, "INPUT_LEVEL_FILES (SUMMARIES ONLY):\n"+inputText)
	}
	return strings.TrimSpace(strings.Join(parts, "\n\n"))
}

func stripJSONCodeFence(s string) string {
	t := strings.TrimSpace(s)
	if !strings.HasPrefix(t, "```") {
		return s
	}
	lines := strings.Split(t, "\n")
	if len(lines) < 2 {
		return s
	}
	first := strings.ToLower(strings.TrimSpace(lines[0]))
	if first != "```json" && first != "```" {
		return s
	}
	if strings.TrimSpace(lines[len(lines)-1]) != "```" {
		return s
	}
	inner := strings.Join(lines[1:len(lines)-1], "\n")
	inner = strings.TrimSpace(inner)
	if inner == "" {
		return ""
	}
	return inner + "\n"
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type payload struct {
	Model          string         `json:"model"`
	Messages       []message      `json:"messages"`
	ResponseFormat map[string]any `json:"response_format,omitempty"`
}

func buildPayload(
	promptConfig map[string]any, sourceText string, routine string, taskOverride string) (payload, string) {
	systemInstruction := "You are a helpful assistant."
	if s, ok := asNonEmptyStr(promptConfig["system_instruction"]); ok {
		systemInstruction = s
	}
	systemPromptForCtx := resolveSystemPrompt(promptConfig)

	userTask := taskOverride
	if userTask == "" {
		userTask = resolveUserTask(promptConfig, routine)
	}
	modelCode := getModelCode(promptConfig)

	ctx := buildJinjaContext(promptConfig, sourceText, userTask, systemPromptForCtx)
	renderedSystem := render(systemPromptForCtx, ctx)
	renderedTask := render(userTask, ctx)

	promptContent := fmt.Sprintf("SYSTEM PROMPT:\n%s\n\nTASK:\n%s\n\nSOURCE MATERIAL:\n%s",
		renderedSystem, renderedTask, sourceText)
	pl := payload{
		Model: modelCode,
		Messages: []message{
			{Role: "system", Content: systemInstruction},
			{Role: "user", Content: promptContent},
		},
	}

	pl.ResponseFormat = responseFormatFromConfig(promptConfig)
	return pl, promptContent
}

func extractUsageFromAPIResponse(apiJSON map[string]any, headers http.Header) Usage {
	usageAny := apiJSON["usage"]
	usageMap, _ := usageAny.(map[string]any)
	getHeader := func(k string) string { return headers.Get(k) }
	getToken := func(jsonKey string, headerKey string) *int {
		if usageMap != nil {
			if v, ok := usageMap[jsonKey]; ok {
				if n := coerceInt(v); n != nil {
					return n
				}
			}
		}
		hv := strings.TrimSpace(getHeader(headerKey))
		if hv == "" {
			return nil
		}
		return coerceInt(hv)
	}

	prompt := getToken("prompt_tokens", "x-openai-prompt-tokens")
	completion := getToken("completion_tokens", "x-openai-completion-tokens")
	total := getToken("total_tokens", "x-openai-total-tokens")
	if total == nil && prompt != nil && completion != nil {
		t := *prompt + *completion
		total = &t
	}
	return Usage{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: total}
}

func applyOneReplacement(original string, oldString string, newString string) (string, string, bool) {
	replaceIf := func(haystack string, needle string) (string, bool) {
		if needle == "" {
			return "", false
		}
		if !strings.Contains(haystack, needle) {
			return "", false
		}
		return strings.ReplaceAll(haystack, needle, newString), true
	}

	updated, ok := replaceIf(original, oldString)
	if ok {
		return updated, oldString, true
	}

	trimmedOld := strings.TrimSpace(oldString)
	if trimmedOld != "" && trimmedOld != oldString {
		updated, ok = replaceIf(original, trimmedOld)
		if ok {
			return updated, trimmedOld, true
		}
	}

	strippedOriginal := strings.TrimSpace(original)
	trimmedOld = strings.TrimSpace(oldString)
	if strippedOriginal != "" && trimmedOld != "" && strings.Contains(strippedOriginal, trimmedOld) {
		leading := original[:len(original)-len(strings.TrimLeft(original, " \t\n\r"))]
		trailing := original[len(strings.TrimRight(original, " \t\n\r")):]
		replacedCore := strings.ReplaceAll(strippedOriginal, trimmedOld, newString)
		return leading + replacedCore + trailing, trimmedOld, true
	}

	return "", "", false
}

func performCreateFile(baseDir, filename, newString, language string) (string, bool) {
	targetPath := filepath.Join(baseDir, filename)
	_ = os.MkdirAll(filepath.Dir(targetPath), 0o755)
	fmt.Printf("\n\x1b[1m%s\x1b[0m\n", filename)
	printNumberedCombinedDiff("", newString, filename, language)
	if err := os.WriteFile(targetPath, []byte(newString), 0o644); err != nil {
		pterm.Error.Printfln("Failed to create file %s: %v", targetPath, err)
		return "", false
	}
	return targetPath, true
}

func applyReplacements(
	repls []Replacement, baseDir string, language string,
	configPath string, allowedFiles []string) []string {
	changed := []string{}
	for i, r := range repls {
		filename := strings.TrimSpace(r.Filename)
		if os.Getenv("DEBUG_JISPER") != "" {
			fmt.Printf("DEBUG: Attempting replacement #%d in %s\n", i, filename)
		}
		if filename == "" || !isFileAllowed(filename, allowedFiles) {
			continue
		}
		targetPath := filepath.Join(baseDir, filename)
		original, ok := readFileContent(baseDir, filename)
		if !ok && strings.TrimSpace(r.OldString) == "" {
			if p, ok := performCreateFile(baseDir, filename, r.NewString, language); ok {
				changed = append(changed, p)
			}
			continue
		}
		if !ok {
			continue
		}
		updated, _, applied := applyOneReplacement(original, r.OldString, r.NewString)
		if !applied {
			writeFailedOldStringToConfig(configPath, r.OldString)
			continue
		}
		if updated == original {
			continue
		}
		fmt.Printf("\x1b[1m%s\x1b[0m\n", filename)
		printNumberedCombinedDiff(original, updated, filename, language)
		if err := os.WriteFile(targetPath, []byte(updated), 0o644); err != nil {
			continue
		}
		changed = append(changed, targetPath)
	}
	return changed
}

func stripANSI(s string) string {
	result := []byte{}
	inEscape := false
	for i := 0; i < len(s); i++ {
		b := s[i]
		if inEscape {
			if b == 'm' {
				inEscape = false
			}
			continue
		}
		if b == '\x1b' && i+1 < len(s) && s[i+1] == '[' {
			inEscape = true
			i++
			continue
		}
		result = append(result, b)
	}
	return string(result)
}

func collapseCRUpdates(s string) string {
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		parts := strings.Split(line, "\r")
		if len(parts) > 0 {
			out = append(out, parts[len(parts)-1])
		}
	}
	return strings.Join(out, "\n")
}

func sanitizeOutput(raw string) string {
	raw = strings.ReplaceAll(raw, "\r\n", "\n")
	raw = collapseCRUpdates(raw)
	raw = stripANSI(raw)
	lines := filterNonEmpty(strings.Split(raw, "\n"))
	raw = strings.Join(lines, "\n")
	if raw != "" && !strings.HasSuffix(raw, "\n") {
		raw += "\n"
	}
	return raw
}

func filterNonEmpty(lines []string) []string {
	out := make([]string, 0, len(lines))
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}

func updatePromptConfigWithBuildResults(path string, stdout, stderr string, code int) {
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var node yaml.Node
	if err := yaml.Unmarshal(b, &node); err != nil || len(node.Content) == 0 {
		return
	}
	root := node.Content[0]
	idx := -1
	for i := 0; i < len(root.Content); i += 2 {
		if root.Content[i].Value == "build" {
			idx = i + 2
			break
		}
	}
	setOrAddKey := func(key, value string, insertAt *int) {
		for i := 0; i < len(root.Content); i += 2 {
			if root.Content[i].Value == key {
				root.Content[i+1].Value = value
				root.Content[i+1].Style = yaml.LiteralStyle
				return
			}
		}
		nk := &yaml.Node{Kind: yaml.ScalarNode, Value: key}
		nv := &yaml.Node{Kind: yaml.ScalarNode, Value: value, Style: yaml.LiteralStyle}
		if *insertAt >= 0 {
			root.Content = append(root.Content[:*insertAt], append([]*yaml.Node{nk, nv}, root.Content[*insertAt:]...)...)
			*insertAt += 2
		} else {
			root.Content = append(root.Content, nk, nv)
		}
	}
	if stdout != "" {
		setOrAddKey("build_stdout", stdout, &idx)
	}
	if stderr != "" {
		setOrAddKey("build_stderr", stderr, &idx)
	}
	if code == 0 {
		setOrAddKey("success", "true", &idx)
	} else {
		setOrAddKey("error", fmt.Sprintf("build failed (%d)", code), &idx)
	}
	out, _ := yaml.Marshal(&node)
	_ = os.WriteFile(path, out, 0o644)
}

func runBuildStep(config map[string]any, configPath string) {
	cmdStr, ok := asNonEmptyStr(config["build"])
	if !ok {
		return
	}
	fmt.Printf("\nBuild: %s\n\n", cmdStr)
	cmd := exec.Command("/bin/sh", "-c", cmdStr)
	var outB, errB bytes.Buffer
	cmd.Stdout, cmd.Stderr = io.MultiWriter(os.Stdout, &outB), io.MultiWriter(os.Stderr, &errB)
	err := cmd.Run()
	code := 0
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		} else {
			code = 1
		}
	}
	updatePromptConfigWithBuildResults(configPath,
		sanitizeOutput(outB.String()),
		sanitizeOutput(errB.String()),
		code)
}

func calculateBackoff(attempt, initialDelayMs, maxDelayMs int) int {
	delay := initialDelayMs
	for i := 1; i < attempt; i++ {
		delay *= 2
		if delay > maxDelayMs {
			delay = maxDelayMs
			break
		}
	}
	return delay
}

func callOpenAICompatibleWithRetry(
	endpointURL string,
	apiKey string,
	pl payload,
	config map[string]any,
) (map[string]any, http.Header, error) {
	maxRetries := DefaultMaxRetries
	if r, ok := config["max_retries"].(float64); ok {
		maxRetries = int(r)
	}
	initialDelayMs := DefaultRetryInitialDelayMs
	if d, ok := config["retry_initial_delay_ms"].(float64); ok {
		initialDelayMs = int(d)
	}
	maxDelayMs := DefaultRetryMaxDelayMs
	if d, ok := config["retry_max_delay_ms"].(float64); ok {
		maxDelayMs = int(d)
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			delay := calculateBackoff(attempt, initialDelayMs, maxDelayMs)
			pterm.Info.Printfln("Retry attempt %d/%d after %dms...", attempt, maxRetries, delay)
			time.Sleep(time.Duration(delay) * time.Millisecond)
		}

		apiJSON, headers, err := callOpenAICompatible(endpointURL, apiKey, pl)
		if err == nil {
			return apiJSON, headers, nil
		}

		lastErr = err
		shouldRetry := false
		retryAfter := 0

		if resp, ok := err.(retryableError); ok {
			shouldRetry = resp.shouldRetry()
			retryAfter = resp.retryAfter()
		}

		if !shouldRetry {
			return nil, nil, err
		}

		if retryAfter > 0 && attempt < maxRetries {
			delay := retryAfter
			if delay > maxDelayMs {
				delay = maxDelayMs
			}
			pterm.Info.Printfln("Rate limited, waiting %dms before retry...", delay)
			time.Sleep(time.Duration(delay) * time.Millisecond)
		}
	}

	return nil, nil, fmt.Errorf("all %d retry attempts failed: %w", maxRetries+1, lastErr)
}

type retryableError struct {
	statusCode        int
	retryAfterSeconds int
	message           string
}

func (e retryableError) Error() string { return e.message }
func (e retryableError) shouldRetry() bool {
	return e.statusCode == 429 || e.statusCode >= 500
}
func (e retryableError) retryAfter() int { return e.retryAfterSeconds }

func callOpenAICompatible(endpointURL string, apiKey string, pl payload) (map[string]any, http.Header, error) {
	b, err := json.Marshal(pl)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}
	req, err := http.NewRequest("POST", endpointURL, bytes.NewReader(b))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create HTTP request to %s: %w", endpointURL, err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("HTTP request to %s failed: %w", endpointURL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.Header, fmt.Errorf("failed to read response body from %s: %w", endpointURL, err)
	}
	if resp.StatusCode != 200 {
		bodyPreview := string(body)
		if len(bodyPreview) > 1000 {
			bodyPreview = bodyPreview[:1000] + "... (truncated)"
		}

		retryAfter := 0
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if n, err := strconv.Atoi(strings.TrimSpace(ra)); err == nil {
				retryAfter = n
			}
		}

		errMsg := fmt.Sprintf(
			"API request to %s returned status %d. Response body: %s",
			endpointURL, resp.StatusCode, bodyPreview)
		return nil, resp.Header, retryableError{statusCode: resp.StatusCode, retryAfterSeconds: retryAfter, message: errMsg}
	}

	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var apiJSON map[string]any
	if err := dec.Decode(&apiJSON); err != nil {
		bodyPreview := string(body)
		if len(bodyPreview) > 500 {
			bodyPreview = bodyPreview[:500] + "... (truncated)"
		}
		return nil, resp.Header, fmt.Errorf(
			"failed to parse API response as JSON from %s. Parse error: %w. Response body: %s",
			endpointURL, err, bodyPreview) // split long fmt.Errorf line
	}
	return apiJSON, resp.Header, nil
}

func parseModelResponse(apiJSON map[string]any) (ModelResponse, error) {
	choicesAny, ok := apiJSON["choices"]
	if !ok {
		return ModelResponse{}, fmt.Errorf("API response missing 'choices' field. Response keys: %v", getKeys(apiJSON))
	}
	choices, ok := choicesAny.([]any)
	if !ok {
		return ModelResponse{}, fmt.Errorf("API response 'choices' is not an array, got %T", choicesAny)
	}
	if len(choices) == 0 {
		return ModelResponse{}, fmt.Errorf("API response 'choices' array is empty")
	}
	choice0, ok := choices[0].(map[string]any)
	if !ok {
		return ModelResponse{}, fmt.Errorf("API response choices[0] is not an object, got %T", choices[0])
	}
	msgAny, ok := choice0["message"]
	if !ok {
		return ModelResponse{}, fmt.Errorf("API response choices[0] missing 'message' field. Keys: %v", getKeys(choice0))
	}
	msg, ok := msgAny.(map[string]any)
	if !ok {
		return ModelResponse{}, fmt.Errorf(
			"API response choices[0].message is not an object, got %T",
			msgAny,
		)
	}
	contentAny, ok := msg["content"]
	if !ok {
		return ModelResponse{}, fmt.Errorf(
			"API response choices[0].message missing 'content' field. Keys: %v",
			getKeys(msg),
		)
	}
	content, ok := contentAny.(string)
	if !ok {
		return ModelResponse{}, fmt.Errorf("API response choices[0].message.content is not a string, got %T", contentAny)
	}
	content = stripJSONCodeFence(content)
	var mr ModelResponse
	dec := json.NewDecoder(strings.NewReader(content))
	dec.UseNumber()
	if err := dec.Decode(&mr); err != nil {
		contentPreview := content
		if len(contentPreview) > 500 {
			contentPreview = contentPreview[:500] + "... (truncated)"
		}
		errMsg := fmt.Errorf(
			"failed to parse model response content as JSON. Parse error: %w. Content: %s",
			err, contentPreview,
		)
		return ModelResponse{}, errMsg
	}
	return mr, nil
}

func getKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func prepareRun(configPath string, routine string, task string) (map[string]any, payload, string, string, string) {
	config, err := loadPromptFile(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load prompt config from %s: %v\n", configPath, err)
		os.Exit(1)
	}
	if strings.TrimSpace(routine) != "" {
		if _, ok := resolveRoutineTask(config, routine); !ok {
			routines, _ := config["routines"].(map[string]any)
			available := []string{}
			for k := range routines {
				available = append(available, k)
			}
			sort.Strings(available)
			if len(available) == 0 {
				fmt.Fprintf(os.Stderr,
					"Routine not found: %s. No routines are defined in the config file.\n",
					routine)
			} else {
				fmt.Fprintf(os.Stderr,
					"Routine not found: %s. Available routines: %s\n",
					routine, strings.Join(available, ", "))
			}
			os.Exit(2)
		}
	}
	endpointURL, keyVar := ResolveEndpointAndAPIKey(
		config,
	)
	apiKey := GetAPIKey(endpointURL, keyVar)
	sysCtx := resolveSystemPrompt(config)
	userCtx := resolveUserTask(config, routine)
	fileCtx := buildJinjaContext(config, "", userCtx, sysCtx)
	render, _ := config["render_source_files_as_jinja"].(bool)
	var srcCtx map[string]any
	if render {
		srcCtx = fileCtx
	}
	cwd, _ := os.Getwd()
	srcMaterial := buildSourceMaterial(config, cwd, srcCtx)
	var pl payload
	var pContent string
	if strings.TrimSpace(task) != "" {
		pl, pContent = buildPayloadWithTask(config, srcMaterial, strings.TrimSpace(task))
	} else {
		pl, pContent = buildPayload(config, srcMaterial, routine, endpointURL)
	}
	return config, pl, pContent, apiKey, endpointURL
}

func buildPayloadWithTask(promptConfig map[string]any, sourceText string, task string) (payload, string) {
	return buildPayload(promptConfig, sourceText, "", task)
}

func callModel(endpointURL string, apiKey string, pl payload, config map[string]any) (ModelResponse, Usage, string) {
	modelCode := getModelCode(config)
	spinner, _ := pterm.DefaultSpinner.Start(fmt.Sprintf("Waiting for %s...", modelCode))
	apiJSON, headers, err := callOpenAICompatibleWithRetry(endpointURL, apiKey, pl, config)
	if err != nil {
		spinner.Fail(fmt.Sprintf("API call failed for model %s at %s: %v", modelCode, endpointURL, err))
		os.Exit(1)
	}
	mr, err := parseModelResponse(apiJSON)
	if err != nil {
		spinner.Fail(fmt.Sprintf("Failed to parse response from %s: %v", modelCode, err))
		os.Exit(1)
	}
	spinner.Success()
	usage := extractUsageFromAPIResponse(apiJSON, headers)
	return mr, usage, modelCode
}

func writeFailedOldStringToConfig(path string, oldString string) {
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var node yaml.Node
	if err := yaml.Unmarshal(b, &node); err != nil || len(node.Content) == 0 {
		return
	}
	root := node.Content[0]
	for i := 0; i < len(root.Content); i += 2 {
		if root.Content[i].Value == "old_string" {
			root.Content[i+1].Value = oldString
			root.Content[i+1].Style = yaml.LiteralStyle
			out, _ := yaml.Marshal(&node)
			_ = os.WriteFile(path, out, 0o644)
			return
		}
	}
	nk := &yaml.Node{Kind: yaml.ScalarNode, Value: "old_string"}
	nv := &yaml.Node{Kind: yaml.ScalarNode, Value: oldString, Style: yaml.LiteralStyle}
	root.Content = append(root.Content, nk, nv)
	out, _ := yaml.Marshal(&node)
	_ = os.WriteFile(path, out, 0o644)
}

func loadIssuesFile(path string) (IssuesFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return IssuesFile{}, fmt.Errorf("failed to read issues file %s: %w", path, err)
	}
	var issues IssuesFile
	if err := json.Unmarshal(b, &issues); err != nil {
		return IssuesFile{}, fmt.Errorf("failed to parse issues file %s: %w", path, err)
	}
	return issues, nil
}

func extractLinesAround(filename string, lineNum int, baseDir string, before, after int) (string, bool) {
	content, ok := readFileContent(baseDir, filename)
	if !ok {
		return "", false
	}
	lines := strings.Split(content, "\n")
	start := lineNum - 1 - before
	if start < 0 {
		start = 0
	}
	end := lineNum + after
	if end > len(lines) {
		end = len(lines)
	}
	var out []string
	for i := start; i < end; i++ {
		out = append(out, lines[i])
	}
	return strings.Join(out, "\n"), true
}

func getIssueContextAndTask(issue Issue, baseDir string) (string, string, bool) {
	context, ok := extractLinesAround(issue.Pos.Filename, issue.Pos.Line, baseDir, 40, 40)
	if !ok {
		return "", "", false
	}
	task := fmt.Sprintf("Fix this linting issue in %s at line %d.\n\nLinter: %s\nMessage: %s\n\nCode:\n%s",
		issue.Pos.Filename, issue.Pos.Line, issue.FromLinter, issue.Text, context)
	return context, task, true
}

func processIssue(issue Issue, config map[string]any, endpointURL, apiKey, promptPath string,
	debug, noModel bool, lang string) float64 {
	context, task, ok := getIssueContextAndTask(issue, ".")
	if !ok {
		pterm.Error.Printfln("Failed to read %s (line %d). File may not exist or line number is out of range.",
			issue.Pos.Filename, issue.Pos.Line)
		return 0
	}
	pl, content := buildPayloadWithTask(config, context, task)
	if debug {
		fmt.Printf("\n--- PROMPT ---\n%s\n--- END PROMPT ---\n", content)
	}
	if noModel {
		enc, _ := json.MarshalIndent(pl, "", "  ")
		fmt.Println(string(enc))
		return 0
	}
	mr, usage, mc := callModel(endpointURL, apiKey, pl, config)
	changed := applyReplacements(
		mr.Edit.Replacements, ".", lang, promptPath,
		[]string{issue.Pos.Filename})
	msg := strings.TrimSpace(mr.Edit.CommitMessage)
	if msg == "" {
		msg = fmt.Sprintf("Fix %s issue in %s", issue.FromLinter, issue.Pos.Filename)
	}
	pterm.Info.Printfln("Commit: %s", msg)
	prices := getModelPrices(config)
	var cost float64
	if c := estimateCostUSD(mc, usage, prices); c != nil {
		pterm.Success.Printfln("$%.4f", *c)
		cost = *c
	}
	repo := initRepoIfMissing(".")
	if len(changed) > 0 {
		stageAndCommit(repo, changed, msg)
	}
	return cost
}

func runIssues(issues IssuesFile, promptPath string, debug bool, noModel bool) {
	config, err := loadPromptFile(promptPath)
	if err != nil {
		fmt.Fprintf(
			os.Stderr,
			"Failed to load config from %s: %v\n",
			promptPath,
			err,
		)
		os.Exit(1)
	}
	endpointURL, keyVar := ResolveEndpointAndAPIKey(
		config,
	)
	apiKey := GetAPIKey(endpointURL, keyVar)
	modelCode := getModelCode(config)
	lang, _ := asNonEmptyStr(config["language"])
	var totalCost float64
	for i, issue := range issues.Issues {
		pterm.Info.Printfln(
			"Processing issue %d/%d (model: %s)",
			i+1,
			len(issues.Issues),
			modelCode,
		)

		totalCost += processIssue(
			issue,
			config,
			endpointURL,
			apiKey,
			promptPath,
			debug,
			noModel,
			lang,
		)
	}
	if totalCost > 0 {
		pterm.Info.Printfln("Total spent: $%.4f", totalCost)
	}
	runBuildStep(config, promptPath)
}

func run(
	path string, routine string, debug bool, noModel bool, task string) (ModelResponse, Usage, string, map[string]any) {
	cfg, pl, content, key, endpoint := prepareRun(path, routine, task)
	if debug {
		fmt.Printf("\n--- PROMPT ---\n%s\n--- END PROMPT ---\n", content)
	}
	if noModel {
		fmt.Println("--no-model specified")
		enc, _ := json.MarshalIndent(pl, "", "  ")
		fmt.Println(string(enc))
		os.Exit(0)
	}
	mr, usage, code := callModel(endpoint, key, pl, cfg)
	return mr, usage, code, cfg
}

func getModelPrices(config map[string]any) map[string]Prices {
	return GetModelPrices(config)
}

func estimateCostUSD(modelCode string, usage Usage, prices map[string]Prices) *float64 {
	return EstimateCostUSD(modelCode, usage, prices)
}
