package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	DefaultPromptFile            = "prompt.yaml"
	DefaultAPIKeyEnvVar          = "OPENAI_API_KEY"
	DefaultURL                   = "https://api.openai.com/v1/chat/completions"
	DefaultFallbackInputUSDPer1M = 5.0
	DefaultFallbackOutputUSDPer1M = 15.0
)

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
	"qwen/qwen3-coder:exacto":    {InUSDPer1M: 0.22, OutUSDPer1M: 1.8},
	"moonshotai/kimi-k2.5":       {InUSDPer1M: 0.25, OutUSDPer1M: 2.25},
	"z-ai/glm-5":                 {InUSDPer1M: 1.0, OutUSDPer1M: 3.2},
	"openai/gpt-oss-120b:nitro":  {InUSDPer1M: 0.35, OutUSDPer1M: 0.95},
}

func asNonEmptyStr(v any) (string, bool) {
	s, ok := v.(string)
	if !ok {
		return "", false
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false
	}
	return s, true
}

func asListOfNonEmptyStr(v any) []string {
	xs, ok := v.([]any)
	if !ok {
		ss, ok := v.([]string)
		if !ok {
			return []string{}
		}
		out := make([]string, 0, len(ss))
		for _, s := range ss {
			t := strings.TrimSpace(s)
			if t != "" {
				out = append(out, t)
			}
		}
		return out
	}
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		s, ok := asNonEmptyStr(x)
		if ok {
			out = append(out, s)
		}
	}
	return out
}

func dedupeKeepOrder(xs []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		if x == "" {
			continue
		}
		if seen[x] {
			continue
		}
		seen[x] = true
		out = append(out, x)
	}
	return out
}

func readTextOrNone(path string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return string(b), true
}

func getModelCode(config map[string]any) string {
	model, ok := asNonEmptyStr(config["model"])
	if !ok {
		fmt.Fprintln(os.Stderr, "Missing required config key: model")
		os.Exit(2)
	}
	return model
}

func resolveRoutineTask(config map[string]any, routineName string) (string, bool) {
	name := strings.TrimSpace(routineName)
	if name == "" {
		return "", false
	}
	routinesAny, ok := config["routines"]
	if !ok {
		return "", false
	}
	routines, ok := routinesAny.(map[string]any)
	if !ok {
		return "", false
	}
	taskAny, ok := routines[name]
	if !ok {
		return "", false
	}
	return asNonEmptyStr(taskAny)
}

func toRel(baseDir string, p string) string {
	rel, err := filepath.Rel(baseDir, p)
	if err != nil {
		return p
	}
	return rel
}

func resolvePathsAndGlobs(values []string, baseDir string) []string {
	one := func(v string) []string {
		s := strings.TrimSpace(v)
		if s == "" {
			return []string{}
		}
		p := filepath.Join(baseDir, s)
		st, err := os.Stat(p)
		if err == nil && st.IsDir() {
			ents, err := os.ReadDir(p)
			if err != nil {
				return []string{}
			}
			names := make([]string, 0, len(ents))
			for _, e := range ents {
				if e.IsDir() {
					continue
				}
				names = append(names, e.Name())
			}
			sort.Strings(names)
			out := make([]string, 0, len(names))
			for _, n := range names {
				out = append(out, toRel(baseDir, filepath.Join(p, n)))
			}
			return out
		}
		if err == nil && !st.IsDir() {
			return []string{toRel(baseDir, p)}
		}
		matches, _ := filepath.Glob(filepath.Join(baseDir, s))
		out := make([]string, 0, len(matches))
		for _, m := range matches {
			mst, err := os.Stat(m)
			if err != nil {
				continue
			}
			if mst.IsDir() {
				continue
			}
			out = append(out, toRel(baseDir, m))
		}
		sort.Strings(out)
		return out
	}

	out := make([]string, 0, len(values))
	for _, v := range values {
		out = append(out, one(v)...)
	}
	return dedupeKeepOrder(out)
}

func resolveIncludedFiles(config map[string]any, baseDir string) IncludedFiles {
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
	return IncludedFiles{FullFiles: fullFiles, StructuralLevelFiles: structOut, InputLevelFiles: inputOut, SourceFiles: sourceFiles}
}

func loadPromptFile(path string) (map[string]any, error) {
	ext := strings.ToLower(filepath.Ext(path))
	b, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{}, err
	}
	if ext == ".yaml" || ext == ".yml" {
		return map[string]any{}, fmt.Errorf("YAML parsing not yet implemented in Go scaffold (no third-party deps allowed)")
	}
	var cfg map[string]any
	err = json.Unmarshal(b, &cfg)
	if err != nil {
		return map[string]any{}, err
	}
	if cfg == nil {
		cfg = map[string]any{}
	}
	return cfg, nil
}

func renderJinjaTemplate(templateText string, context map[string]any) string {
	return templateText
}

func buildJinjaContext(promptConfig map[string]any, sourceText string, userTask string, systemPrompt string) map[string]any {
	ctx := map[string]any{}
	for k, v := range promptConfig {
		ctx[k] = v
	}
	ctx["source_text"] = sourceText
	ctx["task"] = userTask
	ctx["system_prompt"] = systemPrompt
	ctx["error_message"] = promptConfig["error"]
	ctx["build_stdout"] = promptConfig["build_stdout"]
	ctx["build_stderr"] = promptConfig["build_stderr"]
	ctx["success"] = promptConfig["success"]
	ctx["error"] = promptConfig["error"]
	return ctx
}

func readAndConcatenateFiles(fileList []string, baseDir string, jinjaContext map[string]any) string {
	parts := make([]string, 0, len(fileList))
	for _, filename := range fileList {
		p := filepath.Join(baseDir, filename)
		txt, ok := readTextOrNone(p)
		if !ok {
			fmt.Fprintf(os.Stderr, "Missing input file: %s\n", filename)
			continue
		}
		rendered := txt
		if jinjaContext != nil {
			rendered = renderJinjaTemplate(txt, jinjaContext)
		}
		parts = append(parts, fmt.Sprintf("--- FILENAME: %s ---\n%s", filename, rendered))
	}
	return strings.Join(parts, "\n\n")
}

func buildSourceMaterial(promptConfig map[string]any, baseDir string, jinjaContext map[string]any) string {
	includes := resolveIncludedFiles(promptConfig, baseDir)

	fullText := strings.TrimSpace(readAndConcatenateFiles(includes.FullFiles, baseDir, jinjaContext))
	structText := strings.TrimSpace(readAndConcatenateFiles(includes.StructuralLevelFiles, baseDir, jinjaContext))
	inputText := strings.TrimSpace(readAndConcatenateFiles(includes.InputLevelFiles, baseDir, jinjaContext))

	parts := []string{}
	if fullText != "" {
		parts = append(parts, fullText)
	}
	if structText != "" {
		parts = append(parts, "STRUCTURAL_LEVEL_FILES (FULL TEXT; summaries not yet implemented):\n"+structText)
	}
	if inputText != "" {
		parts = append(parts, "INPUT_LEVEL_FILES (FULL TEXT; summaries not yet implemented):\n"+inputText)
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
	Model          string           `json:"model"`
	Messages       []message        `json:"messages"`
	ResponseFormat map[string]any   `json:"response_format,omitempty"`
}

func buildPayload(promptConfig map[string]any, sourceText string, routineName string, endpointURL string) (payload, string) {
	systemInstruction := "You are a helpful assistant."
	if s, ok := asNonEmptyStr(promptConfig["system_instruction"]); ok {
		systemInstruction = s
	}
	systemPromptRaw, _ := asNonEmptyStr(promptConfig["system_prompt"])
	projectPrompt, _ := asNonEmptyStr(promptConfig["project"])
	systemPromptForCtx := systemPromptRaw
	if projectPrompt != "" {
		systemPromptForCtx = systemPromptForCtx + "\n\n" + projectPrompt
	}

	userTask := ""
	if t, ok := resolveRoutineTask(promptConfig, routineName); ok {
		userTask = t
	} else if t, ok := asNonEmptyStr(promptConfig["task"]); ok {
		userTask = t
	}

	modelCode := getModelCode(promptConfig)
	ctx := buildJinjaContext(promptConfig, sourceText, userTask, systemPromptForCtx)
	renderedSystem := renderJinjaTemplate(systemPromptForCtx, ctx)
	renderedTask := renderJinjaTemplate(userTask, ctx)

	promptContent := "SYSTEM PROMPT:\n" + renderedSystem + "\n\nTASK:\n" + renderedTask + "\n\nSOURCE MATERIAL:\n" + sourceText

	pl := payload{
		Model: modelCode,
		Messages: []message{
			{Role: "system", Content: systemInstruction},
			{Role: "user", Content: promptContent},
		},
	}

	if schema, ok := promptConfig["output_schema"]; ok && schema != nil {
		pl.ResponseFormat = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response_schema",
				"strict": true,
				"schema": schema,
			},
		}
	}

	return pl, promptContent
}

func coerceInt(v any) *int {
	switch x := v.(type) {
	case float64:
		i := int(x)
		if float64(i) == x {
			return &i
		}
		return nil
	case int:
		i := x
		return &i
	case json.Number:
		i64, err := x.Int64()
		if err != nil {
			return nil
		}
		i := int(i64)
		return &i
	case string:
		s := strings.TrimSpace(x)
		if s == "" {
			return nil
		}
		var n json.Number = json.Number(s)
		i64, err := n.Int64()
		if err != nil {
			return nil
		}
		i := int(i64)
		return &i
	default:
		return nil
	}
}

func extractUsageFromAPIResponse(apiJSON map[string]any, headers http.Header) Usage {
	usageAny, _ := apiJSON["usage"]
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
	matchedOld := oldString
	if ok {
		return updated, matchedOld, true
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

	return "", matchedOld, false
}

func applyReplacements(repls []Replacement, baseDir string) []string {
	changed := []string{}
	for i, r := range repls {
		filename := strings.TrimSpace(r.Filename)
		if filename == "" {
			fmt.Fprintf(os.Stderr, "Replacement #%d missing filename; skipping\n", i)
			continue
		}
		targetPath := filepath.Join(baseDir, filename)
		original, ok := readTextOrNone(targetPath)
		if !ok && strings.TrimSpace(r.OldString) == "" {
			_ = os.MkdirAll(filepath.Dir(targetPath), 0o755)
			_ = os.WriteFile(targetPath, []byte(r.NewString), 0o644)
			fmt.Printf("%s\n", filename)
			changed = append(changed, targetPath)
			continue
		}
		if !ok {
			fmt.Fprintf(os.Stderr, "Target file not found: %s\n", targetPath)
			continue
		}
		updated, _, applied := applyOneReplacement(original, r.OldString, r.NewString)
		if !applied {
			fmt.Printf("old_string not found in %s; skipping\n", filename)
			continue
		}
		if updated == original {
			fmt.Printf("No changes applied to %s (replacement produced identical content)\n", filename)
			continue
		}
		_ = os.WriteFile(targetPath, []byte(updated), 0o644)
		fmt.Printf("%s\n", filename)
		changed = append(changed, targetPath)
	}
	return changed
}

func repoFromDir(baseDir string) (string, bool) {
	p := baseDir
	for {
		gitDir := filepath.Join(p, ".git")
		st, err := os.Stat(gitDir)
		if err == nil && st.IsDir() {
			return p, true
		}
		parent := filepath.Dir(p)
		if parent == p {
			return "", false
		}
		p = parent
	}
}

func runCmd(dir string, name string, args ...string) (string, string, int) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	var outBuf bytes.Buffer
	var errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf
	err := cmd.Run()
	exitCode := 0
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = 1
		}
	}
	return outBuf.String(), errBuf.String(), exitCode
}

func stageAndCommit(repoRoot string, changedFiles []string, message string) {
	relpaths := make([]string, 0, len(changedFiles))
	for _, p := range changedFiles {
		relpaths = append(relpaths, toRel(repoRoot, p))
	}
	relpaths = dedupeKeepOrder(relpaths)
	if len(relpaths) == 0 {
		return
	}
	_, _, _ = runCmd(repoRoot, "git", append([]string{"add"}, relpaths...)...)
	_, _, _ = runCmd(repoRoot, "git", "commit", "-m", message)
}

func requireValidRepo(baseDir string) (string, bool) {
	repoRoot, ok := repoFromDir(baseDir)
	if !ok {
		fmt.Fprintln(os.Stderr, "Not a git repository")
		return "", false
	}
	_, _, code := runCmd(repoRoot, "git", "rev-parse", "--verify", "HEAD")
	if code != 0 {
		fmt.Fprintln(os.Stderr, "No valid commits")
		return "", false
	}
	return repoRoot, true
}

func undoLastCommit(baseDir string) int {
	repoRoot, ok := requireValidRepo(baseDir)
	if !ok {
		return 1
	}
	_, _, code := runCmd(repoRoot, "git", "rev-parse", "--verify", "HEAD~1")
	if code != 0 {
		fmt.Fprintln(os.Stderr, "No parent commit to reset to")
		return 1
	}
	_, _, code = runCmd(repoRoot, "git", "reset", "--hard", "HEAD~1")
	if code != 0 {
		return 1
	}
	return 0
}

func redoLastCommit(baseDir string) int {
	repoRoot, ok := requireValidRepo(baseDir)
	if !ok {
		return 1
	}
	origPath := filepath.Join(repoRoot, ".git", "ORIG_HEAD")
	if _, err := os.Stat(origPath); err != nil {
		fmt.Fprintln(os.Stderr, "No ORIG_HEAD found to redo to")
		return 1
	}
	out, _, _ := runCmd(repoRoot, "git", "rev-parse", "ORIG_HEAD")
	sha := strings.TrimSpace(out)
	if sha == "" {
		fmt.Fprintln(os.Stderr, "No ORIG_HEAD found to redo to")
		return 1
	}
	_, _, code := runCmd(repoRoot, "git", "reset", "--hard", sha)
	if code != 0 {
		return 1
	}
	return 0
}

func initRepoIfMissing(baseDir string) string {
	repoRoot, ok := repoFromDir(baseDir)
	if ok {
		return repoRoot
	}
	_, _, _ = runCmd(baseDir, "git", "init")
	fmt.Printf("Initialized empty Git repository in %s/.git\n", baseDir)
	return baseDir
}

func runBuildStep(config map[string]any, configPath string) *int {
	buildAny, ok := config["build"]
	if !ok {
		return nil
	}
	buildCmd, ok := asNonEmptyStr(buildAny)
	if !ok {
		return nil
	}
	fmt.Printf("\nBuild: %s\n\n", buildCmd)

	cmd := exec.Command("/bin/sh", "-c", buildCmd)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		} else {
			code = 1
		}
	}
	_ = configPath
	return &code
}

func callOpenAICompatible(endpointURL string, apiKey string, pl payload) (map[string]any, http.Header, error) {
	b, err := json.Marshal(pl)
	if err != nil {
		return nil, nil, err
	}
	req, err := http.NewRequest("POST", endpointURL, bytes.NewReader(b))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode != 200 {
		return nil, resp.Header, fmt.Errorf("API error status=%d body=%s", resp.StatusCode, string(body))
	}

	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var apiJSON map[string]any
	if err := dec.Decode(&apiJSON); err != nil {
		return nil, resp.Header, err
	}
	return apiJSON, resp.Header, nil
}

func parseModelResponse(apiJSON map[string]any) (ModelResponse, error) {
	choicesAny, ok := apiJSON["choices"]
	if !ok {
		return ModelResponse{}, fmt.Errorf("missing choices")
	}
	choices, ok := choicesAny.([]any)
	if !ok || len(choices) == 0 {
		return ModelResponse{}, fmt.Errorf("invalid choices")
	}
	choice0, ok := choices[0].(map[string]any)
	if !ok {
		return ModelResponse{}, fmt.Errorf("invalid choice")
	}
	msgAny, ok := choice0["message"]
	if !ok {
		return ModelResponse{}, fmt.Errorf("missing message")
	}
	msg, ok := msgAny.(map[string]any)
	if !ok {
		return ModelResponse{}, fmt.Errorf("invalid message")
	}
	contentAny, ok := msg["content"]
	if !ok {
		return ModelResponse{}, fmt.Errorf("missing content")
	}
	content, ok := contentAny.(string)
	if !ok {
		return ModelResponse{}, fmt.Errorf("invalid content")
	}
	content = stripJSONCodeFence(content)
	var mr ModelResponse
	dec := json.NewDecoder(strings.NewReader(content))
	dec.UseNumber()
	if err := dec.Decode(&mr); err != nil {
		return ModelResponse{}, err
	}
	return mr, nil
}

func run(configPath string, routineName string, debug bool, noModel bool) (ModelResponse, Usage, string, map[string]any) {
	config, err := loadPromptFile(configPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	if strings.TrimSpace(routineName) != "" {
		if _, ok := resolveRoutineTask(config, routineName); !ok {
			routinesAny, _ := config["routines"]
			routines, _ := routinesAny.(map[string]any)
			available := []string{}
			for k := range routines {
				available = append(available, k)
			}
			sort.Strings(available)
			fmt.Fprintf(os.Stderr, "Routine not found: %s\n", routineName)
			if len(available) > 0 {
				fmt.Fprintf(os.Stderr, "Available routines: %s\n", strings.Join(available, ", "))
			}
			os.Exit(2)
		}
	}

	endpointURL := DefaultURL
	if s, ok := asNonEmptyStr(config["endpoint"]); ok {
		endpointURL = s
	}
	apiKeyEnvVar := DefaultAPIKeyEnvVar
	if s, ok := asNonEmptyStr(config["api_key_env_var"]); ok {
		apiKeyEnvVar = s
	}
	if strings.Contains(endpointURL, "openrouter.ai") && apiKeyEnvVar == DefaultAPIKeyEnvVar {
		apiKeyEnvVar = "OPENROUTER_API_KEY"
	}
	apiKey := strings.TrimSpace(os.Getenv(apiKeyEnvVar))

	systemPromptRaw, _ := asNonEmptyStr(config["system_prompt"])
	projectPrompt, _ := asNonEmptyStr(config["project"])
	systemPromptForCtx := systemPromptRaw
	if projectPrompt != "" {
		systemPromptForCtx = systemPromptForCtx + "\n\n" + projectPrompt
	}
	userTaskForCtx := ""
	if t, ok := resolveRoutineTask(config, routineName); ok {
		userTaskForCtx = t
	} else if t, ok := asNonEmptyStr(config["task"]); ok {
		userTaskForCtx = t
	}

	fileJinjaContext := buildJinjaContext(config, "", userTaskForCtx, systemPromptForCtx)
	renderSourcesAsJinja := false
	if v, ok := config["render_source_files_as_jinja"]; ok {
		if b, ok := v.(bool); ok {
			renderSourcesAsJinja = b
		}
	}
	var sourceJinjaContext map[string]any
	if renderSourcesAsJinja {
		sourceJinjaContext = fileJinjaContext
	}

	cwd, _ := os.Getwd()
	sourceMaterial := buildSourceMaterial(config, cwd, sourceJinjaContext)
	pl, promptContent := buildPayload(config, sourceMaterial, routineName, endpointURL)

	if debug {
		fmt.Fprintln(os.Stdout, "\n--- PROMPT (user message content) ---\n")
		fmt.Fprintln(os.Stdout, promptContent)
		fmt.Fprintln(os.Stdout, "\n--- END PROMPT ---\n")
	}

	if noModel {
		fmt.Fprintln(os.Stdout, "--no-model specified; stopping before API request")
		enc, _ := json.MarshalIndent(pl, "", "  ")
		fmt.Fprintln(os.Stdout, string(enc))
		os.Exit(0)
	}

	apiJSON, respHeaders, err := callOpenAICompatible(endpointURL, apiKey, pl)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	modelCode := getModelCode(config)
	usage := extractUsageFromAPIResponse(apiJSON, respHeaders)
	mr, err := parseModelResponse(apiJSON)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	return mr, usage, modelCode, config
}

func estimateCostUSD(modelCode string, usage Usage) *float64 {
	pt := 0
	ct := 0
	if usage.PromptTokens != nil {
		pt = *usage.PromptTokens
	}
	if usage.CompletionTokens != nil {
		ct = *usage.CompletionTokens
	}
	if pt == 0 && ct == 0 {
		return nil
	}
	prices, ok := ModelPricesUSDPer1M[modelCode]
	if !ok {
		prices = Prices{InUSDPer1M: DefaultFallbackInputUSDPer1M, OutUSDPer1M: DefaultFallbackOutputUSDPer1M}
	}
	cost := (float64(pt)*prices.InUSDPer1M + float64(ct)*prices.OutUSDPer1M) / 1_000_000.0
	return &cost
}

func main() {
	promptPath := flag.String("prompt", DefaultPromptFile, "Path to prompt/config file")
	promptPathShort := flag.String("p", DefaultPromptFile, "Path to prompt/config file")
	newFlag := flag.Bool("new", false, "Copy bundled default prompt into CWD (not implemented in Go scaffold)")
	undoFlag := flag.Bool("undo", false, "Undo last git commit (hard reset to HEAD~1)")
	undoFlagShort := flag.Bool("u", false, "Undo last git commit (hard reset to HEAD~1)")
	redoFlag := flag.Bool("redo", false, "Redo last undo (hard reset to ORIG_HEAD)")
	debugFlag := flag.Bool("debug", false, "Print the prompt content before sending")
	noModelFlag := flag.Bool("no-model", false, "Stop before API request")
	flag.Parse()

	cfgPath := *promptPath
	if *promptPathShort != DefaultPromptFile || cfgPath == DefaultPromptFile {
		cfgPath = *promptPathShort
	}

	cwd, _ := os.Getwd()
	if *undoFlag || *undoFlagShort {
		os.Exit(undoLastCommit(cwd))
	}
	if *redoFlag {
		os.Exit(redoLastCommit(cwd))
	}
	if *newFlag {
		fmt.Fprintln(os.Stderr, "--new is not implemented in Go scaffold")
		os.Exit(1)
	}

	routineName := ""
	args := flag.Args()
	if len(args) > 0 {
		routineName = args[0]
	}

	mr, usage, modelCode, config := run(cfgPath, routineName, *debugFlag, *noModelFlag)
	changed := applyReplacements(mr.Edit.Replacements, cwd)

	commitMessage := strings.TrimSpace(mr.Edit.CommitMessage)
	if commitMessage == "" {
		commitMessage = "Apply model edits"
	}
	fmt.Printf("\nCommit message: %s\n\n", commitMessage)

	if cost := estimateCostUSD(modelCode, usage); cost != nil {
		fmt.Printf("$%.4f\n", *cost)
	}

	repoRoot := initRepoIfMissing(cwd)
	if len(changed) == 0 {
		fmt.Println("No files changed; skipping commit")
		return
	}
	stageAndCommit(repoRoot, changed, commitMessage)

	_ = config
	_ = runBuildStep(config, cfgPath)
}
