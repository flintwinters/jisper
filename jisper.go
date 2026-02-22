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
	"strings"
	"time"

	"github.com/hexops/gotextdiff"
	"github.com/hexops/gotextdiff/myers"
	"github.com/hexops/gotextdiff/span"
	"github.com/pterm/pterm"
	cli "github.com/urfave/cli"
	"go.yaml.in/yaml/v4"
)

const (
	DefaultPromptFile             = "prompt.yaml"
	DefaultTemplatePromptFile     = "default_prompt.yaml"
	DefaultAPIKeyEnvVar           = "OPENAI_API_KEY"
	DefaultURL                    = "https://api.openai.com/v1/chat/completions"
	DefaultFallbackInputUSDPer1M  = 5.0
	DefaultFallbackOutputUSDPer1M = 15.0
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

func buildPayload(promptConfig map[string]any, sourceText string, routineName string, _ string) (payload, string) {
	systemInstruction := "You are a helpful assistant."
	if s, ok := asNonEmptyStr(promptConfig["system_instruction"]); ok {
		systemInstruction = s
	}
	systemPromptForCtx := resolveSystemPrompt(promptConfig)

	userTask := resolveUserTask(promptConfig, routineName)
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

	if schema, ok := promptConfig["output_schema"]; ok && schema != nil {
		pl.ResponseFormat = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response_schema",
				"strict": true,
				"schema": schema,
			},
		}
	} else {
		pl.ResponseFormat = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response_schema",
				"strict": true,
				"schema": DefaultOutputSchema,
			},
		}
	}

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

func guessLexer(text string, filename string, language string) string {
	if language != "" {
		return language
	}
	if filename != "" {
		if l, ok := GetExtLexerMapping()[strings.ToLower(filepath.Ext(filename))]; ok {
			return l
		}
	}
	sample := strings.ToLower(text)
	if len(sample) > 500 {
		sample = sample[:500]
	}
	if strings.HasPrefix(sample, "diff ") || strings.Contains(sample, "+++") || strings.Contains(sample, "@@ ") {
		return "diff"
	}
	if strings.HasPrefix(sample, "{") || strings.HasPrefix(sample, "[") {
		return "json"
	}
	if strings.Contains(sample, "def ") || strings.Contains(sample, "import ") {
		return "python"
	}
	return "text"
}

var styleKeyword = pterm.NewStyle(pterm.FgLightBlue)
var styleString = pterm.NewStyle(pterm.FgYellow)
var styleNumber = pterm.NewStyle(pterm.FgLightMagenta)
var styleComment = pterm.NewStyle(pterm.FgGray)

func isLetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func isDigit(b byte) bool { return b >= '0' && b <= '9' }

func isWordChar(b byte) bool { return isLetter(b) || isDigit(b) || b == '_' }

func keywordSetForLexer(lexer string) map[string]bool {
	if lexer == "go" {
		return map[string]bool{
			"break": true, "case": true, "chan": true, "const": true,
			"continue": true, "default": true, "defer": true, "else": true,
			"fallthrough": true, "for": true, "func": true, "go": true,
			"goto": true, "if": true, "import": true, "interface": true,
			"map": true, "package": true, "range": true, "return": true,
			"select": true, "struct": true, "switch": true, "type": true,
			"var": true,
		}
	}
	if lexer == "python" {
		return map[string]bool{
			"elif": true, "else": true, "except": true, "False": true,
			"finally": true, "for": true, "from": true, "global": true,
			"if": true, "import": true, "in": true, "is": true,
			"lambda": true, "None": true, "nonlocal": true, "not": true,
			"or": true, "pass": true, "raise": true, "return": true,
			"True": true, "try": true, "while": true, "with": true,
			"yield": true,
		}
	}
	if lexer == "javascript" || lexer == "typescript" {
		return map[string]bool{
			"break": true, "case": true, "catch": true, "class": true,
			"const": true, "continue": true, "debugger": true, "default": true,
			"delete": true, "do": true, "else": true, "export": true,
			"extends": true, "finally": true, "for": true, "function": true,
			"if": true, "import": true, "in": true, "instanceof": true,
			"let": true, "new": true, "return": true, "super": true,
			"switch": true, "this": true, "throw": true, "try": true,
			"typeof": true, "var": true, "void": true, "while": true,
			"with": true, "yield": true,
		}
	}
	return map[string]bool{}
}

func splitLineComment(line string, lexer string) (string, string) {
	marker := ""
	if lexer == "go" || lexer == "javascript" || lexer == "typescript" {
		marker = "//"
	}
	if lexer == "python" || lexer == "yaml" || lexer == "bash" {
		marker = "#"
	}
	if marker == "" {
		return line, ""
	}
	quote := byte(0)
	escape := false
	for i := 0; i < len(line); i++ {
		b := line[i]
		if quote != 0 {
			if quote != '`' && escape {
				escape = false
				continue
			}
			if quote != '`' && b == '\\' {
				escape = true
				continue
			}
			if b == quote {
				quote = 0
			}
			continue
		}
		if b == '\'' || b == '"' || b == '`' {
			quote = b
			continue
		}
		if marker == "#" && b == '#' {
			return line[:i], line[i:]
		}
		if marker == "//" && i+1 < len(line) && line[i:i+2] == "//" {
			return line[:i], line[i:]
		}
	}
	return line, ""
}

func processChar(b byte, i int, state *lexerState) {
	if state.quote != 0 {
		state.flushWord(i)
		state.flushNumber(i)
		if state.quote != '`' && state.escape {
			state.escape = false
			return
		}
		if state.quote != '`' && b == '\\' {
			state.escape = true
			return
		}
		state.out.WriteString(styleString.Sprint(string(b)))
		if b == state.quote {
			state.quote = 0
		}
		return
	}
	if b == '\'' || b == '"' || b == '`' {
		state.flushWord(i)
		state.flushNumber(i)
		state.quote = b
		state.out.WriteString(styleString.Sprint(string(b)))
		return
	}
	if state.wordStart >= 0 && !isWordChar(b) {
		state.flushWord(i)
	}
	if state.numberStart >= 0 && !isDigit(b) && b != '.' && b != '_' {
		state.flushNumber(i)
	}
	if state.wordStart < 0 && (isLetter(b) || b == '_') {
		state.wordStart = i
		return
	}
	if state.numberStart < 0 && isDigit(b) {
		state.numberStart = i
		return
	}
	if state.wordStart < 0 && state.numberStart < 0 {
		state.out.WriteByte(b)
	}
}

type lexerState struct {
	quote                  byte
	escape                 bool
	wordStart, numberStart int
	keywords               map[string]bool
	code                   string
	out                    *strings.Builder
	flushWord, flushNumber func(int)
}

func highlightCode(code string, lexer string, out *strings.Builder) {
	state := &lexerState{
		wordStart: -1, numberStart: -1, code: code, out: out, keywords: keywordSetForLexer(lexer),
	}
	state.flushWord = func(i int) {
		if state.wordStart < 0 {
			return
		}
		if w := code[state.wordStart:i]; state.keywords[w] {
			out.WriteString(styleKeyword.Sprint(w))
		} else {
			out.WriteString(w)
		}
		state.wordStart = -1
	}
	state.flushNumber = func(i int) {
		if state.numberStart < 0 {
			return
		}
		out.WriteString(styleNumber.Sprint(code[state.numberStart:i]))
		state.numberStart = -1
	}
	for i := 0; i < len(code); i++ {
		processChar(code[i], i, state)
	}
	state.flushWord(len(code))
	state.flushNumber(len(code))
}

func highlightLine(line string, lexer string) string {
	line = strings.TrimSuffix(line, "\n")
	if lexer == "" || lexer == "text" || lexer == "diff" || lexer == "markdown" {
		return line
	}
	code, comment := splitLineComment(line, lexer)
	var out strings.Builder
	highlightCode(code, lexer, &out)
	if comment != "" {
		out.WriteString(styleComment.Sprint(comment))
	}
	return out.String()
}

func segmentDiffRanges(lines []gotextdiff.Line, context int) [][2]int {
	ranges := [][2]int{}
	i := 0
	for i < len(lines) {
		for i < len(lines) && lines[i].Kind == gotextdiff.Equal {
			i++
		}
		if i >= len(lines) {
			break
		}
		start := i - context
		if start < 0 {
			start = 0
		}
		end := i + context + 1
		j := i + 1
		for j < len(lines) {
			if lines[j].Kind != gotextdiff.Equal {
				end = j + context + 1
			}
			if j >= end {
				break
			}
			j++
		}
		if end > len(lines) {
			end = len(lines)
		}
		ranges = append(ranges, [2]int{start, end})
		i = end
	}
	return ranges
}

func styledLineNumber(ln *int, style *pterm.Style, width int) string {
	if ln == nil {
		return strings.Repeat(" ", width)
	}
	s := fmt.Sprintf("%*d", width, *ln)
	if style != nil {
		return style.Sprint(s)
	}
	return s
}

func printNumberedCombinedDiff(oldText, newText, filename, language string) {
	lines := formatCombinedDiffLines(oldText, newText, filename, language, 2)
	if len(lines) == 0 {
		pterm.Info.Println("(no diff; content is identical)")
		return
	}
	for _, l := range lines {
		fmt.Println(l)
	}
}

func formatCombinedDiffLines(oldText, newText, filename, language string, contextLines int) []string {
	edits := myers.ComputeEdits(span.URIFromPath("a"), oldText, newText)
	diff := gotextdiff.ToUnified("a", "b", oldText, edits)
	lexer := guessLexer(oldText+"\n"+newText, filename, language)
	styleDel := pterm.NewStyle(pterm.FgLightRed)
	styleIns := pterm.NewStyle(pterm.FgLightGreen)
	styleDelPre, styleInsPre := pterm.NewStyle(pterm.FgLightRed), pterm.NewStyle(pterm.FgLightGreen)
	out := []string{}
	for _, hunk := range diff.Hunks {
		lines := hunk.Lines
		if len(lines) == 0 {
			continue
		}
		oldPrefix, newPrefix := make([]int, len(lines)+1), make([]int, len(lines)+1)
		for i, line := range lines {
			oldPrefix[i+1], newPrefix[i+1] = oldPrefix[i], newPrefix[i]
			if line.Kind != gotextdiff.Insert {
				oldPrefix[i+1]++
			}
			if line.Kind != gotextdiff.Delete {
				newPrefix[i+1]++
			}
		}
		for _, r := range segmentDiffRanges(lines, contextLines) {
			oldLn, newLn := hunk.FromLine+oldPrefix[r[0]], hunk.ToLine+newPrefix[r[0]]
			for i := r[0]; i < r[1]; i++ {
				line := lines[i]
				content := strings.TrimSuffix(line.Content, "\n")
				s := ""
				switch line.Kind {
				case gotextdiff.Insert:
					s = styledLineNumber(&newLn, styleIns, 4) + styleInsPre.Sprint(" + ") + highlightLine(content, lexer)
					newLn++
				case gotextdiff.Delete:
					s = styledLineNumber(&oldLn, styleDel, 4) + styleDelPre.Sprint(" - ") + highlightLine(content, lexer)
					oldLn++
				default:
					s = styledLineNumber(&oldLn, nil, 4) + "   " + highlightLine(content, lexer)
					oldLn++
					newLn++
				}
				out = append(out, s)
			}
		}
	}
	return out
}

func applyReplacements(repls []Replacement, baseDir string, language string, configPath string) []string {
	changed := []string{}
	for i, r := range repls {
		filename := strings.TrimSpace(r.Filename)
		if filename == "" {
			pterm.Error.Printfln("Replacement #%d missing filename; skipping. Replacement object: %+v", i, r)
			continue
		}
		targetPath := filepath.Join(baseDir, filename)
		original, ok := readFileContent(baseDir, filename)
		if !ok && strings.TrimSpace(r.OldString) == "" {
			_ = os.MkdirAll(filepath.Dir(targetPath), 0o755)
			fmt.Printf("\n\x1b[1m%s\x1b[0m\n", filename)
			printNumberedCombinedDiff("", r.NewString, filename, language)
			if err := os.WriteFile(targetPath, []byte(r.NewString), 0o644); err != nil {
				pterm.Error.Printfln("Failed to create file %s: %v", targetPath, err)
				continue
			}
			changed = append(changed, targetPath)
			continue
		}
		if !ok {
			pterm.Error.Printfln("Target file not found: %s (resolved to: %s). Cannot apply replacement #%d.", filename, targetPath, i)
			continue
		}
		updated, _, applied := applyOneReplacement(original, r.OldString, r.NewString)
		if !applied {
			oldPreview := r.OldString
			if len(oldPreview) > 200 {
				oldPreview = oldPreview[:200] + "... (truncated, total length: " + fmt.Sprintf("%d", len(r.OldString)) + " chars)"
			}
			pterm.Warning.Printfln(
				"old_string not found in %s; skipping repl #%d. Searched for (first 200 chars): %q",
				filename, i, oldPreview,
			)
			writeFailedOldStringToConfig(configPath, r.OldString)
			continue
		}
		if updated == original {
			pterm.Info.Printfln("No changes applied to %s (old_string matched but replacement produced identical content)", filename)
			continue
		}
		fmt.Printf("\x1b[1m%s\x1b[0m\n", filename)
		printNumberedCombinedDiff(original, updated, filename, language)
		if err := os.WriteFile(targetPath, []byte(updated), 0o644); err != nil {
			pterm.Error.Printfln("Failed to write file %s: %v", targetPath, err)
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

func runBuildStep(config map[string]any, configPath string) *int {
	cmdStr, ok := asNonEmptyStr(config["build"])
	if !ok {
		return nil
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
	return &code
}

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
		return nil, resp.Header, fmt.Errorf("API request to %s returned status %d. Response body: %s", endpointURL, resp.StatusCode, bodyPreview)
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
		return ModelResponse{}, fmt.Errorf("API response choices[0].message is not an object, got %T", msgAny)
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

func prepareRun(configPath string, routineName string) (map[string]any, payload, string, string, string) {
	config, err := loadPromptFile(configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load prompt config from %s: %v\n", configPath, err)
		os.Exit(1)
	}
	if strings.TrimSpace(routineName) != "" {
		if _, ok := resolveRoutineTask(config, routineName); !ok {
			routines, _ := config["routines"].(map[string]any)
			available := []string{}
			for k := range routines {
				available = append(available, k)
			}
			sort.Strings(available)
			if len(available) == 0 {
				fmt.Fprintf(os.Stderr, "Routine not found: %s. No routines are defined in the config file.\n", routineName)
			} else {
				fmt.Fprintf(os.Stderr, "Routine not found: %s. Available routines: %s\n", routineName, strings.Join(available, ", "))
			}
			os.Exit(2)
		}
	}
	endpointURL := DefaultURL
	if s, ok := asNonEmptyStr(config["endpoint"]); ok {
		endpointURL = s
	}
	keyVar := DefaultAPIKeyEnvVar
	if s, ok := asNonEmptyStr(config["api_key_env_var"]); ok {
		keyVar = s
	}
	if strings.Contains(endpointURL, "openrouter.ai") && keyVar == DefaultAPIKeyEnvVar {
		keyVar = "OPENROUTER_API_KEY"
	}
	apiKey := strings.TrimSpace(os.Getenv(keyVar))
	if apiKey == "" {
		fmt.Fprintf(os.Stderr, "API key not found: environment variable %s is not set or empty. Set it with: export %s=your-api-key\n", keyVar, keyVar)
		os.Exit(1)
	}
	sysCtx := resolveSystemPrompt(config)
	userCtx := resolveUserTask(config, routineName)
	fileCtx := buildJinjaContext(config, "", userCtx, sysCtx)
	render, _ := config["render_source_files_as_jinja"].(bool)
	var srcCtx map[string]any
	if render {
		srcCtx = fileCtx
	}
	cwd, _ := os.Getwd()
	srcMaterial := buildSourceMaterial(config, cwd, srcCtx)
	pl, pContent := buildPayload(config, srcMaterial, routineName, endpointURL)
	return config, pl, pContent, apiKey, endpointURL
}

func buildPayloadWithTask(promptConfig map[string]any, sourceText string, task string) (payload, string) {
	systemInstruction := "You are a helpful assistant."
	if s, ok := asNonEmptyStr(promptConfig["system_instruction"]); ok {
		systemInstruction = s
	}
	systemPromptForCtx := resolveSystemPrompt(promptConfig)
	ctx := buildJinjaContext(promptConfig, sourceText, task, systemPromptForCtx)
	renderedSystem := render(systemPromptForCtx, ctx)
	renderedTask := render(task, ctx)
	modelCode := getModelCode(promptConfig)
	promptContent := fmt.Sprintf("SYSTEM PROMPT:\n%s\n\nTASK:\n%s\n\nSOURCE MATERIAL:\n%s",
		renderedSystem, renderedTask, sourceText)
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
	} else {
		pl.ResponseFormat = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response_schema",
				"strict": true,
				"schema": DefaultOutputSchema,
			},
		}
	}
	return pl, promptContent
}

func callModel(endpointURL string, apiKey string, pl payload, config map[string]any) (ModelResponse, Usage, string) {
	modelCode := getModelCode(config)
	spinner, _ := pterm.DefaultSpinner.Start(fmt.Sprintf("Waiting for %s...", modelCode))
	apiJSON, headers, err := callOpenAICompatible(endpointURL, apiKey, pl)
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

func runIssues(issues IssuesFile, promptPath string, debug bool, noModel bool) {
	config, err := loadPromptFile(promptPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config from %s: %v\n", promptPath, err)
		os.Exit(1)
	}
	endpointURL, keyVar := setupEndpointAndKey(config)
	apiKey := validateAndGetAPIKey(keyVar)
	modelCode := getModelCode(config)
	fmt.Fprintf(os.Stderr, "DEBUG: modelCode=%q endpointURL=%q keyVar=%q\n", modelCode, endpointURL, keyVar)
	for i, issue := range issues.Issues {
		pterm.Info.Printfln("Processing issue %d/%d (model: %s)", i+1, len(issues.Issues), modelCode)
		context, ok := extractLinesAround(issue.Pos.Filename, issue.Pos.Line, ".", 40, 40)
		if !ok {
			pterm.Error.Printfln("Failed to read %s (line %d). File may not exist or line number is out of range.", issue.Pos.Filename, issue.Pos.Line)
			continue
		}
		task := fmt.Sprintf("Fix this linting issue in %s at line %d.\n\nLinter: %s\nMessage: %s\n\nCode:\n%s",
			issue.Pos.Filename, issue.Pos.Line, issue.FromLinter, issue.Text, context)
		pl, content := buildPayloadWithTask(config, context, task)
		if debug {
			fmt.Printf("\n--- PROMPT ---\n%s\n--- END PROMPT ---\n", content)
		}
		if noModel {
			enc, _ := json.MarshalIndent(pl, "", "  ")
			fmt.Println(string(enc))
			continue
		}
		mr, usage, mc := callModel(endpointURL, apiKey, pl, config)
		lang, _ := asNonEmptyStr(config["language"])
		changed := applyReplacements(mr.Edit.Replacements, ".", lang, promptPath)
		msg := strings.TrimSpace(mr.Edit.CommitMessage)
		if msg == "" {
			msg = fmt.Sprintf("Fix %s issue in %s", issue.FromLinter, issue.Pos.Filename)
		}
		pterm.Info.Printfln("Commit: %s", msg)
		prices := getModelPrices(config)
		if cost := estimateCostUSD(mc, usage, prices); cost != nil {
			pterm.Success.Printfln("$%.4f", *cost)
		}
		repo := initRepoIfMissing(".")
		if len(changed) > 0 {
			stageAndCommit(repo, changed, msg)
		}
	}
}

func run(path string, routine string, debug bool, noModel bool) (ModelResponse, Usage, string, map[string]any) {
	cfg, pl, content, key, endpoint := prepareRun(path, routine)
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
	prices := make(map[string]Prices)
	for k, v := range ModelPricesUSDPer1M {
		prices[k] = v
	}
	customAny, ok := config["model_prices_usd_per_1m"]
	if !ok {
		return prices
	}
	custom, ok := customAny.(map[string]any)
	if !ok {
		return prices
	}
	for model, val := range custom {
		if arr, ok := val.([]any); ok && len(arr) >= 2 {
			inPrice, ok1 := coerceFloat(arr[0])
			outPrice, ok2 := coerceFloat(arr[1])
			if ok1 && ok2 {
				prices[model] = Prices{InUSDPer1M: inPrice, OutUSDPer1M: outPrice}
			}
		}
	}
	return prices
}

func estimateCostUSD(modelCode string, usage Usage, prices map[string]Prices) *float64 {
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
	p, ok := prices[modelCode]
	if !ok {
		p = Prices{InUSDPer1M: DefaultFallbackInputUSDPer1M, OutUSDPer1M: DefaultFallbackOutputUSDPer1M}
	}
	cost := (float64(pt)*p.InUSDPer1M + float64(ct)*p.OutUSDPer1M) / 1_000_000.0
	return &cost
}

func writeDefaultPromptToCWD() int {
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Cannot determine executable path: %v\n", err)
		return 1
	}
	src := filepath.Join(filepath.Dir(exe), DefaultTemplatePromptFile)
	if _, err := os.Stat(src); err != nil {
		msg := "Missing template prompt file: %s (error: %v). " +
			"The executable may be corrupted or installed incorrectly.\n"
		fmt.Fprintf(os.Stderr, msg, src, err)
		return 1
	}
	dst := filepath.Join(".", DefaultPromptFile)
	if _, err := os.Stat(dst); err == nil {
		fmt.Fprintf(os.Stderr, "%s already exists; refusing to overwrite. Remove it first if you want to create a new one.\n", DefaultPromptFile)
		return 1
	}
	content, err := os.ReadFile(src)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read template from %s: %v\n", src, err)
		return 1
	}
	if err := os.WriteFile(dst, content, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to write %s: %v\n", dst, err)
		return 1
	}
	fmt.Printf("Wrote %s\n", DefaultPromptFile)
	return 0
}

func handleGlobalFlags(c *cli.Context) bool {
	if c.Bool("new") {
		os.Exit(writeDefaultPromptToCWD())
	}
	if c.Bool("redo") {
		os.Exit(redoLastCommit("."))
	}
	if c.Bool("undo") {
		os.Exit(undoLastCommit("."))
	}
	return false
}

func executeRunAction(c *cli.Context) error {
	handleGlobalFlags(c)
	promptPath := c.String("prompt")
	if c.IsSet("issues") {
		issuesFile := c.String("issues")
		issues, err := loadIssuesFile(issuesFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to load issues file %s: %v\n", issuesFile, err)
			os.Exit(1)
		}
		if len(issues.Issues) == 0 {
			fmt.Fprintf(os.Stderr, "Issues file %s contains no issues to process.\n", issuesFile)
			os.Exit(0)
		}
		runIssues(issues, promptPath, c.Bool("debug"), c.Bool("no-model"))
		return nil
	}
	if !c.IsSet("prompt") && c.NArg() > 0 {
		if _, err := os.Stat(c.Args().Get(0)); err == nil {
			promptPath = c.Args().Get(0)
		}
	}
	routine := ""
	if c.NArg() > 0 && c.Args().Get(0) != promptPath {
		routine = c.Args().Get(0)
	} else if c.NArg() > 1 {
		routine = c.Args().Get(1)
	}
	mr, usage, mc, config := run(promptPath, routine, c.Bool("debug"), c.Bool("no-model"))
	lang, _ := asNonEmptyStr(config["language"])
	changed := applyReplacements(mr.Edit.Replacements, ".", lang, promptPath)
	msg := strings.TrimSpace(mr.Edit.CommitMessage)
	if msg == "" {
		msg = "Apply model edits"
	}
	pterm.Info.Printfln("Commit: %s", msg)
	prices := getModelPrices(config)
	if cost := estimateCostUSD(mc, usage, prices); cost != nil {
		pterm.Success.Printfln("$%.4f", *cost)
	}
	repo := initRepoIfMissing(".")
	if len(changed) == 0 {
		fmt.Println("No changes; skipping commit")
		return nil
	}
	stageAndCommit(repo, changed, msg)
	_ = runBuildStep(config, promptPath)
	return nil
}

func main() {
	app := &cli.App{
		Name: "jisper", Usage: "CLI for Jisper",
		Flags: []cli.Flag{
			cli.StringFlag{Name: "prompt, p", Value: DefaultPromptFile},
			cli.BoolFlag{Name: "new"}, cli.BoolFlag{Name: "undo, u"},
			&cli.BoolFlag{Name: "redo"}, &cli.BoolFlag{Name: "debug"}, &cli.BoolFlag{Name: "no-model"},
			cli.StringFlag{Name: "issues", Value: "issues.json", Usage: "Path to issues JSON file"},
		},
		Action: executeRunAction,
	}
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
