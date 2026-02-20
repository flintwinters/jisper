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
	yaml "go.yaml.in/yaml/v4"
)

const (
	DefaultPromptFile             = "prompt.yaml"
	DefaultAPIKeyEnvVar           = "OPENAI_API_KEY"
	DefaultURL                    = "https://api.openai.com/v1/chat/completions"
	DefaultFallbackInputUSDPer1M  = 5.0
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
	"qwen/qwen3-coder:exacto":   {InUSDPer1M: 0.22, OutUSDPer1M: 1.8},
	"moonshotai/kimi-k2.5":      {InUSDPer1M: 0.25, OutUSDPer1M: 2.25},
	"z-ai/glm-5":                {InUSDPer1M: 1.0, OutUSDPer1M: 3.2},
	"openai/gpt-oss-120b:nitro": {InUSDPer1M: 0.35, OutUSDPer1M: 0.95},
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
		txt, ok := readTextOrNone(filepath.Join(baseDir, f))
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
		txt, ok := readTextOrNone(filepath.Join(baseDir, filename))
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
		n := json.Number(s)
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

func guessLexer(text string, filename string, language string) string {
	if language != "" {
		return language
	}
	if filename != "" {
		ext := strings.ToLower(filepath.Ext(filename))
		mapping := map[string]string{
			".py": "python", ".json": "json", ".json5": "json",
			".go": "go", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
			".diff": "diff", ".patch": "diff", ".toml": "toml",
			".sh": "bash", ".bash": "bash", ".js": "javascript",
			".ts": "typescript", ".html": "html", ".css": "css", ".sql": "sql",
			".rs": "rust", ".c": "c", ".cpp": "cpp", ".h": "cpp",
		}
		if l, ok := mapping[ext]; ok {
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
	styleDel, styleIns := pterm.NewStyle(pterm.FgLightRed, pterm.BgRed), pterm.NewStyle(pterm.FgLightGreen, pterm.BgGreen)
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

func applyReplacements(repls []Replacement, baseDir string, language string) []string {
	changed := []string{}
	for i, r := range repls {
		filename := strings.TrimSpace(r.Filename)
		if filename == "" {
			pterm.Error.Printfln("Replacement #%d missing filename; skipping", i)
			continue
		}
		targetPath := filepath.Join(baseDir, filename)
		original, ok := readTextOrNone(targetPath)
		if !ok && strings.TrimSpace(r.OldString) == "" {
			_ = os.MkdirAll(filepath.Dir(targetPath), 0o755)
			pterm.DefaultHeader.WithFullWidth().Println(filename)
			printNumberedCombinedDiff("", r.NewString, filename, language)
			_ = os.WriteFile(targetPath, []byte(r.NewString), 0o644)
			changed = append(changed, targetPath)
			continue
		}
		if !ok {
			pterm.Error.Printfln("Target file not found: %s", targetPath)
			continue
		}
		updated, _, applied := applyOneReplacement(original, r.OldString, r.NewString)
		if !applied {
			pterm.Warning.Printfln("old_string not found in %s; skipping", filename)
			continue
		}
		if updated == original {
			pterm.Info.Printfln("No changes applied to %s", filename)
			continue
		}
		pterm.DefaultHeader.WithFullWidth().Println(filename)
		printNumberedCombinedDiff(original, updated, filename, language)
		_ = os.WriteFile(targetPath, []byte(updated), 0o644)
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

func runCmd(dir, _ string, args ...string) (string, int) {
	cmd := exec.Command("git", args...)
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
	return outBuf.String(), exitCode
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
	_, _ = runCmd(repoRoot, "git", append([]string{"add"}, relpaths...)...)
	_, _ = runCmd(repoRoot, "git", "commit", "-m", message)
}

func requireValidRepo(baseDir string) (string, bool) {
	repoRoot, ok := repoFromDir(baseDir)
	if !ok {
		fmt.Fprintln(os.Stderr, "Not a git repository")
		return "", false
	}
	_, code := runCmd(repoRoot, "git", "rev-parse", "--verify", "HEAD")
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
	_, code := runCmd(repoRoot, "git", "rev-parse", "--verify", "HEAD~1")
	if code != 0 {
		fmt.Fprintln(os.Stderr, "No parent commit to reset to")
		return 1
	}
	_, code = runCmd(repoRoot, "git", "reset", "--hard", "HEAD~1")
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
	out, _ := runCmd(repoRoot, "git", "rev-parse", "ORIG_HEAD")
	sha := strings.TrimSpace(out)
	if sha == "" {
		fmt.Fprintln(os.Stderr, "No ORIG_HEAD found to redo to")
		return 1
	}
	_, code := runCmd(repoRoot, "git", "reset", "--hard", sha)
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
	_, _ = runCmd(baseDir, "git", "init")
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
	b, err := os.ReadFile(configPath)
	if err != nil {
		return &code
	}
	var node yaml.Node
	if err := yaml.Unmarshal(b, &node); err != nil || len(node.Content) == 0 {
		return &code
	}
	root := node.Content[0]
	updateKey := func(k string, v any) {
		valStr := fmt.Sprintf("%v", v)
		for i := 0; i < len(root.Content); i += 2 {
			if root.Content[i].Value == k {
				root.Content[i+1].Value = valStr
				return
			}
		}
		root.Content = append(root.Content,
			&yaml.Node{Kind: yaml.ScalarNode, Value: k},
			&yaml.Node{Kind: yaml.ScalarNode, Value: valStr})
	}
	if code == 0 {
		updateKey("success", "true")
	} else {
		updateKey("error", fmt.Sprintf("build failed (%d)", code))
	}
	out, _ := yaml.Marshal(&node)
	_ = os.WriteFile(configPath, out, 0o644)
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
	defer func() { _ = resp.Body.Close() }()

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

func prepareRun(configPath string, routineName string) (map[string]any, payload, string, string, string) {
	config, err := loadPromptFile(configPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
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
			fmt.Fprintf(os.Stderr, "Routine not found: %s\nAvailable: %s\n", routineName, strings.Join(available, ", "))
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
	sysRaw, _ := asNonEmptyStr(config["system_prompt"])
	projPrompt, _ := asNonEmptyStr(config["project"])
	sysCtx := sysRaw
	if projPrompt != "" {
		sysCtx = sysCtx + "\n\n" + projPrompt
	}
	userCtx := ""
	if t, ok := resolveRoutineTask(config, routineName); ok {
		userCtx = t
	} else if t, ok := asNonEmptyStr(config["task"]); ok {
		userCtx = t
	}
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

func callModel(endpoint string, key string, pl payload, config map[string]any) (ModelResponse, Usage, string) {
	spinner, _ := pterm.DefaultSpinner.Start("Waiting for model...")
	apiJSON, headers, err := callOpenAICompatible(endpoint, key, pl)
	if err != nil {
		spinner.Fail(err.Error())
		os.Exit(1)
	}
	spinner.Success()
	usage := extractUsageFromAPIResponse(apiJSON, headers)
	mr, err := parseModelResponse(apiJSON)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	return mr, usage, getModelCode(config)
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

func handleGlobalFlags(c *cli.Context) bool {
	if c.Bool("new") {
		fmt.Fprintln(os.Stderr, "--new is not implemented")
		os.Exit(1)
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
	mr, usage, code, config := run(promptPath, routine, c.Bool("debug"), c.Bool("no-model"))
	lang, _ := asNonEmptyStr(config["language"])
	changed := applyReplacements(mr.Edit.Replacements, ".", lang)
	msg := strings.TrimSpace(mr.Edit.CommitMessage)
	if msg == "" {
		msg = "Apply model edits"
	}
	pterm.Info.Printfln("Commit message: %s", msg)
	if cost := estimateCostUSD(code, usage); cost != nil {
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
		},
		Action: executeRunAction,
	}
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
