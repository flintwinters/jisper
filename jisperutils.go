package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func resolveRoutineTask(config map[string]any, routineName string) (string, bool) {
	name := strings.TrimSpace(routineName)
	if name == "" {
		return "", false
	}
	routines, ok := config["routines"].(map[string]any)
	if !ok {
		return "", false
	}
	return asNonEmptyStr(routines[name])
}

func GetExtLexerMapping() map[string]string {
	return map[string]string{
		".py": "python", ".json": "json", ".json5": "json",
		".go": "go", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
		".diff": "diff", ".patch": "diff", ".toml": "toml",
		".sh": "bash", ".bash": "bash", ".js": "javascript",
		".ts": "typescript", ".html": "html", ".css": "css", ".sql": "sql",
		".rs": "rust", ".c": "c", ".cpp": "cpp", ".h": "cpp",
	}
}

func ResolveEndpointAndAPIKey(config map[string]any) (string, string) {
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
	return endpointURL, keyVar
}

func GetAPIKey(endpointURL, keyVar string) string {
	apiKey := strings.TrimSpace(os.Getenv(keyVar))
	if apiKey == "" {
		errMsg := fmt.Sprintf(
			"API key not found: environment variable %s is not set or empty. "+
				"Set it with: export %s=your-api-key",
			keyVar, keyVar)
		fmt.Fprintf(os.Stderr, "%s\n", errMsg)
		os.Exit(1)
	}
	return apiKey
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
		return filterEmpty(ss)
	}
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		if s, ok := asNonEmptyStr(x); ok {
			out = append(out, s)
		}
	}
	return out
}

func filterEmpty(ss []string) []string {
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		if t := strings.TrimSpace(s); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func dedupeKeepOrder(xs []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		if x == "" || seen[x] {
			continue
		}
		seen[x] = true
		out = append(out, x)
	}
	return out
}

func flatMapStr(xs []string, fn func(string) []string) []string {
	out := make([]string, 0)
	for _, x := range xs {
		out = append(out, fn(x)...)
	}
	return out
}

func readFileContent(baseDir, filename string) (string, bool) {
	path := filepath.Join(baseDir, filename)
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return string(b), true
}

func toRel(base, target string) string {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return target
	}
	return rel
}

func buildJinjaContext(cfg map[string]any, source, task, sys string) map[string]any {
	m := map[string]any{"source_text": source, "task": task, "system_prompt": sys}
	for k, v := range cfg {
		m[k] = v
	}
	if v, ok := cfg["build_stdout"]; ok {
		m["build_stdout"] = v
	}
	if v, ok := cfg["build_stderr"]; ok {
		m["build_stderr"] = v
	}
	if v, ok := cfg["success"]; ok {
		m["success"] = v
	}
	if v, ok := cfg["error"]; ok {
		m["error"] = v
	}
	return m
}

func render(tmpl string, ctx map[string]any) string {
	for k, v := range ctx {
		tmpl = strings.ReplaceAll(tmpl, "{{"+k+"}}", fmt.Sprintf("%v", v))
	}
	return tmpl
}

func resolveSystemPrompt(config map[string]any) string {
	sysRaw, _ := asNonEmptyStr(config["system_prompt"])
	projPrompt, _ := asNonEmptyStr(config["project"])
	if projPrompt != "" {
		return sysRaw + "\n\n" + projPrompt
	}
	return sysRaw
}

func resolveUserTask(config map[string]any, routineName string) string {
	if t, ok := resolveRoutineTask(config, routineName); ok {
		return t
	}
	task, _ := asNonEmptyStr(config["task"])
	return task
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

func coerceFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	case string:
		s := strings.TrimSpace(x)
		if s == "" {
			return 0, false
		}
		n := json.Number(s)
		f, err := n.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}

type IssuePos struct {
	Filename string `json:"Filename"`
	Offset   int    `json:"Offset"`
	Line     int    `json:"Line"`
	Column   int    `json:"Column"`
}

type Issue struct {
	FromLinter           string   `json:"FromLinter"`
	Text                 string   `json:"Text"`
	Severity             string   `json:"Severity"`
	SourceLines          []string `json:"SourceLines"`
	Pos                  IssuePos `json:"Pos"`
	ExpectNoLint         bool     `json:"ExpectNoLint"`
	ExpectedNoLintLinter string   `json:"ExpectedNoLintLinter"`
}

type IssuesFile struct {
	Issues []Issue `json:"Issues"`
}

func extractLinesAround(filename string, lineNum int, baseDir string, before int, after int) (string, bool) {
	content, ok := readFileContent(baseDir, filename)
	if !ok {
		return "", false
	}
	lines := strings.Split(content, "\n")
	idx := lineNum - 1
	if idx < 0 || idx >= len(lines) {
		return "", false
	}
	start := idx - before
	if start < 0 {
		start = 0
	}
	end := idx + after + 1
	if end > len(lines) {
		end = len(lines)
	}
	return strings.Join(lines[start:end], "\n"), true
}

func loadIssuesFile(path string) (IssuesFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return IssuesFile{}, err
	}
	var issues IssuesFile
	if err := json.Unmarshal(b, &issues); err != nil {
		return IssuesFile{}, err
	}
	return issues, nil
}
