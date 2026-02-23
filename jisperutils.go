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

func GetAPIKey(_ string, keyVar string) string {
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

const (
	DefaultFallbackInputUSDPer1M  = 5.0
	DefaultFallbackOutputUSDPer1M = 15.0
)

func EstimateCostUSD(modelCode string, usage Usage, prices map[string]Prices) *float64 {
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

func GetModelPrices(config map[string]any) map[string]Prices {
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
