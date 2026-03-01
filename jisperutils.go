package main

import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "sort"
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

func truncateString(s string, maxLen int) string {
    if len(s) > maxLen {
        return s[:maxLen] + "... (truncated)"
    }
    return s
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

func EstimateCostUSD(modelCode string, usage Usage, prices map[string]Prices) *float64 {
    if usage.PromptTokens == nil || usage.CompletionTokens == nil {
        return nil
    }
    p, ok := prices[modelCode]
    if !ok {
        p = Prices{InUSDPer1M: DefaultFallbackInputUSDPer1M, OutUSDPer1M: DefaultFallbackOutputUSDPer1M}
    }
    inC := float64(*usage.PromptTokens) / 1_000_000 * p.InUSDPer1M
    outC := float64(*usage.CompletionTokens) / 1_000_000 * p.OutUSDPer1M
    cost := inC + outC
    return &cost
}

func isFileAllowed(filename string, allowedFiles []string) bool {
    if len(allowedFiles) == 0 {
        return true
    }
    normalized := filepath.Clean(filename)
    for _, allowed := range allowedFiles {
        if filepath.Clean(allowed) == normalized {
            return true
        }
    }
    return false
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

func getKeys(m map[string]any) []string {
    keys := make([]string, 0, len(m))
    for k := range m {
        keys = append(keys, k)
    }
    return keys
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
