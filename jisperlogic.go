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
    "strconv"
    "strings"
    "time"

    "github.com/pterm/pterm"
    "go.yaml.in/yaml/v4"
)

type message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type payload struct {
    Model          string         `json:"model"`
    Messages       []message      `json:"messages"`
    ResponseFormat map[string]any `json:"response_format"`
    Provider       map[string]any `json:"provider,omitempty"`
}

type Usage struct {
    PromptTokens     *int `json:"prompt_tokens"`
    CompletionTokens *int `json:"completion_tokens"`
    TotalTokens      *int `json:"total_tokens"`
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

func buildPayload(
    promptConfig map[string]any,
    sourceText string,
    routine string,
    taskOverride string,
    endpointURL string,
) (payload, string) {
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

    promptContent := fmt.Sprintf(
        "SYSTEM PROMPT:\n%s\n\nTASK:\n%s\n\nSOURCE MATERIAL:\n%s",
        renderedSystem, renderedTask, sourceText)
    pl := payload{
        Model: modelCode,
        Messages: []message{
            {Role: "system", Content: systemInstruction},
            {Role: "user", Content: promptContent},
        },
        ResponseFormat: responseFormatFromConfig(promptConfig),
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
        leading := original[:len(original)-len(strings.TrimLeft(original, "\n"))]
        trailing := original[len(strings.TrimRight(original, "\n")):]
        replacedCore := strings.ReplaceAll(strippedOriginal, trimmedOld, newString)
        return leading + replacedCore + trailing, trimmedOld, true
    }

    return "", "", false
}

func performCreateFile(baseDir, filename, newString, language string) (string, bool) {
    targetPath := filepath.Join(baseDir, filename)
    _ = os.MkdirAll(filepath.Dir(targetPath), 0o755)
    fmt.Printf("\x1b[1m%s\x1b[0m", filename)
    printNumberedCombinedDiff("", newString, filename, language)
    if err := os.WriteFile(targetPath, []byte(newString), 0o644); err != nil {
        pterm.Error.Printfln("Failed to create file %s: %v", targetPath, err)
        return "", false
    }
    return targetPath, true
}

func processReplacement(
    r Replacement,
    baseDir string,
    language string,
    configPath string,
    allowedFiles []string,
    autoRetry bool,
) (string, bool) {
    filename := strings.TrimSpace(r.Filename)
    if filename == "" || !isFileAllowed(filename, allowedFiles) {
        return "", false
    }
    original, ok := readFileContent(baseDir, filename)
    if !ok {
        if strings.TrimSpace(r.OldString) == "" {
            return performCreateFile(baseDir, filename, r.NewString, language)
        }
        return "", false
    }
    if os.Getenv("DEBUG_JISPER") != "" {
        fmt.Printf("DEBUG: applying replacement for %s\n", filename)
    }
    updated, actualOld, applied := applyOneReplacement(original, r.OldString, r.NewString)
    if !applied {
        pterm.Warning.Printfln("old_string not found in %s; skipping", filename)
        if os.Getenv("DEBUG_JISPER") != "" {
            fmt.Printf("DEBUG: failed to find %s\n", r.OldString)
        }
        if !autoRetry {
            writeFailedOldStringToConfig(configPath, r.OldString)
        }
        return "", false
    }
    if os.Getenv("DEBUG_JISPER") != "" {
        fmt.Printf("DEBUG: applied replacement in %s using anchor: %s\n", filename, actualOld)
    }
    fmt.Printf("\x1b[1m%s\x1b[0m\n", filename)
    printNumberedCombinedDiff(original, updated, filename, language)
    targetPath := filepath.Join(baseDir, filename)
    if err := os.WriteFile(targetPath, []byte(updated), 0o644); err != nil {
        pterm.Error.Printfln("Failed to write file %s: %v", targetPath, err)
        return "", false
    }
    return targetPath, true
}

func applyReplacements(
    repls []Replacement,
    baseDir string,
    language string,
    configPath string,
    allowedFiles []string,
    autoRetry bool,
    config map[string]any,
    endpointURL string,
    apiKey string,
) []string {
    changed := []string{}
    for _, r := range repls {
        filename := strings.TrimSpace(r.Filename)
        if p, ok := processReplacement(r, baseDir, language, configPath, allowedFiles, autoRetry); ok {
            changed = append(changed, p)
            continue
        }
        if autoRetry {
            continue
        }
        pterm.Info.Printfln("Auto-retrying failed replacement for %s...", filename)
        original, _ := readFileContent(baseDir, filename)
        retryTask := fmt.Sprintf("The string replacement for '%s' failed. Fix the "+
            "old_string to match the file content exactly.\n\nFAILED OLD:\n%s\n\nFAILED NEW:\n%s",
            filename, r.OldString, r.NewString)
        pl, _ := buildPayload(config, original, "", retryTask, endpointURL)
        retryMr, _, _ := callModel(endpointURL, apiKey, pl, config)
        retryPaths := applyReplacements(
            retryMr.Edit.Replacements, baseDir, language, configPath,
            allowedFiles, true, config, endpointURL, apiKey)
        changed = append(changed, retryPaths...)
    }
    return changed
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
        value = strings.ReplaceAll(value, "    ", "\t")
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
            added := []*yaml.Node{nk, nv}
            root.Content = append(root.Content[:*insertAt], append(added, root.Content[*insertAt:]...)...)
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
    fmt.Printf("Build: %s", cmdStr)
    cmd := exec.Command("/bin/sh", "-c", cmdStr)
    var outB, errB bytes.Buffer
    cmd.Stdout = io.MultiWriter(os.Stdout, &outB)
    cmd.Stderr = io.MultiWriter(os.Stderr, &errB)
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

func newRetryableError(statusCode int, headers http.Header, endpointURL string, body []byte) retryableError {
    bodyPreview := truncateString(string(body), 1000)
    retryAfter := 0
    if ra := headers.Get("Retry-After"); ra != "" {
        if n, err := strconv.Atoi(strings.TrimSpace(ra)); err == nil {
            retryAfter = n
        }
    }
    return retryableError{
        statusCode:        statusCode,
        retryAfterSeconds: retryAfter,
        message: fmt.Sprintf(
            "API request to %s returned status %d. Response body: %s",
            endpointURL, statusCode, bodyPreview,
        ),
    }
}

func callOpenAICompatible(
    endpointURL string,
    apiKey string,
    pl payload,
) (map[string]any, http.Header, error) {
    b, err := json.Marshal(pl)
    if err != nil {
        return nil, nil, fmt.Errorf("failed to marshal request payload: %w", err)
    }
    b = bytes.ReplaceAll(b, []byte("\\u003c"), []byte("<"))
    b = bytes.ReplaceAll(b, []byte("\\u003e"), []byte(">"))
    b = bytes.ReplaceAll(b, []byte("\\u0026"), []byte("&"))
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
        return nil, resp.Header, newRetryableError(resp.StatusCode, resp.Header, endpointURL, body)
    }

    dec := json.NewDecoder(bytes.NewReader(body))
    dec.UseNumber()
    var apiJSON map[string]any
    if err := dec.Decode(&apiJSON); err != nil {
        bodyPreview := truncateString(string(body), 500)
        return nil, resp.Header, fmt.Errorf(
            "failed to parse API response as JSON from %s. Parse error: %w. Response body: %s",
            endpointURL,
            err,
            bodyPreview)
    }
    return apiJSON, resp.Header, nil
}

func parseModelResponse(apiJSON map[string]any) (ModelResponse, error) {
    choicesAny, ok := apiJSON["choices"]
    if !ok {
        return ModelResponse{}, fmt.Errorf(
            "API response missing 'choices' field. Response keys: %v",
            getKeys(apiJSON),
        )
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
        return ModelResponse{}, fmt.Errorf(
            "API response choices[0] missing 'message' field. Keys: %v",
            getKeys(choice0),
        )
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
        return ModelResponse{}, fmt.Errorf(
            "API response choices[0].message.content is not a string, got %T",
            contentAny,
        )
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

func getIssueContextAndTask(issue Issue, baseDir string) (string, string, bool) {
    context, ok := extractLinesAround(issue.Pos.Filename, issue.Pos.Line, baseDir, 40, 40)
    if !ok {
        return "", "", false
    }
    task := fmt.Sprintf("Fix this linting issue in %s at line %d.\nLinter: %s\nMessage: %s\nCode:\n%s",
        issue.Pos.Filename, issue.Pos.Line, issue.FromLinter, issue.Text, context)
    return context, task, true
}

func processIssue(
    issue Issue,
    config map[string]any,
    endpointURL string,
    apiKey string,
    promptPath string,
    debug bool,
    noModel bool,
    lang string,
) float64 {
    context, task, ok := getIssueContextAndTask(issue, ".")
    if !ok {
        pterm.Error.Printfln("Failed to read %s (line %d). File may not exist or line number is out of range.",
            issue.Pos.Filename, issue.Pos.Line)
        return 0
    }
    pl, content := buildPayload(config, context, "", task, endpointURL)
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
        []string{issue.Pos.Filename}, false, config, endpointURL, apiKey)
    msg := strings.TrimSpace(mr.Edit.CommitMessage)
    if msg == "" {
        msg = fmt.Sprintf("Fix %s issue in %s", issue.FromLinter, issue.Pos.Filename)
    }
    pterm.Info.Printfln("Commit: %s", msg)
    cost := reportCost(mc, usage, config)
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
