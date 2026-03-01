// Package main provides the Jisper CLI for model-driven code edits.
package main

import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "sort"
    "strings"

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
    schemaAny, schemaKeyExists := config["output_schema"]

    var schema map[string]any
    if !schemaKeyExists {
        schema = DefaultOutputSchema
    } else {
        if schemaAny == nil {
            return nil
        }
        custom, ok := schemaAny.(map[string]any)
        if !ok {
            return nil
        }
        if len(custom) == 0 {
            return nil
        }
        schema = custom
    }

    return map[string]any{
        "type": "json_schema",
        "json_schema": map[string]any{
            "name":   "response_schema",
            "strict": true,
            "schema": schema,
        },
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

func reportCost(modelCode string, usage Usage, config map[string]any) float64 {
    prices := GetModelPrices(config)
    p := 0
    if usage.PromptTokens != nil {
        p = *usage.PromptTokens
    }
    c := 0
    if usage.CompletionTokens != nil {
        c = *usage.CompletionTokens
    }
    if p > 0 || c > 0 {
        if est := EstimateCostUSD(modelCode, usage, prices); est != nil {
            pterm.Success.Printfln("$%.4f", *est)
            return *est
        }
    }
    return 0.0
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
    pl, pContent := buildPayload(config, srcMaterial, routine, strings.TrimSpace(task), endpointURL)
    return config, pl, pContent, apiKey, endpointURL
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

func run(
    path string, routine string, debug bool, noModel bool, task string,
) (ModelResponse, Usage, string, map[string]any, string, string) {
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
    reportCost(code, usage, cfg)
    return mr, usage, code, cfg, endpoint, key
}
