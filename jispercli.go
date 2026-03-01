package main

import (
    "fmt"
    "os"
    "path/filepath"
    "strings"

    "github.com/hexops/gotextdiff"
    "github.com/hexops/gotextdiff/myers"
    "github.com/hexops/gotextdiff/span"
    "github.com/pterm/pterm"
    cli "github.com/urfave/cli"
)

var (
    styleKeyword = pterm.NewStyle(pterm.FgLightCyan)
    styleNumber  = pterm.NewStyle(pterm.FgLightMagenta)
    styleString  = pterm.NewStyle(pterm.FgLightYellow)
    styleComment = pterm.NewStyle(pterm.FgGray, pterm.Italic)
)

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

type lexerState struct {
    quote                  byte
    escape                 bool
    wordStart, numberStart int
    keywords               map[string]bool
    code                   string
    out                    *strings.Builder
    flushWord, flushNumber func(int)
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

func formatCombinedDiffLines(oldText, newText, filename, language string, contextLines int) []string {
    edits := myers.ComputeEdits(span.URIFromPath("a"), oldText, newText)
    diff := gotextdiff.ToUnified("a", "b", oldText, edits)
    lexer := guessLexer(oldText+"\n"+newText, filename, language)
    styleDel := pterm.NewStyle(pterm.FgLightRed)
    styleIns := pterm.NewStyle(pterm.FgLightGreen)
    styleDelPre := pterm.NewStyle(pterm.FgLightRed)
    styleInsPre := pterm.NewStyle(pterm.FgLightGreen)
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
                    s = styledLineNumber(&newLn, styleIns, 4) + 
                        styleInsPre.Sprint(" + ") + highlightLine(content, lexer)
                    newLn++
                case gotextdiff.Delete:
                    s = styledLineNumber(&oldLn, styleDel, 4) + 
                        styleDelPre.Sprint(" - ") + highlightLine(content, lexer)
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
        fmt.Fprintf(os.Stderr,
            "%s already exists; refusing to overwrite. Remove it first if you want to create a new one.\n",
            DefaultPromptFile)
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

func handleRunIssueAction(c *cli.Context, promptPath string) error {
    issuesFile := c.String("issues")
    issues, err := loadIssuesFile(issuesFile)
    if err != nil {
        os.Exit(1)
    }
    runIssues(issues, promptPath, c.Bool("debug"), c.Bool("no-model"))
    return nil
}

func runActionHandler(c *cli.Context) error {
    handleGlobalFlags(c)
    promptPath := c.String("prompt")
    if c.Bool("build") {
        config, _ := loadPromptFile(promptPath)
        runBuildStep(config, promptPath)
        return nil
    }
    if c.IsSet("issues") {
        return handleRunIssueAction(c, promptPath)
    }
    routine := ""
    if c.NArg() > 0 {
        routine = c.Args().Get(0)
    }
    mr, usage, mc, config, endpointURL, apiKey := run(promptPath, routine, c.Bool("debug"), c.Bool("no-model"), c.String("task"))
    if c.Bool("debug") {
        fmt.Printf("DEBUG: promptPath=%s routine=%s usage=%+v model_config=%+v\n", promptPath, routine, usage, mc)
        fmt.Printf("DEBUG: full_files=%v\n", config["full_files"])
    }
    lang, _ := asNonEmptyStr(config["language"])
    includes := resolveIncludedFiles(config, ".")
    changed := applyReplacements(
        mr.Edit.Replacements, ".", lang, promptPath, 
        includes.SourceFiles, c.Bool("auto-retry"), config, endpointURL, apiKey)
    msg := strings.TrimSpace(mr.Edit.CommitMessage)
    if msg == "" {
        msg = "Apply model edits"
    }
    repo := initRepoIfMissing(".")
    stageAndCommit(repo, changed, msg)
    runBuildStep(config, promptPath)
    return nil
}

func main() {
    app := &cli.App{
        Name: "jisper", Usage: "CLI for Jisper",
        Flags: []cli.Flag{
            cli.StringFlag{Name: "prompt, p", Value: DefaultPromptFile},
            cli.BoolFlag{Name: "new"}, cli.BoolFlag{Name: "undo, u"},
            &cli.BoolFlag{Name: "redo"}, &cli.BoolFlag{Name: "build"},
            &cli.BoolFlag{Name: "debug"}, &cli.BoolFlag{Name: "no-model"},
            &cli.BoolFlag{Name: "auto-retry"},
            cli.StringFlag{Name: "issues", Value: "issues.json", Usage: "Path to issues JSON file"},
            cli.StringFlag{Name: "task, t", Usage: "Task to perform (overrides config task and routine)"},
        },
        Action: runActionHandler,
    }
    if err := app.Run(os.Args); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}
