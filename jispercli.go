package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pterm/pterm"
	cli "github.com/urfave/cli"
)

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
		promptPath = c.Args().First()
	}
	if _, err := os.Stat(c.Args().Get(0)); err == nil {
		promptPath = c.Args().Get(0)
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
	runBuildStep(config, promptPath)
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
