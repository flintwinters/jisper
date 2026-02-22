package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

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
		fmt.Fprintln(os.Stderr, "Not a git repository. Run 'git init' to create one, or navigate to a directory inside a git repository.")
		return "", false
	}
	_, code := runCmd(repoRoot, "git", "rev-parse", "--verify", "HEAD")
	if code != 0 {
		fmt.Fprintln(os.Stderr, "No valid commits in repository. Create at least one commit before using undo/redo.")
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
		fmt.Fprintln(os.Stderr, "Cannot undo: no parent commit exists (this is the initial commit). Use 'git reset --soft HEAD~1' to uncommit while keeping changes.")
		return 1
	}
	_, code = runCmd(repoRoot, "git", "reset", "--hard", "HEAD~1")
	if code != 0 {
		fmt.Fprintln(os.Stderr, "Git reset failed. Check for uncommitted changes or repository corruption.")
		return 1
	}
	fmt.Println("Undid last commit (hard reset to HEAD~1)")
	return 0
}

func redoLastCommit(baseDir string) int {
	repoRoot, ok := requireValidRepo(baseDir)
	if !ok {
		return 1
	}
	origPath := filepath.Join(repoRoot, ".git", "ORIG_HEAD")
	if _, err := os.Stat(origPath); err != nil {
		fmt.Fprintln(os.Stderr, "Cannot redo: .git/ORIG_HEAD not found. ORIG_HEAD is only set after certain git operations like reset.", "Run 'git reflog' to find the commit you want to restore.")
		return 1
	}
	out, _ := runCmd(repoRoot, "git", "rev-parse", "ORIG_HEAD")
	sha := strings.TrimSpace(out)
	if sha == "" {
		fmt.Fprintln(os.Stderr, "Cannot redo: ORIG_HEAD is empty. Run 'git reflog' to find the commit you want to restore.")
		return 1
	}
	_, code := runCmd(repoRoot, "git", "reset", "--hard", sha)
	if code != 0 {
		fmt.Fprintf(os.Stderr, "Git reset to ORIG_HEAD (%s) failed. The reflog may have more information.\n", sha)
		return 1
	}
	fmt.Printf("Redid undo (reset to ORIG_HEAD: %s)\n", sha[:min(8, len(sha))])
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
