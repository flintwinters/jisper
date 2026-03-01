package main

import (
    "fmt"
    "os"
    "path/filepath"

    "github.com/go-git/go-git/v5"
    "github.com/go-git/go-git/v5/plumbing"
    "github.com/go-git/go-git/v5/plumbing/object"
)

func repoFromDir(baseDir string) (*git.Repository, bool) {
    r, err := git.PlainOpenWithOptions(baseDir, &git.PlainOpenOptions{DetectDotGit: true})
    if err != nil {
        return nil, false
    }
    return r, true
}

func stageAndCommit(r *git.Repository, changedFiles []string, message string) {
    w, err := r.Worktree()
    if err != nil {
        return
    }
    conf, _ := r.Config()
    for _, p := range dedupeKeepOrder(changedFiles) {
        rel, _ := filepath.Rel(conf.Core.Worktree, p)
        _, _ = w.Add(rel)
    }
    _, _ = w.Commit(message, &git.CommitOptions{
        Author: &object.Signature{Name: "Jisper", Email: "jisper@localhost"},
    })
}

func requireValidRepo(baseDir string) (string, bool) {
    repoRoot, ok := repoFromDir(baseDir)
    if !ok {
        fmt.Fprintln(os.Stderr,
            "Not a git repository. Run 'git init' to create one, or navigate to a directory inside a git repository.")
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
        fmt.Fprintln(os.Stderr, "Cannot undo: no parent commit exists (this is the initial commit).")
        fmt.Fprintln(os.Stderr, "Use 'git reset --soft HEAD~1' to uncommit while keeping changes.")
        return 1
    }
    _, code = runCmd(r, "git", "reset", "--hard", "HEAD~1")
    if code != 0 {
        fmt.Fprintln(os.Stderr, "Git reset failed. Check for uncommitted changes or repository corruption.")
        return 1
    }
    fmt.Println("Undid last commit (hard reset to HEAD~1)")
    return 0
}

func redoLastCommit(baseDir string) int {
    r, ok := repoFromDir(baseDir)
    if !ok {
        return 1
    }
    ref, err := r.Reference(plumbing.ReferenceName("ORIG_HEAD"), true)
    if err != nil {
        return 1
    }
    w, _ := r.Worktree()
    _ = w.Reset(&git.ResetOptions{Mode: git.HardReset, Commit: ref.Hash()})
    fmt.Printf("Redid undo (reset to ORIG_HEAD: %s)\n", ref.Hash().String()[:8])
    return 0
}

func initRepoIfMissing(baseDir string) *git.Repository {
    r, ok := repoFromDir(baseDir)
    if ok {
        return r
    }
    r, _ = git.PlainInit(baseDir, false)
    fmt.Printf("Initialized empty Git repository in %s/.git\n", baseDir)
    return r
}
