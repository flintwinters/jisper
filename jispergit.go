package main

import (
    "fmt"
    "os"
    "os/exec"

    "github.com/go-git/go-git/v5"
    "github.com/go-git/go-git/v5/config"
    "github.com/go-git/go-git/v5/plumbing"
    "github.com/go-git/go-git/v5/plumbing/object"
    "time"
)

func repoFromDir(baseDir string) (*git.Repository, bool) {
    r, err := git.PlainOpenWithOptions(baseDir, &git.PlainOpenOptions{DetectDotGit: true})
    if err != nil {
        return nil, false
    }
    return r, true
}

func createAndCheckoutBranch(r *git.Repository,
	branchName string) {
	w, err := r.Worktree()
	if err != nil {
		return
	}
	head, err := r.Head()
	if err != nil {
		return
	}
	name := plumbing.NewBranchReferenceName(branchName)
	err = w.Checkout(&git.CheckoutOptions{
		Hash:   head.Hash(),
		Branch: name,
		Create: true,
	})
	if err != nil {
		_ = w.Checkout(&git.CheckoutOptions{
			Branch: name,
		})
	}
}

func stageAndCommit(
	r *git.Repository,
	changedFiles []string,
	message string,
) {
    w, err := r.Worktree()
    if err != nil {
        return
    }
    configObj, _ := r.Config()
    author, _ := r.ConfigScoped(config.LocalScope)
    if os.Getenv("DEBUG_JISPER") != "" {
        fmt.Printf("DEBUG: Git Author config: %+v\n", author)
        fmt.Printf("DEBUG: Core worktree: %s\n", configObj.Core.Worktree)
    }
    if author == nil {
        author = configObj
    }
    userName := author.User.Name
    userEmail := author.User.Email
    if userName == "" || userEmail == "" {
        userName = "Jisper"
        userEmail = "jisper@example.com"
    }
    _, _ = w.Commit(message, &git.CommitOptions{
        Author: &object.Signature{
            Name:  userName,
            Email: userEmail,
            When:  time.Now(),
        },
    })
}

func runCmd(r *git.Repository, name string, arg ...string) (string, int) {
    w, err := r.Worktree()
    if err != nil {
        return "", 1
    }
    cmd := exec.Command(name, arg...)
    cmd.Dir = w.Filesystem.Root()
    if os.Getenv("DEBUG_JISPER") != "" {
        fmt.Printf("DEBUG: Running %s %v in %s\n", name, arg, cmd.Dir)
    }
    out, err := cmd.CombinedOutput()
    if err != nil {
        if ee, ok := err.(*exec.ExitError); ok {
            return string(out), ee.ExitCode()
        }
        return err.Error(), 1
    }
    return string(out), 0
}

func requireValidRepo(baseDir string) (*git.Repository, bool) {
    repo, ok := repoFromDir(baseDir)
    if !ok {
        fmt.Fprintln(os.Stderr, "Not a git repository.")
        return nil, false
    }
    _, code := runCmd(repo, "git", "rev-parse", "--verify", "HEAD")
    if code != 0 {
        fmt.Fprintln(os.Stderr, "No valid commits in repository.")
        return nil, false
    }
    return repo, true
}

func undoLastCommit(baseDir string) int {
    repo, ok := requireValidRepo(baseDir)
    if !ok {
        return 1
    }
    _, code := runCmd(repo, "git", "rev-parse", "--verify", "HEAD~1")
    if code != 0 {
        fmt.Fprintln(os.Stderr, "Cannot undo: no parent commit exists.")
        return 1
    }
    _, code = runCmd(repo, "git", "reset", "--hard", "HEAD~1")
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
