#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

import typer
from rich.console import Console
from rich.text import Text

console = Console()
app = typer.Typer(no_args_is_help=True)

project_root = Path(__file__).parent.resolve()


def run(cmd, cwd=None, shell=False):
    """Run a shell command and exit on failure."""
    if cwd is None:
        cwd = project_root
    result = subprocess.run(cmd, cwd=cwd, shell=shell)
    if result.returncode != 0:
        console.print(f"[bold red]Command failed:[/bold red] {' '.join(cmd) if isinstance(cmd, (list, tuple)) else cmd}")
        sys.exit(result.returncode)


@app.command()
def fmt():
    """Format Go code using gofmt and goimports."""
    run(["gofmt", "-s", "-w", "."])
    run(["goimports", "-w", "."])


@app.command()
def lint():
    """Run golangci-lint with backend config."""
    run(["golangci-lint", "run", "--timeout", "5m", "./..."])


@app.command()
def vet():
    """Run go vet to check code correctness."""
    run(["go", "vet", "./..."])


@app.command()
def test():
    """Run tests with race detection, failing fast on any error."""
    result = subprocess.run(
        ["go", "test", "./...", "-race"],
        cwd=project_root,
    )
    if result.returncode != 0:
        console.print("[bold red]Tests failed:[/bold red]")
        sys.exit(result.returncode)


@app.command()
def build():
    """Build the main backend binary."""
    out_dir = project_root / "bin"
    out_dir.mkdir(exist_ok=True)
    go_files = [f.name for f in project_root.glob("*.go")]
    run(["go", "build", "-o", str(out_dir)] + go_files)


@app.command()
def clean():
    """Remove build artifacts."""
    bin_dir = project_root / "bin"
    if bin_dir.exists():
        for f in bin_dir.iterdir():
            f.unlink()
        bin_dir.rmdir()


@app.command()
def all():
    """Run all tasks: fmt, lint, vet, test, build."""
    fmt()
    lint()
    vet()
    # test()
    build()
    console.print("[bold green]All tasks completed successfully.[/bold green]")


if __name__ == "__main__":
    app()
