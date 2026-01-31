import requests
import os
import json5 as json
import argparse
import difflib
from pathlib import Path
import re
import git
from rich import print
from rich.console import Console
from rich.text import Text
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn


console = Console()

DEFAULT_PROMPT_FILE = "prompt.json"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_URL = "https://api.openai.com/v1/chat/completions"

def get_base_config_value(config: dict, key: str, default: str) -> str:
    v = config.get(key)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return default

def get_model_code(config: dict) -> str:
    return get_base_config_value(config, "model", DEFAULT_MODEL)

def get_api_key_env_var_name(config: dict) -> str:
    return get_base_config_value(config, "api_key_env_var", DEFAULT_API_KEY_ENV_VAR)

def get_endpoint_url(config: dict) -> str:
    return get_base_config_value(config, "endpoint", DEFAULT_URL)

def get_api_key_from_env(env_var_name: str) -> str | None:
    if not env_var_name:
        return None
    v = os.getenv(env_var_name)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None

def load_prompt_file(path: Path) -> dict:
    """Load the prompt/config JSON5 file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_and_concatenate_files(file_list):
    contents = []
    for filename in file_list:
        p = Path(filename)
        if not p.exists():
            print(f"[red]Missing input file: {filename}[/red]")
            continue
        contents.append(f"--- FILENAME: {p.name} ---\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(contents)

def build_payload(prompt_config: dict, source_text: str):
    system_instruction = prompt_config.get("system_instruction", "You are a helpful assistant.")
    system_prompt = prompt_config["system_prompt"]
    user_task = prompt_config["task"]
    schema = prompt_config["output_schema"]
    model_code = get_model_code(prompt_config)

    prompt_content = f"SYSTEM PROMPT:\n{system_prompt}\n\nTASK:\n{user_task}\n\nSOURCE MATERIAL:\n{source_text}"

    payload = {
        "model": model_code,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_content},
        ],
    }

    if schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": schema,
            },
        }

    return payload

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_PROMPT_FILE)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_prompt_file(config_path)

    # Base-level config fields
    endpoint_url = get_endpoint_url(config)
    api_key_env_var = get_api_key_env_var_name(config)
    api_key = get_api_key_from_env(api_key_env_var)

    # Extract file list from prompt.json
    files_to_read = config["input_files"]
    concatenated_text = read_and_concatenate_files(files_to_read)

    headers = {
        "Authorization": f"Bearer {api_key or ''}",
        "Content-Type": "application/json",
    }

    payload = build_payload(config, concatenated_text)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="Waiting for model...", total=None)
        response = requests.post(endpoint_url, headers=headers, json=payload)

    return json.loads(response.json()["choices"][0]["message"]["content"])

def tokenize_for_intraline_diff(s: str) -> list[str]:
    parts = re.findall(r"\s+|[A-Za-z0-9_]+|[^\w\s]", s, flags=re.UNICODE)
    return parts


def is_trivial_separator_token(tok: str) -> bool:
    if not tok:
        return True
    if tok.isspace():
        return True
    return tok in {".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "=", "->"}


def merge_change_opcodes(
    opcodes: list[tuple[str, int, int, int, int]],
    a_tokens: list[str],
    b_tokens: list[str],
) -> list[tuple[str, int, int, int, int]]:
    merged: list[tuple[str, int, int, int, int]] = []

    def can_bridge_equal(i1: int, i2: int, j1: int, j2: int) -> bool:
        if i2 - i1 != j2 - j1:
            return False
        eq_a = a_tokens[i1:i2]
        eq_b = b_tokens[j1:j2]
        if eq_a != eq_b:
            return False
        return all(map(is_trivial_separator_token, eq_a))

    def push(tag: str, i1: int, i2: int, j1: int, j2: int):
        if not merged:
            merged.append((tag, i1, i2, j1, j2))
            return
        ptag, pi1, pi2, pj1, pj2 = merged[-1]
        if ptag == tag:
            merged[-1] = (ptag, pi1, i2, pj1, j2)
            return
        merged.append((tag, i1, i2, j1, j2))

    i = 0
    while i < len(opcodes):
        tag, i1, i2, j1, j2 = opcodes[i]

        if tag == "equal":
            push(tag, i1, i2, j1, j2)
            i += 1
            continue

        start_i1, start_j1 = i1, j1
        end_i2, end_j2 = i2, j2
        i += 1

        while i < len(opcodes):
            ntag, ni1, ni2, nj1, nj2 = opcodes[i]
            if ntag in {"delete", "insert", "replace"}:
                end_i2, end_j2 = ni2, nj2
                i += 1
                continue

            if ntag == "equal" and can_bridge_equal(ni1, ni2, nj1, nj2):
                end_i2, end_j2 = ni2, nj2
                i += 1
                continue

            break

        push("replace", start_i1, end_i2, start_j1, end_j2)

    return merged


def rich_inline_diff(old: str, new: str) -> Text:
    """Rich Text with inline diff highlighting (red deletions, green inserts).

    Uses a token-based matcher and merges nearby change hunks to avoid overly
    fragmented red/green segments.
    """
    t = Text()
    a_tokens = tokenize_for_intraline_diff(old)
    b_tokens = tokenize_for_intraline_diff(new)

    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    opcodes = merge_change_opcodes(sm.get_opcodes(), a_tokens, b_tokens)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            t.append("".join(a_tokens[i1:i2]))
            continue

        if tag == "delete":
            t.append("".join(a_tokens[i1:i2]), style="red")
            continue

        if tag == "insert":
            t.append("".join(b_tokens[j1:j2]), style="green")
            continue

        if tag == "replace":
            t.append("".join(a_tokens[i1:i2]), style="red")
            t.append("".join(b_tokens[j1:j2]), style="green")
            continue

    return t


def find_substring_start_line(haystack: str, needle: str) -> int | None:
    """Return 1-based line number where needle begins in haystack, or None."""
    if not needle:
        return None
    idx = haystack.find(needle)
    if idx < 0:
        return None
    return haystack[:idx].count("\n") + 1


def compute_replacement_line_numbers(original: str, matched_old: str, new_string: str) -> tuple[int, int] | None:
    """Compute starting line numbers (old_start, new_start) for the replacement in a file."""
    old_start = find_substring_start_line(original, matched_old)
    if old_start is None:
        return None
    new_start = old_start
    return (old_start, new_start)


def format_numbered_unified_diff(
    old_text: str,
    new_text: str,
    *,
    fromfile: str,
    tofile: str,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
) -> list[str]:
    old_lines = old_text.splitlines(keepends=False)
    new_lines = new_text.splitlines(keepends=False)

    header = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=fromfile,
            tofile=tofile,
            fromfiledate=f"line {from_start}",
            tofiledate=f"line {to_start}",
            lineterm="",
            n=context_lines,
        )
    )

    new_ln = to_start

    def fmt(ln: int | None) -> str:
        return f"{ln:>6}" if ln is not None else "      "

    def format_body_line(prefix: str, text: str) -> str:
        nonlocal new_ln
        if prefix == " ":
            out = f"{fmt(new_ln)}  {prefix}{text}"
            new_ln += 1
            return out
        if prefix == "-":
            return f"{fmt(None)}  {prefix}{text}"
        if prefix == "+":
            out = f"{fmt(new_ln)}  {prefix}{text}"
            new_ln += 1
            return out
        return f"{fmt(None)}  {prefix}{text}"

    out_lines: list[str] = []
    for line in header:
        if line.startswith("---") or line.startswith("+++"):
            out_lines.append(line)
            continue
        if line.startswith("@@"):
            out_lines.append(line)
            continue
        if not line:
            out_lines.append(line)
            continue

        prefix = line[:1]
        text = line[1:]
        out_lines.append(format_body_line(prefix, text))

    return out_lines


def print_numbered_unified_diff(
    old_text: str,
    new_text: str,
    *,
    fromfile: str,
    tofile: str,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
    title: str | None = None,
):
    if title:
        console.print(f"\n[bold]{title}[/bold]")

    lines = format_numbered_unified_diff(
        old_text,
        new_text,
        fromfile=fromfile,
        tofile=tofile,
        from_start=from_start,
        to_start=to_start,
        context_lines=context_lines,
    )

    if not lines:
        console.print("[yellow](no diff; contents are identical)[/yellow]")
        return

    console.print(Syntax("\n".join(lines), "diff", theme="ansi_dark", line_numbers=False))


def format_numbered_combined_diff(
    old_text: str,
    new_text: str,
    *,
    fromfile: str,
    tofile: str,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
) -> list[Text]:
    """Return a single combined diff with line numbers and intra-line highlights.

    Output is based on unified diff lines, but paired '-'/'+' lines are merged into
    a single '~' line with rich inline highlighting.
    """
    diff_lines = unified_diff_lines(
        old_text,
        new_text,
        fromfile=fromfile,
        tofile=tofile,
        context_lines=context_lines,
    )

    out: list[Text] = []
    new_ln = to_start

    def fmt_ln(ln: int | None) -> str:
        return f"{ln:>6}" if ln is not None else "      "

    def push_header(line: str):
        out.append(Text(line))

    def push_context(text: str):
        nonlocal new_ln
        out.append(Text(f"{fmt_ln(new_ln)}    {text}"))
        new_ln += 1

    def push_delete(text: str):
        out.append(Text(f"{fmt_ln(None)}  - ", style="bold red") + rich_inline_diff(text, ""))

    def push_insert(text: str):
        nonlocal new_ln
        out.append(Text(f"{fmt_ln(new_ln)}  + ", style="bold green") + rich_inline_diff("", text))
        new_ln += 1

    def push_replace(old_line: str, new_line: str):
        nonlocal new_ln
        merged_prefix = Text(f"{fmt_ln(new_ln)}  ~ ", style="bold")
        out.append(merged_prefix + rich_inline_diff(old_line, new_line))
        new_ln += 1

    def is_hunk_header(line: str) -> bool:
        return line.startswith("@@")

    def is_file_header(line: str) -> bool:
        return line.startswith("---") or line.startswith("+++")

    pending_minus: str | None = None

    for line in diff_lines:
        if is_file_header(line) or is_hunk_header(line):
            if pending_minus is not None:
                push_delete(pending_minus[1:])
                pending_minus = None
            push_header(line)
            continue

        if not line:
            continue

        prefix = line[:1]
        body = line[1:]

        if prefix == " ":
            if pending_minus is not None:
                push_delete(pending_minus[1:])
                pending_minus = None
            push_context(body)
            continue

        if prefix == "-":
            if pending_minus is not None:
                push_delete(pending_minus[1:])
            pending_minus = line
            continue

        if prefix == "+":
            if pending_minus is not None:
                push_replace(pending_minus[1:], body)
                pending_minus = None
            else:
                push_insert(body)
            continue

        if pending_minus is not None:
            push_delete(pending_minus[1:])
            pending_minus = None
        push_header(line)

    if pending_minus is not None:
        push_delete(pending_minus[1:])

    return out


def print_numbered_combined_diff(
    old_text: str,
    new_text: str,
    *,
    fromfile: str,
    tofile: str,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
    title: str | None = None,
):
    if title:
        console.print(f"\n[bold]{title}[/bold]")

    lines = format_numbered_combined_diff(
        old_text,
        new_text,
        fromfile=fromfile,
        tofile=tofile,
        from_start=from_start,
        to_start=to_start,
        context_lines=context_lines,
    )

    if not lines:
        console.print("[yellow](no diff; contents are identical)[/yellow]")
        return

    for t in lines:
        console.print(t)

def unified_diff_lines(
    old_text: str,
    new_text: str,
    *,
    fromfile: str,
    tofile: str,
    context_lines: int = 3,
) -> list[str]:
    old_lines = old_text.splitlines(keepends=False)
    new_lines = new_text.splitlines(keepends=False)
    return list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
            n=context_lines,
        )
    )


def print_intraline_diff(
    old_text: str,
    new_text: str,
    *,
    fromfile: str,
    tofile: str,
    context_lines: int = 3,
    title: str | None = None,
):
    """Backward-compatible wrapper: print the combined diff view."""
    print_numbered_combined_diff(
        old_text,
        new_text,
        fromfile=fromfile,
        tofile=tofile,
        from_start=1,
        to_start=1,
        context_lines=context_lines,
        title=title,
    )


def print_change_preview(filename: str, old_string: str, new_string: str, original: str, updated: str):
    console.print(f"\n[bold]Preview for[/bold] [white]{filename}[/white]")

    line_info = compute_replacement_line_numbers(original, old_string, new_string)
    old_start = line_info[0] if line_info else 1
    new_start = line_info[1] if line_info else 1

    print_numbered_combined_diff(
        old_string,
        new_string,
        fromfile=f"old_string ({filename})",
        tofile=f"new_string ({filename})",
        from_start=old_start,
        to_start=new_start,
        context_lines=2,
        title="Replacement text (combined diff)",
    )


def apply_replacements(replacements, base_dir: Path | None = None) -> list[Path]:
    """Apply {filename, old_string, new_string} edits to files on disk.

    Returns a list of Paths that were modified.
    """
    base_dir = base_dir or Path.cwd()
    changed: list[Path] = []

    for i, r in enumerate(replacements or []):
        filename = r.get("filename")
        old_string = r.get("old_string")
        new_string = r.get("new_string")

        if not filename:
            print(f"[red]Replacement #{i} missing filename; skipping[/red]")
            continue
        if old_string is None or new_string is None:
            print(f"[red]Replacement for {filename} missing old_string/new_string; skipping[/red]")
            continue

        target_path = (base_dir / filename).resolve()
        if not target_path.exists():
            print(f"[red]Target file not found: {target_path}[/red]")
            continue

        original = target_path.read_text(encoding="utf-8")

        def apply_if_found(haystack: str, needle: str) -> str | None:
            if needle in haystack:
                return haystack.replace(needle, new_string)
            return None

        # 1) Exact match
        updated = apply_if_found(original, old_string)

        # 2) Retry with trimmed old_string (common failure: extra leading/trailing newlines)
        matched_old = old_string
        if updated is None:
            trimmed_old = old_string.strip()
            if trimmed_old and trimmed_old != old_string:
                updated = apply_if_found(original, trimmed_old)
                if updated is not None:
                    matched_old = trimmed_old

        # 3) Retry on stripped file content; preserve original outer whitespace
        if updated is None:
            stripped_original = original.strip()
            trimmed_old = old_string.strip()
            if stripped_original and trimmed_old and trimmed_old in stripped_original:
                leading = original[: len(original) - len(original.lstrip())]
                trailing = original[len(original.rstrip()):]
                replaced_core = stripped_original.replace(trimmed_old, new_string)
                updated = f"{leading}{replaced_core}{trailing}"
                matched_old = trimmed_old

        if updated is None:
            print(f"[yellow]old_string not found in {filename}; skipping[/yellow]")
            # Still show what it *wanted* to do, for debugging.
            print_numbered_unified_diff(
                old_string,
                new_string,
                fromfile=f"old_string ({filename})",
                tofile=f"new_string ({filename})",
                from_start=1,
                to_start=1,
                context_lines=2,
                title="Replacement text (numbered unified diff)",
            )
            continue

        if updated == original:
            print(f"[yellow]No changes applied to {filename} (replacement produced identical content)[/yellow]")
            continue

        # Show very clear diff previews before writing.
        print_change_preview(filename, matched_old, new_string, original, updated)

        target_path.write_text(updated, encoding="utf-8")
        changed.append(target_path)
        print(f"[green]Applied replacement to {filename}[/green]")

    return changed


def print_model_change_notes(model_output: dict):
    """Print the model response and edit explanation (if present)."""
    if not isinstance(model_output, dict):
        return

    edits = model_output.get("edit") or {}
    print("\nEdit explanation:")
    print(edits.get("explanation"))


def extract_commit_message(model_output: dict) -> str | None:
    if not isinstance(model_output, dict):
        return None

    msg = model_output.get("commit_message")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()

    edits = model_output.get("edit") or {}
    msg = edits.get("commit_message")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()

    return None


def repo_from_dir(base_dir: Path) -> git.Repo | None:
    if (base_dir / ".git").exists():
        return git.Repo(base_dir)

    current = base_dir
    while current != current.parent:
        if (current / ".git").exists():
            return git.Repo(current)
        current = current.parent

    return None


def stage_and_commit(repo: git.Repo, changed_files: list[Path], message: str) -> str | None:
    repo_root = Path(repo.working_tree_dir or ".").resolve()
    relpaths = list(map(lambda p: str(p.resolve().relative_to(repo_root)), changed_files))
    relpaths = list(filter(lambda s: bool(s and s.strip()), relpaths))

    if not relpaths:
        return None

    repo.index.add(relpaths)
    repo.index.commit(message)
    return message


if __name__ == "__main__":
    response = run()
    print_model_change_notes(response or {})

    edits = (response or {}).get("edit", {})
    replacements = edits.get("replacements", [])
    changed_files = apply_replacements(replacements)

    commit_message = extract_commit_message(response or {})
    if not commit_message:
        commit_message = "Apply model edits"

    repo = repo_from_dir(Path.cwd())
    if repo is None:
        print("[yellow]Not a git repository; skipping commit[/yellow]")

    if repo is not None and changed_files:
        committed_message = stage_and_commit(repo, changed_files, commit_message)
        if committed_message:
            print(f"[green]Committed changes:[/green] {committed_message}")
    if repo is not None and not changed_files:
        print("[yellow]No files changed; skipping commit[/yellow]")
