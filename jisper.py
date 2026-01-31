import requests
import os
import json5 as json
import difflib
from pathlib import Path
import git
import typer
from rich import print
from rich.console import Console
from rich.text import Text

console = Console(soft_wrap=False)
app = typer.Typer(add_completion=False)

DEFAULT_PROMPT_FILE = "prompt.json"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_FALLBACK_INPUT_USD_PER_1M = 5.0
DEFAULT_FALLBACK_OUTPUT_USD_PER_1M = 15.0

MODEL_PRICES_USD_PER_1M = {
    "gpt-5.2": (5.0, 15.0),
    "gpt-5-mini": (1.0, 3.0),
}

@app.callback(invoke_without_command=True)
def main(
    config: Path = typer.Option(
        DEFAULT_PROMPT_FILE,
        "-p",
        "--prompt",
        help="Path to prompt/config JSON5 file (default: prompt.json).",
        show_default=True,
    ),
    undo: bool = typer.Option(
        False,
        "-u",
        "--undo",
        help="Undo the last git commit (mixed reset to HEAD~1).",
        show_default=False,
    ),
    redo: bool = typer.Option(
        False,
        "--redo",
        help="Redo the last undo by moving HEAD to a more recent commit (fast-forward to ORIG_HEAD when available).",
        show_default=False,
    ),
) -> None:
    if redo:
        raise typer.Exit(code=redo_last_commit(Path.cwd()))

    if undo:
        raise typer.Exit(code=undo_last_commit(Path.cwd()))

    response, usage, model_code = run(config)
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

    config_obj = load_prompt_file(config)
    in_usd_per_1m, out_usd_per_1m = get_model_prices_usd_per_1m(config_obj, model_code)
    print(format_token_cost_line(model_code, usage or {}, in_usd_per_1m, out_usd_per_1m))

def get_non_empty_str(v) -> str | None:
    """Return a stripped string when the input is a non-empty string, otherwise None."""
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def get_base_config_value(config: dict, key: str, default: str) -> str:
    """Read a string config value with fallback to a provided default."""
    return get_non_empty_str(config.get(key)) or default


def get_model_code(config: dict) -> str:
    """Return the configured model code or the default model."""
    return get_base_config_value(config, "model", DEFAULT_MODEL)


def get_api_key_env_var_name(config: dict) -> str:
    """Return the environment variable name used to read the API key."""
    return get_base_config_value(config, "api_key_env_var", DEFAULT_API_KEY_ENV_VAR)


def get_endpoint_url(config: dict) -> str:
    """Return the API endpoint URL from config with a default fallback."""
    return get_base_config_value(config, "endpoint", DEFAULT_URL)


def get_api_key_from_env(env_var_name: str) -> str | None:
    """Read and normalize an API key from an environment variable."""
    return get_non_empty_str(os.getenv(env_var_name or ""))

def is_file_header_line(line: str) -> bool:
    """Return True for unified-diff file header lines (---/+++)."""
    return line.startswith("---") or line.startswith("+++")


def is_hunk_header_line(line: str) -> bool:
    """Return True for unified-diff hunk header lines (@@ ... @@)."""
    return line.startswith("@@")


def is_diff_meta_line(line: str) -> bool:
    """Return True for unified-diff metadata lines that are not content."""
    return is_file_header_line(line) or is_hunk_header_line(line)

def format_fixed_width_line_number(ln: int | None, *, width: int = 4) -> str:
    """Format an optional line number into a fixed-width field for diff output."""
    if ln is None:
        return " " * width
    return f"{ln:>{width}}"

def text_with_prefix(prefix: str, body: str) -> Text:
    """Build a Text line from a prefix and body without using Rich markup."""
    t = Text(prefix)
    t.append(body)
    return t

def styled_line_number(ln: int | None, *, width: int = 4, style: str | None = None) -> Text:
    """Create a fixed-width line-number Text with an optional style."""
    s = format_fixed_width_line_number(ln, width=width)
    t = Text(s)
    if style and ln is not None:
        t.stylize(style, 0, len(s))
    return t

def get_nested_str(d: dict, path: list[str]) -> str | None:
    """Safely read and normalize a nested non-empty string value from a dict path."""
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if isinstance(cur, str) and cur.strip():
        return cur.strip()
    return None

def load_prompt_file(path: Path) -> dict:
    """Load the prompt/config JSON5 file."""
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            print("[bold red]< json syntax error >[/bold red]")
            exit(1)

def read_file_text_or_none(path: Path) -> str | None:
    """Return a file's UTF-8 text content if it exists, otherwise None."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_and_concatenate_files(file_list):
    """Read a list of files and concatenate them into a single labeled source block."""
    def one(filename: str) -> str | None:
        p = Path(filename)
        txt = read_file_text_or_none(p)
        if txt is None:
            print(f"[red]Missing input file: {filename}[/red]")
            return None
        return f"--- FILENAME: {p.name} ---\n{txt}"

    return "\n\n".join(filter(None, map(one, file_list or [])))

def build_payload(prompt_config: dict, source_text: str):
    """Build the chat-completions request payload from config and concatenated source text."""
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

def get_int_from_any(v) -> int | None:
    """Coerce common JSON-like values into an int when safely possible."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
    return None


def safe_get(d: dict, key: str):
    """Get a key from a dict, returning None for non-dict inputs."""
    if not isinstance(d, dict):
        return None
    return d.get(key)


def lower_keyed_dict(d: dict | None) -> dict:
    """Return a copy of a mapping with string-lowered keys for case-insensitive lookup."""
    return dict(map(lambda kv: (str(kv[0]).lower(), kv[1]), (d or {}).items()))


def extract_usage_from_api_response(api_json: dict, response_headers: dict) -> dict:
    """Extract token usage counts from the API JSON and/or response headers."""
    usage = safe_get(api_json, "usage")
    prompt_tokens = get_int_from_any(safe_get(usage, "prompt_tokens"))
    completion_tokens = get_int_from_any(safe_get(usage, "completion_tokens"))
    total_tokens = get_int_from_any(safe_get(usage, "total_tokens"))

    header_map = lower_keyed_dict(response_headers)
    prompt_tokens = prompt_tokens or get_int_from_any(header_map.get("x-openai-prompt-tokens"))
    completion_tokens = completion_tokens or get_int_from_any(header_map.get("x-openai-completion-tokens"))
    total_tokens = total_tokens or get_int_from_any(header_map.get("x-openai-total-tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def get_model_prices_usd_per_1m(config: dict, model_code: str) -> tuple[float, float]:
    """Resolve input/output token prices for a model from config, known defaults, or fallbacks."""
    conf_prices = config.get("model_prices_usd_per_1m")
    if isinstance(conf_prices, dict):
        v = conf_prices.get(model_code)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            in_p = v[0]
            out_p = v[1]
            if isinstance(in_p, (int, float)) and isinstance(out_p, (int, float)):
                return (float(in_p), float(out_p))

    if model_code in MODEL_PRICES_USD_PER_1M:
        return MODEL_PRICES_USD_PER_1M[model_code]

    fallback_in = config.get("fallback_input_usd_per_1m")
    fallback_out = config.get("fallback_output_usd_per_1m")
    in_p = float(fallback_in) if isinstance(fallback_in, (int, float)) else DEFAULT_FALLBACK_INPUT_USD_PER_1M
    out_p = float(fallback_out) if isinstance(fallback_out, (int, float)) else DEFAULT_FALLBACK_OUTPUT_USD_PER_1M
    return (in_p, out_p)


def estimate_cost_usd(prompt_tokens: int | None, completion_tokens: int | None, in_usd_per_1m: float, out_usd_per_1m: float) -> float | None:
    """Estimate request cost in USD from token counts and per-1M pricing."""
    if prompt_tokens is None and completion_tokens is None:
        return None
    pt = float(prompt_tokens or 0)
    ct = float(completion_tokens or 0)
    return (pt * in_usd_per_1m + ct * out_usd_per_1m) / 1_000_000.0


def format_token_cost_line(model_code: str, usage: dict, in_usd_per_1m: float, out_usd_per_1m: float) -> str:
    """Format a single dim cost line for display from usage and pricing."""
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    cost = estimate_cost_usd(pt, ct, in_usd_per_1m, out_usd_per_1m)

    cost_s = f"${cost:.4f}"
    return f"[bright_black]{cost_s}[/bright_black]"


def run(config_path: Path) -> tuple[dict, dict, str]:
    """Call the model API using the config file and return parsed output, usage, and model code."""
    config = load_prompt_file(config_path)

    endpoint_url = get_endpoint_url(config)
    api_key_env_var = get_api_key_env_var_name(config)
    api_key = get_api_key_from_env(api_key_env_var)

    concatenated_text = read_and_concatenate_files(config["input_files"])

    headers = {
        "Authorization": f"Bearer {api_key or ''}",
        "Content-Type": "application/json",
    }

    payload = build_payload(config, concatenated_text)

    with console.status("Waiting for model...", spinner="dots"):
        response = requests.post(endpoint_url, headers=headers, json=payload)

    api_json = response.json()
    model_code = get_model_code(config)
    usage = extract_usage_from_api_response(api_json, dict(response.headers))
    model_out = json.loads(api_json["choices"][0]["message"]["content"])
    return (model_out, usage, model_code)


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


def print_no_diff_notice():
    """Print a standardized message indicating there is no diff to display."""
    console.print("[yellow](no diff; contents are identical)[/yellow]")


def unified_diff_lines(
    old_text: str,
    new_text: str,
    *,
    context_lines: int = 3,
) -> list[str]:
    """Compute unified-diff lines between two strings with configurable context."""
    old_lines = old_text.splitlines(keepends=False)
    new_lines = new_text.splitlines(keepends=False)
    return list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=context_lines))


def format_combined_diff_lines(
    old_text: str,
    new_text: str,
    *,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
) -> list[tuple[str, Text]]:
    """Render unified-diff output into typed Rich Text lines with line numbers."""
    diff_lines = unified_diff_lines(old_text, new_text, context_lines=context_lines)

    out: list[tuple[str, Text]] = []
    old_ln = from_start
    new_ln = to_start

    def push(kind: str, line: Text):
        out.append((kind, line))

    def numbered_line(ln: int | None, *, style: str | None, mid: str, body: str) -> Text:
        ln_t = styled_line_number(ln, style=style)
        return ln_t + Text(mid + body)

    pending_minus: str | None = None

    for line in diff_lines:
        if is_diff_meta_line(line) or not line:
            continue

        prefix = line[:1]
        body = line[1:]

        if prefix == " ":
            if pending_minus is not None:
                push("delete", numbered_line(old_ln, style="bright_red", mid=" - ", body=pending_minus[1:]))
                old_ln += 1
                pending_minus = None
            push("context", numbered_line(new_ln, style=None, mid="   ", body=body))
            old_ln += 1
            new_ln += 1
            continue

        if prefix == "-":
            if pending_minus is not None:
                push("delete", numbered_line(old_ln, style="bright_red", mid=" - ", body=pending_minus[1:]))
                old_ln += 1
            pending_minus = line
            continue

        if prefix == "+":
            if pending_minus is not None:
                push("delete", numbered_line(old_ln, style="bright_red", mid=" - ", body=pending_minus[1:]))
                pending_minus = None
                old_ln += 1
            push("insert", numbered_line(new_ln, style="bright_green", mid=" + ", body=body))
            new_ln += 1
            continue

        if pending_minus is not None:
            push("delete", numbered_line(old_ln, style="bright_red", mid=" - ", body=pending_minus[1:]))
            old_ln += 1
            pending_minus = None
        push("header", Text(line))

    if pending_minus is not None:
        push("delete", numbered_line(old_ln, style="bright_red", mid=" - ", body=pending_minus[1:]))

    return out


def print_numbered_combined_diff(
    old_text: str,
    new_text: str,
    *,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
    title: str | None = None,
):
    """Print a numbered, colorized combined diff between two text blocks."""
    if title:
        console.print(f"\n[bold]{title}[/bold]")

    lines = format_combined_diff_lines(
        old_text,
        new_text,
        from_start=from_start,
        to_start=to_start,
        context_lines=context_lines,
    )

    if not lines:
        print_no_diff_notice()
        return

    base_bg = "#2b2b2b"
    add_bg = "#0f3d0f"
    del_bg = "#4a1414"
    rep_bg = "#3a2f0a"

    def bg_for_kind(kind: str) -> str:
        if kind == "insert":
            return add_bg
        if kind == "delete":
            return del_bg
        if kind == "replace":
            return rep_bg
        return base_bg

    def apply_bg(t: Text, bg: str):
        if t.plain:
            t.stylize(f"on {bg}", 0, len(t.plain))

    def terminal_width() -> int:
        return max(0, int(getattr(console, "width", 0) or 0))

    def pad_to_terminal_width(t: Text) -> Text:
        w = terminal_width()
        if w <= 0:
            return t
        pad = max(0, w - len(t.plain))
        if pad <= 0:
            return t
        return t + Text(" " * pad)

    def render(kind_and_text: tuple[str, Text]) -> Text:
        kind, t = kind_and_text
        out = pad_to_terminal_width(t)
        apply_bg(out, bg_for_kind(kind))
        return out

    for t in map(render, lines):
        console.print(t)

def print_change_preview(filename: str, old_string: str, new_string: str, original: str, updated: str):
    """Print a focused diff preview of a pending replacement within a file."""
    line_info = compute_replacement_line_numbers(original, old_string, new_string)
    old_start = line_info[0] if line_info else 1
    new_start = line_info[1] if line_info else 1

    print_numbered_combined_diff(
        old_string,
        new_string,
        from_start=old_start,
        to_start=new_start,
        context_lines=2,
        title=filename,
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
            print_numbered_combined_diff(
                old_string,
                new_string,
                from_start=1,
                to_start=1,
                context_lines=2,
                title="Replacement text (preview)",
            )
            continue

        if updated == original:
            print(f"[yellow]No changes applied to {filename} (replacement produced identical content)[/yellow]")
            continue

        # Show very clear diff previews before writing.
        print_change_preview(filename, matched_old, new_string, original, updated)

        target_path.write_text(updated, encoding="utf-8")
        changed.append(target_path)

    return changed


def print_model_change_notes(model_output: dict):
    """Print the model response and edit explanation (if present)."""
    if not isinstance(model_output, dict):
        return

    edits = model_output.get("edit") or {}
    print(edits.get("explanation"))


def extract_commit_message(model_output: dict) -> str | None:
    """Extract a commit message from model output using common schema locations."""
    if not isinstance(model_output, dict):
        return None

    return get_nested_str(model_output, ["commit_message"]) or get_nested_str(model_output, ["edit", "commit_message"])


def repo_from_dir(base_dir: Path) -> git.Repo | None:
    """Find and open the nearest git repository at or above a directory."""
    if (base_dir / ".git").exists():
        return git.Repo(base_dir)

    current = base_dir
    while current != current.parent:
        if (current / ".git").exists():
            return git.Repo(current)
        current = current.parent

    return None


def stage_and_commit(repo: git.Repo, changed_files: list[Path], message: str) -> str | None:
    """Stage a set of changed files and create a commit with the given message."""
    repo_root = Path(repo.working_tree_dir or ".").resolve()
    relpaths = list(map(lambda p: str(p.resolve().relative_to(repo_root)), changed_files))
    relpaths = list(filter(lambda s: bool(s and s.strip()), relpaths))

    if not relpaths:
        return None

    repo.index.add(relpaths)
    repo.index.commit(message)
    return message


def undo_last_commit(base_dir: Path) -> int:
    """Undo the most recent commit by resetting HEAD to its parent (mixed)."""
    repo = repo_from_dir(base_dir)
    if repo is None:
        print("Not a git repository")
        return 1

    if not repo.head.is_valid():
        print("No commits to undo")
        return 1

    if not repo.head.commit.parents:
        print("No parent commit to reset to")
        return 1

    repo.git.reset("--mixed", "HEAD~1")
    return 0


def redo_last_commit(base_dir: Path) -> int:
    """Redo an undone commit by fast-forwarding HEAD to ORIG_HEAD when available."""
    repo = repo_from_dir(base_dir)
    if repo is None:
        print("Not a git repository")
        return 1

    if not repo.head.is_valid():
        print("No commits to redo")
        return 1

    orig_head = None
    try:
        orig_head = repo.commit("ORIG_HEAD")
    except Exception:
        orig_head = None

    if orig_head is None:
        print("No ORIG_HEAD found to redo to")
        return 1

    if orig_head.hexsha == repo.head.commit.hexsha:
        print("Already at the most recent commit")
        return 0

    repo.git.merge("--ff-only", orig_head.hexsha)
    return 0


if __name__ == "__main__":
    app()
