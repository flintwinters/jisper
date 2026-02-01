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
from rich.syntax import Syntax


# soft_wrap=True is VERY IMPORTANT for ergonomics.  DO NOT CHANGE
console = Console(soft_wrap=True, markup=False, highlight=False, no_color=False)
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

def as_non_empty_str(v) -> str | None:
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None

def dict_get(d: dict | None, key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)

def coerce_int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        return int(s) if s.isdigit() else None
    return None

def lower_keys(d: dict | None) -> dict:
    return dict(map(lambda kv: (str(kv[0]).lower(), kv[1]), (d or {}).items()))

def read_text_or_none(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None

def read_json5(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def env_non_empty(name: str) -> str | None:
    return as_non_empty_str(os.getenv(name or ""))

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
            print(f"\n[green]Committed changes:[/green] {committed_message}")
    if repo is not None and not changed_files:
        print("[yellow]No files changed; skipping commit[/yellow]")

    config_obj = load_prompt_file(config)
    in_usd_per_1m, out_usd_per_1m = get_model_prices_usd_per_1m(config_obj, model_code)
    print(format_token_cost_line(model_code, usage or {}, in_usd_per_1m, out_usd_per_1m))

def get_base_config_value(config: dict, key: str, default: str) -> str:
    return as_non_empty_str(dict_get(config, key)) or default


def get_model_code(config: dict) -> str:
    return get_base_config_value(config, "model", DEFAULT_MODEL)


def get_api_key_env_var_name(config: dict) -> str:
    return get_base_config_value(config, "api_key_env_var", DEFAULT_API_KEY_ENV_VAR)


def get_endpoint_url(config: dict) -> str:
    return get_base_config_value(config, "endpoint", DEFAULT_URL)


def get_api_key_from_env(env_var_name: str) -> str | None:
    return env_non_empty(env_var_name)

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

def styled_line_number(ln: int | None, *, width: int = 4, style: str | None = None) -> Text:
    """Create a fixed-width line-number Text with an optional style."""
    s = format_fixed_width_line_number(ln, width=width)
    t = Text(s)
    if style and ln is not None:
        t.stylize(style, 0, len(s))
    return t

def get_nested_str(d: dict, path: list[str]) -> str | None:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return as_non_empty_str(cur)

def load_prompt_file(path: Path) -> dict:
    return read_json5(path)

def read_file_text_or_none(path: Path) -> str | None:
    return read_text_or_none(path)


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



def extract_usage_from_api_response(api_json: dict, response_headers: dict) -> dict:
    usage = dict_get(api_json, "usage")
    prompt_tokens = coerce_int(dict_get(usage, "prompt_tokens"))
    completion_tokens = coerce_int(dict_get(usage, "completion_tokens"))
    total_tokens = coerce_int(dict_get(usage, "total_tokens"))

    header_map = lower_keys(response_headers)
    prompt_tokens = prompt_tokens or coerce_int(header_map.get("x-openai-prompt-tokens"))
    completion_tokens = completion_tokens or coerce_int(header_map.get("x-openai-completion-tokens"))
    total_tokens = total_tokens or coerce_int(header_map.get("x-openai-total-tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}


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
    pt = dict_get(usage, "prompt_tokens")
    ct = dict_get(usage, "completion_tokens")
    cost = estimate_cost_usd(pt, ct, in_usd_per_1m, out_usd_per_1m) or 0.0
    return f"[bright_black]${cost:.4f}[/bright_black]"


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
    console.print("[yellow](no diff; content is identical)[/yellow]")


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


def guess_syntax_lexer_name(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50]).lower()

    looks_like_diff = sample.startswith("diff ") or sample.startswith("---") or sample.startswith("+++") or "@@" in sample
    if looks_like_diff:
        return "diff"

    looks_like_json = sample.startswith("{") or sample.startswith("[")
    if looks_like_json:
        return "json"

    looks_like_python = "def " in sample or "import " in sample or "class " in sample
    if looks_like_python:
        return "python"

    return "text"


def syntax_text(body: str, *, lexer_name: str) -> Text:
    s = Syntax(
        body,
        lexer_name,
        theme="ansi_dark",
        line_numbers=False,
        word_wrap=False,
        code_width=0,
        indent_guides=False,
    )
    return s.highlight(body)


def format_combined_diff_lines(
    old_text: str,
    new_text: str,
    *,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
    lexer_name: str | None = None,
) -> list[tuple[str, Text]]:
    """Render unified-diff output into typed Rich Text lines with line numbers."""
    diff_lines = unified_diff_lines(old_text, new_text, context_lines=context_lines)

    out: list[tuple[str, Text]] = []
    old_ln = from_start
    new_ln = to_start

    old_lexer = lexer_name or guess_syntax_lexer_name(old_text)
    new_lexer = lexer_name or guess_syntax_lexer_name(new_text)

    def push(kind: str, line: Text):
        out.append((kind, line[:-1]))

    def numbered_line(ln: int | None, *, style: str | None, mid: str, body: str, lexer: str) -> Text:
        return styled_line_number(ln, style=style) + Text(mid) + syntax_text(body, lexer_name=lexer)

    def flush_pending_minus(pending: str | None) -> str | None:
        nonlocal old_ln
        if pending is None:
            return None
        push("delete", numbered_line(old_ln, style="bright_red on dark_red", mid=" - ", body=pending[1:], lexer=old_lexer))
        old_ln += 1
        return None

    pending_minus: str | None = None

    for line in diff_lines:
        if is_diff_meta_line(line) or not line:
            continue

        prefix = line[:1]
        body = line[1:]

        if prefix == " ":
            pending_minus = flush_pending_minus(pending_minus)
            push("context", numbered_line(new_ln, style=None, mid="   ", body=body, lexer=new_lexer))
            old_ln += 1
            new_ln += 1
            continue

        if prefix == "-":
            pending_minus = flush_pending_minus(pending_minus)
            pending_minus = line
            continue

        if prefix == "+":
            pending_minus = flush_pending_minus(pending_minus)
            push("insert", numbered_line(new_ln, style="bright_green on dark_green", mid=" + ", body=body, lexer=new_lexer))
            new_ln += 1
            continue

        pending_minus = flush_pending_minus(pending_minus)
        push("header", Text(line))

    flush_pending_minus(pending_minus)

    return out


def print_numbered_combined_diff(
    old_text: str,
    new_text: str,
    *,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
    title: str | None = None,
    lexer_name: str | None = None,
):
    """Print a numbered, colorized combined diff between two text blocks."""
    if title:
        console.print(f"\n{title}")

    lines = format_combined_diff_lines(
        old_text,
        new_text,
        from_start=from_start,
        to_start=to_start,
        context_lines=context_lines,
        lexer_name=lexer_name,
    )

    if not lines:
        print_no_diff_notice()
        return

    for _, t in lines:
        console.print(t)

def print_change_preview(filename: str, old_string: str, new_string: str, original: str):
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


def apply_one_replacement(original: str, old_string: str, new_string: str) -> tuple[str | None, str]:
    def replace_if(haystack: str, needle: str) -> str | None:
        return haystack.replace(needle, new_string) if needle and needle in haystack else None

    updated = replace_if(original, old_string)
    matched_old = old_string

    if updated is None:
        trimmed_old = old_string.strip()
        updated = replace_if(original, trimmed_old) if trimmed_old and trimmed_old != old_string else None
        matched_old = trimmed_old if updated is not None else matched_old

    if updated is None:
        stripped_original = original.strip()
        trimmed_old = old_string.strip()
        if stripped_original and trimmed_old and trimmed_old in stripped_original:
            leading = original[: len(original) - len(original.lstrip())]
            trailing = original[len(original.rstrip()):]
            replaced_core = stripped_original.replace(trimmed_old, new_string)
            updated = f"{leading}{replaced_core}{trailing}"
            matched_old = trimmed_old

    return (updated, matched_old)


def apply_replacements(replacements, base_dir: Path | None = None) -> list[Path]:
    """Apply {filename, old_string, new_string} edits to files on disk."""
    base_dir = base_dir or Path.cwd()

    def get_fields(i: int, r: dict) -> tuple[str | None, str | None, str | None] | None:
        filename = dict_get(r, "filename")
        old_string = dict_get(r, "old_string")
        new_string = dict_get(r, "new_string")
        if not filename:
            print(f"[red]Replacement #{i} missing filename; skipping[/red]")
            return None
        if old_string is None or new_string is None:
            print(f"[red]Replacement for {filename} missing old_string/new_string; skipping[/red]")
            return None
        return (filename, old_string, new_string)

    def preview_missing_old(filename: str, old_string: str, new_string: str):
        print(f"[yellow]old_string not found in {filename}; skipping[/yellow]")
        print_numbered_combined_diff(old_string, new_string, from_start=1, to_start=1, context_lines=2, title="Replacement text (preview)")

    def apply_one(i_r) -> Path | None:
        i, r = i_r
        fields = get_fields(i, r or {})
        if fields is None:
            return None
        filename, old_string, new_string = fields

        target_path = (base_dir / filename).resolve()
        original = read_text_or_none(target_path)
        if original is None:
            print(f"[red]Target file not found: {target_path}[/red]")
            return None

        updated, matched_old = apply_one_replacement(original, old_string, new_string)
        if updated is None:
            preview_missing_old(filename, old_string, new_string)
            return None

        if updated == original:
            print(f"[yellow]No changes applied to {filename} (replacement produced identical content)[/yellow]")
            return None

        print_change_preview(filename, matched_old, new_string, original)
        target_path.write_text(updated, encoding="utf-8")
        return target_path

    return list(filter(None, map(apply_one, enumerate(replacements or []))))


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
    """Undo the most recent commit by resetting HEAD and the working tree to its parent."""
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

    repo.git.reset("--hard", "HEAD~1")
    return 0


def redo_last_commit(base_dir: Path) -> int:
    """Redo an undone commit by restoring HEAD and the working tree to ORIG_HEAD when available."""
    repo = repo_from_dir(base_dir)
    if repo is None:
        print("Not a git repository")
        return 1

    if not repo.head.is_valid():
        print("No commits to redo")
        return 1

    if not (Path(repo.git_dir) / "ORIG_HEAD").exists():
        print("No ORIG_HEAD found to redo to")
        return 1

    orig_head = repo.commit("ORIG_HEAD")

    if orig_head.hexsha == repo.head.commit.hexsha:
        print("Already at the most recent commit")
        return 0

    repo.git.reset("--hard", orig_head.hexsha)
    return 0


if __name__ == "__main__":
    app()
