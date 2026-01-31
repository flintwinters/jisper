import requests
import os
import json5 as json
import difflib
from pathlib import Path
import re
import git
import typer
from rich import print
from rich.console import Console
from rich.text import Text
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn


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

def is_file_header_line(line: str) -> bool:
    return line.startswith("---") or line.startswith("+++")

def is_hunk_header_line(line: str) -> bool:
    return line.startswith("@@")

def format_fixed_width_line_number(ln: int | None, *, width: int = 4) -> str:
    if ln is None:
        return " " * width
    return f"{ln:>{width}}"

def get_nested_str(d: dict, path: list[str]) -> str | None:
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

def get_int_from_any(v) -> int | None:
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
    if not isinstance(d, dict):
        return None
    return d.get(key)


def extract_usage_from_api_response(api_json: dict, response_headers: dict) -> dict:
    usage = safe_get(api_json, "usage")
    prompt_tokens = get_int_from_any(safe_get(usage, "prompt_tokens"))
    completion_tokens = get_int_from_any(safe_get(usage, "completion_tokens"))
    total_tokens = get_int_from_any(safe_get(usage, "total_tokens"))

    header_map = dict(map(lambda kv: (str(kv[0]).lower(), kv[1]), (response_headers or {}).items()))
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
    if prompt_tokens is None and completion_tokens is None:
        return None
    pt = float(prompt_tokens or 0)
    ct = float(completion_tokens or 0)
    return (pt * in_usd_per_1m + ct * out_usd_per_1m) / 1_000_000.0


def format_token_cost_line(model_code: str, usage: dict, in_usd_per_1m: float, out_usd_per_1m: float) -> str:
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    cost = estimate_cost_usd(pt, ct, in_usd_per_1m, out_usd_per_1m)

    def fmt_int(v) -> str:
        return str(v) if isinstance(v, int) else "?"

    cost_s = f"${cost:.6f}" if isinstance(cost, float) else "$?"
    return f"[gray]~{cost_s}[/gray]"


def run(config_path: Path) -> tuple[dict, dict, str]:
    config = load_prompt_file(config_path)

    endpoint_url = get_endpoint_url(config)
    api_key_env_var = get_api_key_env_var_name(config)
    api_key = get_api_key_from_env(api_key_env_var)

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

    api_json = response.json()
    model_code = get_model_code(config)
    usage = extract_usage_from_api_response(api_json, dict(response.headers))
    model_out = json.loads(api_json["choices"][0]["message"]["content"])
    return (model_out, usage, model_code)

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
    """Rich Text with inline diff highlighting using background colors.

    Uses a token-based matcher and merges nearby change hunks to avoid overly
    fragmented segments.
    """
    t = Text()
    a_tokens = tokenize_for_intraline_diff(old)
    b_tokens = tokenize_for_intraline_diff(new)

    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    opcodes = merge_change_opcodes(sm.get_opcodes(), a_tokens, b_tokens)

    del_bg = "#4a1414"
    add_bg = "#0f3d0f"

    def append_bg(s: str, bg: str):
        if not s:
            return
        t.append(s, style=f"on {bg}")

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            t.append("".join(a_tokens[i1:i2]))
            continue

        if tag == "delete":
            append_bg("".join(a_tokens[i1:i2]), del_bg)
            continue

        if tag == "insert":
            append_bg("".join(b_tokens[j1:j2]), add_bg)
            continue

        if tag == "replace":
            append_bg("".join(a_tokens[i1:i2]), del_bg)
            append_bg("".join(b_tokens[j1:j2]), add_bg)
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
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
) -> list[str]:
    old_lines = old_text.splitlines(keepends=False)
    new_lines = new_text.splitlines(keepends=False)

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfiledate=f"line {from_start}",
            tofiledate=f"line {to_start}",
            lineterm="",
            n=context_lines,
        )
    )

    new_ln = to_start

    def fmt(ln: int | None) -> str:
        return format_fixed_width_line_number(ln)

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
    for line in diff_lines:
        if is_file_header_line(line) or is_hunk_header_line(line):
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
        from_start=from_start,
        to_start=to_start,
        context_lines=context_lines,
    )

    if not lines:
        console.print("[yellow](no diff; contents are identical)[/yellow]")
        return

    console.print(Syntax("\n".join(lines), "diff", theme="ansi_dark", line_numbers=False, word_wrap=False))


def format_numbered_combined_diff(
    old_text: str,
    new_text: str,
    *,
    from_start: int = 1,
    to_start: int = 1,
    context_lines: int = 3,
) -> list[tuple[str, Text]]:
    """Return a single combined diff with line numbers.

    The first tuple element is a kind: context|delete|insert|replace|header.
    The second element is a Rich Text line that includes the line-number column and prefix.
    """
    diff_lines = unified_diff_lines(
        old_text,
        new_text,
        context_lines=context_lines,
    )

    out: list[tuple[str, Text]] = []
    old_ln = from_start
    new_ln = to_start

    def fmt_ln(ln: int | None) -> str:
        return format_fixed_width_line_number(ln)

    def push(kind: str, line: Text):
        out.append((kind, line))

    def push_header(line: str):
        push("header", Text(line))

    def push_context(text: str):
        nonlocal old_ln, new_ln
        push("context", Text(f"{fmt_ln(new_ln)}  {text}"))
        old_ln += 1
        new_ln += 1

    def push_delete(text: str):
        nonlocal old_ln
        push("delete", Text(f"{fmt_ln(old_ln)}- {text}"))
        old_ln += 1

    def push_insert(text: str):
        nonlocal new_ln
        push("insert", Text(f"{fmt_ln(new_ln)}+ {text}"))
        new_ln += 1

    def push_replace(old_line: str, new_line: str):
        nonlocal old_ln, new_ln
        merged_prefix = Text(f"{fmt_ln(new_ln)}  ~ ", style="bold")
        push("replace", merged_prefix + rich_inline_diff(old_line, new_line))
        old_ln += 1
        new_ln += 1

    is_hunk_header = is_hunk_header_line
    is_file_header = is_file_header_line

    pending_minus: str | None = None

    for line in diff_lines:
        if is_file_header(line) or is_hunk_header(line):
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
        from_start=from_start,
        to_start=to_start,
        context_lines=context_lines,
    )

    if not lines:
        console.print("[yellow](no diff; contents are identical)[/yellow]")
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

    def pad_text_to_console_width(t: Text) -> Text:
        width = console.size.width
        if width <= 0:
            return t
        pad = max(0, width - len(t.plain))
        if pad <= 0:
            return t
        out = t.copy()
        out.append(" " * pad)
        return out

    def line_kind_prefix_len(kind: str) -> int:
        return 10

    def highlight_diff_prefix_len(text_line: str) -> int:
        if not text_line:
            return line_kind_prefix_len("context")
        if len(text_line) < 11:
            return min(len(text_line), line_kind_prefix_len("context"))
        if text_line[8:11] in {"  -", "  +", "  ~"}:
            return 11
        return line_kind_prefix_len("context")

    def highlight_one_line(line: str) -> Text:
        src = line[:-1] if line.endswith("\n") else line
        s = Syntax(src, "diff", theme="ansi_dark", background_color=base_bg, line_numbers=False, word_wrap=False)
        segs = list(s.highlight(src))
        out = Text.assemble(*segs)
        if out.plain.endswith("\n"):
            out = out[:-1]
        return out

    def apply_bg_to_range(t: Text, bg: str, start: int, end: int):
        if start >= end:
            return
        t.stylize(f"on {bg}", start, end)

    def apply_kind_bg(t: Text, kind: str):
        bg = bg_for_kind(kind)
        apply_bg_to_range(t, bg, 0, len(t.plain))

    def apply_inline_replace_bg(t: Text, kind: str):
        if kind != "replace":
            return
        bg = bg_for_kind(kind)
        end = min(highlight_diff_prefix_len(t.plain), len(t.plain))
        apply_bg_to_range(t, bg, 0, end)

    def merge_replace_line(base: Text, replace_line: Text) -> Text:
        prefix_len = min(highlight_diff_prefix_len(base.plain), len(base.plain))
        out = Text(base.plain[:prefix_len])
        out.stylize_ranges(base._spans)
        out.append(replace_line)
        return out

    padded = list(map(lambda kl: (kl[0], pad_text_to_console_width(kl[1])), lines))

    def render_line(kl: tuple[str, Text]) -> Text:
        kind, t = kl
        plain = t.plain
        base = highlight_one_line(plain)
        apply_kind_bg(base, kind)
        if kind != "replace":
            return base

        prefix_len = min(highlight_diff_prefix_len(plain), len(plain))
        base_prefix = base[:prefix_len]
        inline_src = plain[prefix_len:]

        start = inline_src.find("~ ")
        if start < 0:
            merged = base_prefix + highlight_one_line(inline_src)
            apply_inline_replace_bg(merged, kind)
            return merged

        inline_before = inline_src[: start + 2]
        inline_body = inline_src[start + 2 :]

        old_body, new_body = (inline_body.split(" => ", 1) + [""])[:2]
        if not new_body:
            merged = base_prefix + highlight_one_line(inline_src)
            apply_inline_replace_bg(merged, kind)
            return merged

        before_text = highlight_one_line(inline_before)
        diff_text = rich_inline_diff(old_body, new_body)
        merged = base_prefix + before_text + diff_text
        apply_inline_replace_bg(merged, kind)
        return merged

    rendered = list(map(render_line, padded))

    for t in rendered:
        console.print(t)

def unified_diff_lines(
    old_text: str,
    new_text: str,
    *,
    context_lines: int = 3,
) -> list[str]:
    old_lines = old_text.splitlines(keepends=False)
    new_lines = new_text.splitlines(keepends=False)
    return list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm="",
            n=context_lines,
        )
    )


def print_intraline_diff(
    old_text: str,
    new_text: str,
    *,
    context_lines: int = 3,
    title: str | None = None,
):
    """Backward-compatible wrapper: print the combined diff view."""
    print_numbered_combined_diff(
        old_text,
        new_text,
        from_start=1,
        to_start=1,
        context_lines=context_lines,
        title=title,
    )


def print_change_preview(filename: str, old_string: str, new_string: str, original: str, updated: str):
    line_info = compute_replacement_line_numbers(original, old_string, new_string)
    old_start = line_info[0] if line_info else 1
    new_start = line_info[1] if line_info else 1

    print_numbered_combined_diff(
        old_string,
        new_string,
        from_start=old_start,
        to_start=new_start,
        context_lines=2,
        title=f"{filename}",
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
    print(edits.get("explanation"))


def extract_commit_message(model_output: dict) -> str | None:
    if not isinstance(model_output, dict):
        return None

    return get_nested_str(model_output, ["commit_message"]) or get_nested_str(model_output, ["edit", "commit_message"])


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


def redo_last_commit(base_dir: Path) -> int:
    repo = repo_from_dir(base_dir)
    if repo is None:
        print("Not a git repository")
        return 1

    if not repo.head.is_valid():
        print("No commits to redo")
        return 1

    if not repo.head.commit.parents:
        print("No parent commit to reset to")
        return 1

    repo.git.reset("--mixed", "HEAD~1")
    return 0


def undo_last_commit(base_dir: Path) -> int:
    return redo_last_commit(base_dir)


@app.command(add_help_option=False)
def main(
    config: Path = typer.Option(
        DEFAULT_PROMPT_FILE,
        "--config",
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
        help="Redo/undo the last commit (alias of --undo).",
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


if __name__ == "__main__":
    app()
