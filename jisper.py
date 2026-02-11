"""
[FILE SUMMARY]
context:
  INTENT:
    purpose: >
      Implements the Jisper CLI: load a prompt config, gather source material and optional
      [FILE SUMMARY]-derived context, call a chat completions endpoint, preview and apply exact
      string replacements, and optionally commit results to git.

  STRUCTURAL:
    responsibility: >
      Orchestrates end-to-end edit runs (config -> payload -> HTTP -> parse -> preview/apply -> git),
      and provides utilities for deterministic substring replacements with diff previews and token cost
      reporting.
    boundaries:
      owns:
        - Typer CLI entrypoint and command flow (run/undo/redo)
        - Prompt config loading (YAML/JSON5) and routine task resolution
        - Source file inclusion resolution and concatenation
        - [FILE SUMMARY] block extraction and context field selection for prompt construction
        - OpenAI-compatible chat completions request payload construction with JSON schema response_format
        - API usage extraction and token cost estimation/formatting
        - Replacement application semantics (substring match + whitespace-tolerant fallbacks; optional file creation when old_string is blank)
        - Diff preview rendering with stable line-number presentation
        - Git repo discovery, staging, committing, and undo/redo operations
      does_not_own:
        - Authoritative project documentation (owned by README.md)
        - Model-side instruction quality and correctness of replacements (owned by prompt/system_prompt)
        - External endpoint behavior, authentication correctness, or network reliability
    entrypoints:
      - app
      - main
      - run
      - apply_replacements
      - undo_last_commit
      - redo_last_commit
    key_functions:
      - name: load_prompt_file
        signature: load_prompt_file(path: Path) -> dict
        purpose: >
          Loads the user prompt/config file in YAML or JSON5 form and returns a dict configuration.
      - name: resolve_included_files
        signature: resolve_included_files(config: dict) -> dict
        purpose: >
          Computes file inclusion lists and the final de-duplicated ordered source file set.
      - name: build_file_summaries_section
        signature: build_file_summaries_section(files: list[str], *, intent_only: bool) -> str
        purpose: >
          Extracts and emits compact YAML summaries derived from [FILE SUMMARY] blocks to reduce context.
      - name: build_openai_payload
        signature: build_openai_payload(prompt_config: dict, source_text: str, routine_name: str | None = None)
        purpose: >
          Constructs the chat completions payload including system/user messages and optional JSON schema for OpenAI.
      - name: run
        signature: run(config_path: Path, routine_name: str | None = None) -> tuple[dict, dict, str]
        purpose: >
          Executes one end-to-end model call and returns parsed model output, usage, and model code.
      - name: apply_one_replacement
        signature: apply_one_replacement(original: str, old_string: str, new_string: str) -> tuple[str | None, str]
        purpose: >
          Applies deterministic substring replacement with fallbacks and returns updated text plus the matched old string.
      - name: apply_replacements
        signature: apply_replacements(replacements, base_dir: Path | None = None) -> list[Path]
        purpose: >
          Applies a list of (filename, old_string, new_string) operations to disk with diff previews.
      - name: repo_from_dir
        signature: repo_from_dir(base_dir: Path) -> git.Repo | None
        purpose: >
          Locates the nearest git repository root by walking upward.
      - name: stage_and_commit
        signature: stage_and_commit(repo: git.Repo, changed_files: list[Path], message: str) -> str | None
        purpose: >
          Stages changed files and creates a git commit with the provided message.
      - name: undo_last_commit
        signature: undo_last_commit(base_dir: Path) -> int
        purpose: >
          Resets the repository to the parent of HEAD (hard reset).
      - name: redo_last_commit
        signature: redo_last_commit(base_dir: Path) -> int
        purpose: >
          Moves HEAD back to ORIG_HEAD when available (hard reset).
    dependencies:
      - README.md
[/FILE SUMMARY]
"""
import requests
import os
import json5 as json
import yaml
import difflib
from pathlib import Path
import git
import typer
import subprocess
import sys

from rich import print
from rich.console import Console
from rich.text import Text
from rich.syntax import Syntax


# soft_wrap=True is VERY IMPORTANT for ergonomics.  DO NOT CHANGE
console = Console(soft_wrap=True, markup=False, highlight=False, no_color=False)
app = typer.Typer(add_completion=False)

DEFAULT_PROMPT_FILE = "prompt.yaml"
DEFAULT_TEMPLATE_PROMPT_FILE = "default_prompt.yaml"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_FALLBACK_INPUT_USD_PER_1M = 5.0
DEFAULT_FALLBACK_OUTPUT_USD_PER_1M = 15.0
DEFAULT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "edit": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "An short 1-2 sentence explanation of the changes you are making",
                },
                "commit_message": {
                    "type": "string",
                    "description": "The commit message to use for the changes you are making",
                },
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "The file in which to apply the edit",
                            },
                            "old_string": {
                                "type": "string",
                                "description": "The old string to replace in the file",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "The new string to add",
                            },
                        },
                        "required": ["filename", "old_string", "new_string"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["explanation", "commit_message", "replacements"],
            "additionalProperties": False,
        }
    },
    "required": ["edit"],
}

# (input, output)
MODEL_PRICES_USD_PER_1M = {
    "gpt-5.2": (5.0, 15.0),
    "gpt-5-mini": (1.0, 3.0),
    "qwen/qwen3-coder:exacto": (0.22, 1.8),
    "moonshotai/kimi-k2.5": (.25, 2.25)
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



def read_text_or_none(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None





def resolve_routine_task(config: dict, routine_name: str | None) -> str | None:
    name = as_non_empty_str(routine_name)
    if not name:
        return None

    routines = dict_get(config, "routines")
    routines = routines if isinstance(routines, dict) else {}
    task = routines.get(name)
    task = as_non_empty_str(task)
    return task


def as_list_of_non_empty_str(v) -> list[str]:
    if not isinstance(v, list):
        return []
    out = list(filter(None, map(as_non_empty_str, v)))
    return list(map(str, out))


def dedupe_keep_order(xs: list[str]) -> list[str]:
    seen = set()

    def keep(x: str) -> bool:
        if x in seen:
            return False
        seen.add(x)
        return True

    return list(filter(keep, xs or []))


def resolve_paths_and_globs(values: list[str], *, base_dir: Path) -> list[str]:
    def to_rel(p: Path) -> str:
        try_rel = p.resolve()
        base = base_dir.resolve()
        return str(try_rel.relative_to(base)) if str(try_rel).startswith(str(base)) else str(p)

    def one(v: str) -> list[str]:
        s = as_non_empty_str(v)
        if not s:
            return []

        p = base_dir / s
        if p.exists() and p.is_dir():
            return list(
                map(
                    to_rel,
                    filter(
                        lambda c: c.is_file(),
                        sorted(p.iterdir(), key=lambda x: x.name),
                    ),
                )
            )

        if p.exists() and p.is_file():
            return [to_rel(p)]

        matches = sorted(map(to_rel, filter(lambda m: m.is_file(), base_dir.glob(s))))
        return matches

    return dedupe_keep_order(sum(map(one, values or []), []))



def resolve_included_files(config: dict) -> dict:
    base_dir = Path.cwd()
    full_raw = as_list_of_non_empty_str(dict_get(config, "full_files"))
    structural_raw = as_list_of_non_empty_str(dict_get(config, "structural_level_files"))
    input_raw = as_list_of_non_empty_str(dict_get(config, "input_level_files"))

    input_level_files = resolve_paths_and_globs(input_raw, base_dir=base_dir)
    structural_level_files = resolve_paths_and_globs(structural_raw, base_dir=base_dir)
    lower_level_set = set(input_level_files + structural_level_files)

    full_files = resolve_paths_and_globs(full_raw, base_dir=base_dir)
    full_files = list(filter(lambda s: s not in lower_level_set, full_files or []))

    source_files = dedupe_keep_order(full_files + structural_level_files + input_level_files)
    return {
        "full_files": full_files,
        "structural_level_files": structural_level_files,
        "input_level_files": input_level_files,
        "source_files": source_files,
    }


def extract_file_summary_yaml(text: str) -> dict | None:
    s = as_non_empty_str(text)
    if not s:
        return None

    start_tag = "[FILE SUMMARY]"
    end_tag = "[/FILE SUMMARY]"
    start = s.find(start_tag)
    if start < 0:
        return None
    end = s.find(end_tag, start + len(start_tag))
    if end < 0:
        return None

    inner = s[start + len(start_tag) : end]
    inner = inner.strip("\n").strip()
    if not inner:
        return None

    loaded = yaml.safe_load(inner)
    return loaded if isinstance(loaded, dict) else None



def select_context_fields(summary: dict, *, intent_only: bool) -> dict | None:
    ctx = dict_get(summary, "context")
    if not isinstance(ctx, dict):
        return None

    intent = dict_get(ctx, "INTENT")
    structural = dict_get(ctx, "STRUCTURAL")

    if intent_only:
        return {"context": {"INTENT": intent}} if isinstance(intent, dict) else None

    out_ctx = {}
    if isinstance(intent, dict):
        out_ctx["INTENT"] = intent
    if isinstance(structural, dict):
        out_ctx["STRUCTURAL"] = structural
    return {"context": out_ctx} if out_ctx else None


def build_file_summaries_section(files: list[str], *, intent_only: bool) -> str:
    def one(filename: str) -> str | None:
        p = Path(filename)
        txt = read_text_or_none(p)
        if txt is None:
            print(f"[red]Missing input file: {filename}[/red]")
            return None

        summary = extract_file_summary_yaml(txt)
        if summary is None:
            return None

        selected = select_context_fields(summary, intent_only=intent_only)
        if selected is None:
            return None

        dumped = yaml.safe_dump(selected, sort_keys=False).rstrip() if isinstance(selected, dict) else ""
        return f"--- FILENAME: {filename} ---\n{dumped}" if dumped else None

    joined = "\n\n".join(filter(None, map(one, files or [])))
    return joined.strip() if joined else ""


def write_default_prompt_to_cwd() -> int:
    src = Path(__file__).resolve().parent / DEFAULT_TEMPLATE_PROMPT_FILE
    if not src.exists() or not src.is_file():
        print(f"[red]Missing template prompt file:[/red] {src}")
        return 1

    dst = (Path.cwd() / DEFAULT_PROMPT_FILE).resolve()
    if dst.exists():
        print(f"[yellow]{DEFAULT_PROMPT_FILE} already exists; refusing to overwrite[/yellow]")
        return 1

    content = src.read_text(encoding="utf-8")
    dst.write_text(content, encoding="utf-8")
    print(f"[green]Wrote {DEFAULT_PROMPT_FILE}[/green]")
    return 0


@app.callback(invoke_without_command=True)
def main(
    routine: str | None = typer.Argument(
        None,
        help="Optional routine name to use from prompt config's `routines` mapping (overrides default task).",
        show_default=False,
    ),
    config_path: Path = typer.Option(
        DEFAULT_PROMPT_FILE,
        "-p",
        "--prompt",
        help="Path to prompt/config file (.json/.json5 or .yaml/.yml) (default: prompt.yaml).",
        show_default=True,
    ),
    new: bool = typer.Option(
        False,
        "--new",
        help="Copy the bundled default_prompt.yaml (next to jisper.py) into the current directory as prompt.yaml.",
        show_default=False,
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
    if new:
        raise typer.Exit(code=write_default_prompt_to_cwd())

    if redo:
        raise typer.Exit(code=redo_last_commit(Path.cwd()))

    if undo:
        raise typer.Exit(code=undo_last_commit(Path.cwd()))

    response, usage, model_code = run(config_path, routine)

    edits = response["edit"]
    replacements = edits.get("replacements", [])
    changed_files = apply_replacements(replacements)

    model_out = response or {}
    edit_section = model_out.get("edit", {}) or {}
    commit_message = (
        as_non_empty_str(model_out.get("commit_message")) or 
        as_non_empty_str(edit_section.get("commit_message")) 
        or "Apply model edits"
    )

    repo = repo_from_dir(Path.cwd())
    in_usd_per_1m, out_usd_per_1m = MODEL_PRICES_USD_PER_1M[model_code]
    pt = dict_get(usage, "prompt_tokens")
    ct = dict_get(usage, "completion_tokens")
    cost_val = (float(pt or 0) * in_usd_per_1m + float(ct or 0) * out_usd_per_1m) / 1_000_000.0 if pt is not None or ct is not None else 0.0
    print(f"${cost_val:.4f}")
    if repo is None:
        print("[yellow]Not a git repository; skipping commit[/yellow]")
        return

    if not changed_files:
        print("[yellow]No files changed; skipping commit[/yellow]")
        return

    committed_message = stage_and_commit(repo, changed_files, commit_message)
    if committed_message:
        print(f"\n[green]Committed changes:[/green] {committed_message}")


def get_model_code(config: dict) -> str:
    return as_non_empty_str(dict_get(config, "model")) or DEFAULT_MODEL







def styled_line_number(ln: int | None, *, width: int = 4, style: str | None = None) -> Text:
    s = f"{ln:>{width}}" if ln is not None else " " * width
    t = Text(s)
    if style and ln is not None:
        t.stylize(style, 0, len(s))
    return t



def load_prompt_file(path: Path) -> dict:
    suffix = (path.suffix or "").lower()
    if suffix in (".yaml", ".yml"):
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_and_concatenate_files(file_list):
    def one(filename: str) -> str | None:
        p = Path(filename)
        txt = read_text_or_none(p)
        if txt is None:
            print(f"[red]Missing input file: {filename}[/red]")
            return None
        return f"--- FILENAME: {filename} ---\n{txt}"

    return "\n\n".join(filter(None, map(one, file_list or [])))


def build_payload(prompt_config: dict, source_text: str, routine_name: str | None = None, *, endpoint_url: str):
    system_instruction = prompt_config.get("system_instruction", "You are a helpful assistant.")
    system_prompt = prompt_config["system_prompt"]
    user_task = resolve_routine_task(prompt_config, routine_name) or prompt_config["task"]
    schema = prompt_config.get("output_schema", DEFAULT_OUTPUT_SCHEMA)
    model_code = get_model_code(prompt_config)

    includes = resolve_included_files(prompt_config)
    structural_section = build_file_summaries_section(includes["structural_level_files"], intent_only=False)
    input_section = build_file_summaries_section(includes["input_level_files"], intent_only=True)

    summaries_parts = list(
        filter(
            None,
            [
                f"STRUCTURAL_LEVEL_FILES:\n{structural_section}" if structural_section else "",
                f"INPUT_LEVEL_FILES:\n{input_section}" if input_section else "",
            ],
        )
    )
    summaries_blob = "\n\n".join(summaries_parts)
    summaries_chunk = f"\n\nFILE SUMMARIES (YAML):\n{summaries_blob}" if summaries_blob else ""

    prompt_content = f"SYSTEM PROMPT:\n{system_prompt}\n\nTASK:\n{user_task}{summaries_chunk}\n\nSOURCE MATERIAL:\n{source_text}"

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
                "name": "response_schema",
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

    header_map = dict(map(lambda kv: (str(kv[0]).lower(), kv[1]), (response_headers or {}).items()))
    prompt_tokens = prompt_tokens or coerce_int(header_map.get("x-openai-prompt-tokens"))
    completion_tokens = completion_tokens or coerce_int(header_map.get("x-openai-completion-tokens"))
    total_tokens = total_tokens or coerce_int(header_map.get("x-openai-total-tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}




def run_build_step(config: dict) -> int | None:
    build_cmd = config.get("build")
    if not build_cmd:
        return None

    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True, cwd=Path.cwd())
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode == 0:
        return 0

    print(f"[red]Build failed with exit code {result.returncode}[/red]")
    return result.returncode


def run(config_path: Path, routine_name: str | None = None) -> tuple[dict, dict, str]:
    config = load_prompt_file(config_path)
    routine_task = resolve_routine_task(config, routine_name)
    if as_non_empty_str(routine_name) and not routine_task:
        routines = dict_get(config, "routines")
        routines = routines if isinstance(routines, dict) else {}
        available = ", ".join(sorted(map(str, routines.keys())))
        print(f"[red]Routine not found:[/red] {routine_name}")
        if available:
            print(f"[yellow]Available routines:[/yellow] {available}")
        raise typer.Exit(code=2)

    build_code = run_build_step(config)
    if isinstance(build_code, int) and build_code != 0:
        raise typer.Exit(code=build_code)

    endpoint_url = as_non_empty_str(dict_get(config, "endpoint")) or DEFAULT_URL
    api_key_env_var = as_non_empty_str(dict_get(config, "api_key_env_var")) or DEFAULT_API_KEY_ENV_VAR
    if "openrouter.ai" in endpoint_url and api_key_env_var == DEFAULT_API_KEY_ENV_VAR:
        api_key_env_var = "OPENROUTER_API_KEY"
    api_key = as_non_empty_str(os.getenv(api_key_env_var or ""))

    includes = resolve_included_files(config)
    concatenated_text = read_and_concatenate_files(includes["source_files"])

    headers = {
        "Authorization": f"Bearer {api_key or ''}",
        "Content-Type": "application/json",
    }

    payload = build_payload(config, concatenated_text, routine_name, endpoint_url=endpoint_url)

    with console.status("Waiting for model...", spinner="dots"):
        response = requests.post(endpoint_url, headers=headers, json=payload)

    if response.status_code != 200:
        print(endpoint_url, headers, payload)
        if response.status_code == 401:
            print("[red]API Error: Authentication failed (401 Unauthorized)[/red]")
            print("\n[yellow]Please check the following:[/yellow]")
            print(f"1. The API key environment variable is set correctly: [cyan]{api_key_env_var}[/cyan]")
            print("2. The API key itself is valid.")
            print(f"3. The endpoint URL is correct: [cyan]{endpoint_url}[/cyan]")
        else:
            print(f"[red]API Error: Received status code {response.status_code}[/red]")
            print(response.text)
        raise typer.Exit(code=1)

    api_json = response.json()
    model_code = get_model_code(config)
    usage = extract_usage_from_api_response(api_json, dict(response.headers))

    choices = api_json.get("choices")
    if choices:
        return (json.loads(choices[0]["message"]["content"]), usage, model_code)
    print("[red]API formatting error[/red]")
    print(api_json)
    exit(1)




def guess_syntax_lexer_name(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50]).lower()

    looks_like_diff = sample.startswith("diff ") or sample.startswith("---") or sample.startswith("+++") or " @@" in sample
    if looks_like_diff:
        return "diff"

    looks_like_json = sample.startswith("{") or sample.startswith("[")
    if looks_like_json:
        return "json"

    looks_like_python = "def " in sample or "import " in sample or "class " in sample
    if looks_like_python:
        return "python"

    return "text"


def guess_syntax_lexer_name_from_filename(filename: str | None) -> str | None:
    name = as_non_empty_str(filename)
    if not name:
        return None

    suffix = Path(name).suffix.lower()
    ext_map = {
        ".py": "python",
        ".json": "json",
        ".json5": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".diff": "diff",
        ".patch": "diff",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".txt": "text",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sql": "sql",
        ".xml": "xml",
    }

    return ext_map.get(suffix)


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


def parse_unified_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    if not (line.startswith("@@") or line.startswith(" @@")):
        return None

    s = line.strip()
    if not (s.startswith("@@") and "@@" in s[2:]):
        return None

    end = s.find("@@", 2)
    if end < 0:
        return None

    inner = s[2:end].strip()
    parts = list(filter(None, inner.split(" ")))
    if len(parts) < 2:
        return None

    old_part = parts[0]
    new_part = parts[1]
    if not (old_part.startswith("-") and new_part.startswith("+")):
        return None

    def parse_range(p: str) -> tuple[int, int] | None:
        core = p[1:]
        if "," in core:
            a, b = core.split(",", 1)
            a_i = coerce_int(a)
            b_i = coerce_int(b)
            return (a_i, b_i) if a_i is not None and b_i is not None else None
        a_i = coerce_int(core)
        return (a_i, 1) if a_i is not None else None

    old_r = parse_range(old_part)
    new_r = parse_range(new_part)
    if old_r is None or new_r is None:
        return None

    old_start, old_len = old_r
    new_start, new_len = new_r
    return (old_start, old_len, new_start, new_len)


def format_combined_diff_lines(
    old_text: str,
    new_text: str,
    *,
    context_lines: int = 3,
    lexer_name: str | None = None,
    filename: str | None = None,
) -> list[tuple[str, Text]]:
    old_lines = old_text.splitlines(keepends=False)
    new_lines = new_text.splitlines(keepends=False)
    diff_lines = list(difflib.unified_diff(old_lines, new_lines, fromfile="a", tofile="b", lineterm="", n=context_lines))

    out: list[tuple[str, Text]] = []

    filename_lexer = guess_syntax_lexer_name_from_filename(filename)
    old_lexer = lexer_name or filename_lexer or guess_syntax_lexer_name(old_text)
    new_lexer = lexer_name or filename_lexer or guess_syntax_lexer_name(new_text)

    def push(kind: str, line: Text):
        out.append((kind, line[:-1]))

    def numbered_line(left_ln: int | None, *, left_style: str | None, mid: str, body: str, lexer: str) -> Text:
        return styled_line_number(left_ln, style=left_style) + Text(mid) + syntax_text(body, lexer_name=lexer)

    old_ln = 1
    new_ln = 1

    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++") or not line:
            continue

        if line.startswith("@@") or line.startswith(" @@"):
            parsed = parse_unified_hunk_header(line)
            if parsed is not None:
                old_ln = parsed[0]
                new_ln = parsed[2]
            continue

        prefix = line[:1]
        body = line[1:]

        if prefix == " ":
            push("context", numbered_line(old_ln, left_style=None, mid="   ", body=body, lexer=old_lexer))
            old_ln += 1
            new_ln += 1
            continue

        if prefix == "-":
            push("delete", numbered_line(old_ln, left_style="bright_red on dark_red", mid=" - ", body=body, lexer=old_lexer))
            old_ln += 1
            continue

        if prefix == "+":
            push("insert", numbered_line(new_ln, left_style="bright_green on dark_green", mid=" + ", body=body, lexer=new_lexer))
            new_ln += 1
            continue

        push("header", Text(line))

    return out


def print_numbered_combined_diff(
    old_text: str,
    new_text: str,
    *,
    context_lines: int = 3,
    title: str | None = None,
    lexer_name: str | None = None,
    filename: str | None = None,
):
    if title:
        console.print(f"\n{title}")

    lines = format_combined_diff_lines(
        old_text,
        new_text,
        context_lines=context_lines,
        lexer_name=lexer_name,
        filename=filename,
    )

    if not lines:
        console.print("[yellow](no diff; content is identical)[/yellow]")
        return

    for _, t in lines:
        console.print(t)


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
            trailing = original[len(original.rstrip()) :]
            replaced_core = stripped_original.replace(trimmed_old, new_string)
            updated = f"{leading}{replaced_core}{trailing}"
            matched_old = trimmed_old

    return (updated, matched_old)


def apply_replacements(replacements, base_dir: Path | None = None) -> list[Path]:
    base_dir = base_dir or Path.cwd()

    def fields(i: int, r: dict) -> tuple[str, str, str] | None:
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

    def apply_one(i: int, r: dict) -> Path | None:
        f = fields(i, r)
        if f is None:
            return None
        filename, old_string, new_string = f

        target_path = (base_dir / filename).resolve()
        original = read_text_or_none(target_path) if target_path.exists() else None

        if original is None and as_non_empty_str(old_string) is None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            print_numbered_combined_diff("", new_string, context_lines=2, title=filename, filename=filename)
            target_path.write_text(new_string, encoding="utf-8")
            return target_path

        if original is None:
            print(f"[red]Target file not found: {target_path}[/red]")
            return None

        updated, _matched_old = apply_one_replacement(original, old_string, new_string)
        if updated is None:
            print(f"[yellow]old_string not found in {filename}; skipping[/yellow]")
            return None

        if updated == original:
            print(f"[yellow]No changes applied to {filename} (replacement produced identical content)[/yellow]")
            return None

        print_numbered_combined_diff(original, updated, context_lines=2, title=filename, filename=filename)
        target_path.write_text(updated, encoding="utf-8")
        return target_path

    return list(filter(None, (apply_one(i, r) for i, r in enumerate(replacements or []))))



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


def undo_last_commit(base_dir: Path) -> int:
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
