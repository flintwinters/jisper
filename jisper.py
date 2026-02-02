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
      - name: build_payload
        signature: build_payload(prompt_config: dict, source_text: str, routine_name: str | None = None)
        purpose: >
          Constructs the chat completions payload including system/user messages and optional JSON schema.
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
from rich import print
from rich.console import Console
from rich.text import Text
from rich.syntax import Syntax


# soft_wrap=True is VERY IMPORTANT for ergonomics.  DO NOT CHANGE
console = Console(soft_wrap=True, markup=False, highlight=False, no_color=False)
app = typer.Typer(add_completion=False)

DEFAULT_PROMPT_FILE = "prompt.yaml"
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

def get_routines_map(config: dict) -> dict:
    v = dict_get(config, "routines")
    return v if isinstance(v, dict) else {}

def resolve_routine_task(config: dict, routine_name: str | None) -> str | None:
    name = as_non_empty_str(routine_name)
    if not name:
        return None

    routines = get_routines_map(config)
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

        p = (base_dir / s)
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

def subtract_with_order(primary: list[str], subtract: set[str]) -> list[str]:
    return list(filter(lambda s: s not in subtract, primary or []))

def resolve_included_files(config: dict) -> dict:
    base_dir = Path.cwd()
    full_raw = as_list_of_non_empty_str(dict_get(config, "full_files"))
    structural_raw = as_list_of_non_empty_str(dict_get(config, "structural_level_files"))
    input_raw = as_list_of_non_empty_str(dict_get(config, "input_level_files"))

    input_level_files = resolve_paths_and_globs(input_raw, base_dir=base_dir)
    structural_level_files = resolve_paths_and_globs(structural_raw, base_dir=base_dir)
    lower_level_set = set(input_level_files + structural_level_files)

    full_files = resolve_paths_and_globs(full_raw, base_dir=base_dir)
    full_files = subtract_with_order(full_files, lower_level_set)

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
    inner = inner.strip("\n")
    inner = inner.strip()
    if not inner:
        return None

    loaded = yaml.safe_load(inner)
    return loaded if isinstance(loaded, dict) else None

def dump_yaml_block(d: dict) -> str:
    return yaml.safe_dump(d, sort_keys=False).rstrip() if isinstance(d, dict) else ""

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

        dumped = dump_yaml_block(selected)
        return f"--- FILENAME: {p.name} ---\n{dumped}" if dumped else None

    joined = "\n\n".join(filter(None, map(one, files or [])))
    return joined.strip() if joined else ""

@app.callback(invoke_without_command=True)
def main(
    routine: str | None = typer.Argument(
        None,
        help="Optional routine name to use from prompt config's `routines` mapping (overrides default task).",
        show_default=False,
    ),
    config: Path = typer.Option(
        DEFAULT_PROMPT_FILE,
        "-p",
        "--prompt",
        help="Path to prompt/config file (.json/.json5 or .yaml/.yml) (default: prompt.yaml).",
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

    response, usage, model_code = run(config, routine)
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
    return line.startswith("---") or line.startswith("+++")


def is_hunk_header_line(line: str) -> bool:
    return line.startswith("@@")


def is_diff_meta_line(line: str) -> bool:
    return is_file_header_line(line)


def format_fixed_width_line_number(ln: int | None, *, width: int = 4) -> str:
    if ln is None:
        return " " * width
    return f"{ln:>{width}}"


def styled_line_number(ln: int | None, *, width: int = 4, style: str | None = None) -> Text:
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
    suffix = (path.suffix or "").lower()
    if suffix in (".yaml", ".yml"):
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}
    return read_json5(path)


def read_and_concatenate_files(file_list):
    def one(filename: str) -> str | None:
        p = Path(filename)
        txt = read_text_or_none(p)
        if txt is None:
            print(f"[red]Missing input file: {filename}[/red]")
            return None
        return f"--- FILENAME: {p.name} ---\n{txt}"

    return "\n\n".join(filter(None, map(one, file_list or [])))

def build_payload(prompt_config: dict, source_text: str, routine_name: str | None = None):
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
    pt = dict_get(usage, "prompt_tokens")
    ct = dict_get(usage, "completion_tokens")
    cost = estimate_cost_usd(pt, ct, in_usd_per_1m, out_usd_per_1m) or 0.0
    return f"[bright_black]${cost:.4f}[/bright_black]"


def run(config_path: Path, routine_name: str | None = None) -> tuple[dict, dict, str]:
    config = load_prompt_file(config_path)
    routine_task = resolve_routine_task(config, routine_name)
    if as_non_empty_str(routine_name) and not routine_task:
        routines = get_routines_map(config)
        available = ", ".join(sorted(map(str, routines.keys())))
        print(f"[red]Routine not found:[/red] {routine_name}")
        if available:
            print(f"[yellow]Available routines:[/yellow] {available}")
        raise typer.Exit(code=2)

    endpoint_url = get_endpoint_url(config)
    api_key_env_var = get_api_key_env_var_name(config)
    api_key = get_api_key_from_env(api_key_env_var)

    includes = resolve_included_files(config)
    concatenated_text = read_and_concatenate_files(includes["source_files"])

    headers = {
        "Authorization": f"Bearer {api_key or ''}",
        "Content-Type": "application/json",
    }

    payload = build_payload(config, concatenated_text, routine_name)

    with console.status("Waiting for model...", spinner="dots"):
        response = requests.post(endpoint_url, headers=headers, json=payload)

    api_json = response.json()
    model_code = get_model_code(config)
    usage = extract_usage_from_api_response(api_json, dict(response.headers))
    model_out = json.loads(api_json["choices"][0]["message"]["content"])
    return (model_out, usage, model_code)


def print_no_diff_notice():
    console.print("[yellow](no diff; content is identical)[/yellow]")


def unified_diff_lines(
    old_text: str,
    new_text: str,
    *,
    context_lines: int = 3,
    fromfile: str = "a",
    tofile: str = "b",
) -> list[str]:
    old_lines = old_text.splitlines(keepends=False)
    new_lines = new_text.splitlines(keepends=False)
    return list(difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, lineterm="", n=context_lines))


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
    if not is_hunk_header_line(line):
        return None

    s = line.strip()
    if not (s.startswith("@@") and s.endswith("@@")):
        at = s.find("@@", 2)
        if at < 0:
            return None
        s = s[: at + 2]

    inner = s.strip("@ ")
    parts = inner.split(" ")
    parts = list(filter(None, parts))
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
    diff_lines = unified_diff_lines(old_text, new_text, context_lines=context_lines)

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
        if is_file_header_line(line) or not line:
            continue

        if is_hunk_header_line(line):
            parsed = parse_unified_hunk_header(line)
            if parsed is not None:
                old_ln = parsed[0]
                new_ln = parsed[2]
            continue

        prefix = line[:1]
        body = line[1:]

        if prefix == " ":
            push("context", numbered_line(new_ln, left_style=None, mid="   ", body=body, lexer=new_lexer))
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
        print_no_diff_notice()
        return

    for _, t in lines:
        console.print(t)


def print_change_preview(filename: str, original: str, updated: str):
    print_numbered_combined_diff(
        original,
        updated,
        context_lines=2,
        title=filename,
        filename=filename,
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
            trailing = original[len(original.rstrip()) :]
            replaced_core = stripped_original.replace(trimmed_old, new_string)
            updated = f"{leading}{replaced_core}{trailing}"
            matched_old = trimmed_old

    return (updated, matched_old)


def apply_replacements(replacements, base_dir: Path | None = None) -> list[Path]:
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
        print_numbered_combined_diff(
            old_string,
            new_string,
            context_lines=2,
            title="Replacement text (preview)",
            filename=filename,
        )

    def can_create_missing_file(old_string: str) -> bool:
        return as_non_empty_str(old_string) is None

    def apply_one(i_r) -> Path | None:
        i, r = i_r
        fields = get_fields(i, r or {})
        if fields is None:
            return None
        filename, old_string, new_string = fields

        target_path = (base_dir / filename).resolve()
        original = read_text_or_none(target_path)
        if original is None and target_path.exists() is False:
            if can_create_missing_file(old_string):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                print_change_preview(filename, "", new_string)
                target_path.write_text(new_string, encoding="utf-8")
                return target_path
            print(f"[red]Target file not found: {target_path}[/red]")
            return None

        if original is None:
            print(f"[red]Target file not found: {target_path}[/red]")
            return None

        updated, _matched_old = apply_one_replacement(original, old_string, new_string)
        if updated is None:
            preview_missing_old(filename, old_string, new_string)
            return None

        if updated == original:
            print(f"[yellow]No changes applied to {filename} (replacement produced identical content)[/yellow]")
            return None

        print_change_preview(filename, original, updated)
        target_path.write_text(updated, encoding="utf-8")
        return target_path

    return list(filter(None, map(apply_one, enumerate(replacements or []))))


def print_model_change_notes(model_output: dict):
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
