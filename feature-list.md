# Jisper feature list

A parity checklist for rebuilding Jisper in Go. This list reflects current behavior as implemented in `jisper.py` (and reinforced by `README.md` where consistent).

## CLI / commands

- Single CLI entrypoint (Typer) that runs an end-to-end edit pipeline by default.
- Positional argument `routine` (optional): selects a named task override from config `routines`.
- Options:
  - `--prompt/-p`: path to prompt config file (default `prompt.yaml`).
  - `--new`: copy bundled `default_prompt.yaml` (next to `jisper.py`) to CWD as `prompt.yaml` (refuses to overwrite).
  - `--undo/-u`: undo last git commit via hard reset to `HEAD~1`.
  - `--redo`: redo undo by hard reset to `ORIG_HEAD` if present.
  - `--debug`: print the exact user-message prompt content immediately before sending.
  - `--no-model`: stop before API request; print payload/prompt for inspection.

## Prompt/config file loading

- Loads config from YAML (`.yaml`/`.yml`) or JSON5 (all other suffixes treated as JSON5).
- Core config keys used by the runtime:
  - `model` (required): model identifier passed to the API.
  - `endpoint` (optional): OpenAI-compatible chat completions URL (default `https://api.openai.com/v1/chat/completions`).
  - `api_key_env_var` (optional): env var name for API key (default `OPENAI_API_KEY`; auto-switches default to `OPENROUTER_API_KEY` when endpoint contains `openrouter.ai`).
  - `system_instruction` (optional): system role message content (default `"You are a helpful assistant."`).
  - `system_prompt` (required): included in the user message.
  - `task` (required): default task (may be overridden by routine).
  - `routines` (optional): mapping of routine name -> task string.
  - `project` (optional): appended to `system_prompt`.
  - `output_schema` (optional): overrides the JSON schema used for `response_format`.
  - `language` (optional): override lexer selection for diff syntax highlighting.
  - `model_prices_usd_per_1m` is documented in README, but current code uses an internal static map + fallback prices.

## Routine task resolution

- If a `routine` name is provided:
  - Looks up `config.routines[routine]`.
  - If missing, prints an error + list of available routine names and exits with code 2.
- If found, routine task string replaces the default `task` for this run.

## Source material inclusion

- Three inclusion levels (all read from disk, concatenated into one `SOURCE MATERIAL` block):
  - `full_files`: included in full.
  - `structural_level_files`: included in full for reading; contributes *summaries only* to the prompt when a `[FILE SUMMARY]` block is present.
  - `input_level_files`: included in full for reading; contributes *INTENT-only summaries* when a `[FILE SUMMARY]` block is present.
- Path resolution behavior:
  - Values can be:
    - file paths
    - directory paths (includes direct child files only, sorted by filename)
    - glob patterns (expanded via `Path.glob`)
  - De-duplicates while preserving first-seen order.
  - Prevents duplicates across levels (full > structural > input).

## `[FILE SUMMARY]` extraction and compaction

- Detects and parses YAML inside `[FILE SUMMARY] ... [/FILE SUMMARY]` blocks.
- Summary selection rules:
  - Structural level: includes `context.INTENT` and `context.STRUCTURAL` when present.
  - Input level: includes only `context.INTENT` when present.
- Emits selected summary content as YAML under headers like `--- FILENAME: path ---`.

## Jinja templating

- Always renders `system_prompt` and `task` as Jinja templates using a context derived from config + computed fields.
- Optional: `render_source_files_as_jinja` enables rendering included source files (and their `[FILE SUMMARY]` blocks) as Jinja templates too.
- Jinja context includes:
  - all top-level config keys (shallow copy)
  - `source_text`, `task`, `system_prompt`
  - build result fields when present in config: `build_stdout`, `build_stderr`, `success`, `error`

## Prompt construction

- Sends two chat messages:
  - `system`: `system_instruction`
  - `user`: a single string that contains:
    - `SYSTEM PROMPT:` + rendered system prompt
    - `TASK:` + rendered task
    - `SOURCE MATERIAL:` + concatenated source material (full text and/or summaries)

## Model API call (OpenAI-compatible)

- Makes a single HTTP POST to `endpoint` with JSON body containing `model` and `messages`.
- Auth: `Authorization: Bearer <api key from env var>`.
- Requests a structured response using `response_format: { type: "json_schema", json_schema: { strict: true, schema: ... } }` when a schema is present.
- Parses model response from `choices[0].message.content`.
  - If content is wrapped in a ```json (or ```) code fence, strips it before JSON5 parsing.

## Structured response contract (expected)

- Expects top-level JSON with:
  - `edit.commit_message` (string)
  - `edit.replacements` (array of `{ filename, old_string, new_string }`)
- Commit message selection:
  - Prefers `edit.commit_message`; falls back to a default of `"Apply model edits"`.

## Token usage + cost reporting

- Extracts token usage from either:
  - response JSON `usage.{prompt_tokens, completion_tokens, total_tokens}`, or
  - response headers `x-openai-prompt-tokens`, `x-openai-completion-tokens`, `x-openai-total-tokens`.
- Computes and prints an estimated USD cost line when prompt or completion tokens are present.
  - Uses a built-in per-model (input, output) USD-per-1M-tokens map.
  - Falls back to default prices when model is unknown.

## Replacement application semantics

- Applies each replacement independently to disk (CWD as base directory):
  - Replacement fields required: `filename`, `old_string`, `new_string`.
  - Missing fields cause a skip with a message.
- File creation behavior:
  - If the target file does not exist and `old_string` is blank/whitespace, creates the file with `new_string`.
  - Creates parent directories as needed.
- Match/replace behavior (deterministic substring replacement):
  - Primary: `old_string` must appear exactly; replaces all occurrences via `str.replace`.
  - Fallback 1: tries `old_string.strip()` against original.
  - Fallback 2: compares within `original.strip()` and performs replacement while preserving original leading/trailing whitespace.
- If no match is found, the replacement is skipped (no write).

## Diff preview rendering

- Before writing each changed file, prints a unified diff preview with:
  - context lines (n=2)
  - stable left-side line numbers for deletes/contexts and right-side line numbers for inserts
  - syntax highlighting with lexer guessed from filename extension (or overridden by config `language`).
- If generated updated content equals original, prints a message and skips writing.

## Git integration

- Repository detection:
  - Walks upward from CWD to find the nearest directory containing `.git`.
  - If none is found, initializes a new git repo in CWD.
- After applying replacements:
  - If no files changed, skips commit.
  - Otherwise stages changed files (relative to repo root) and creates a commit using the selected commit message.
- Undo/redo:
  - Undo: `git reset --hard HEAD~1`.
  - Redo: `git reset --hard ORIG_HEAD` when `ORIG_HEAD` exists.

## Optional build step (post-commit)

- If config key `build` is set:
  - Runs the shell command via `/bin/sh -c <build>`.
  - Streams stdout/stderr live to the terminal while capturing both.
  - Normalizes captured output for storage (CR update collapse, ANSI stripping, removal of empty lines).
  - Writes `build_stdout` and `build_stderr` back into the prompt YAML file as literal block scalars.
  - Writes a `success: true` flag on exit code 0; otherwise writes `error: "build failed (<code>)"`.
  - Removes any existing `build_stdout`, `build_stderr`, `success`, `error` keys before rewriting.

## Misc / defaults

- Default filenames and constants:
  - default prompt file: `prompt.yaml`
  - default template prompt file: `default_prompt.yaml`
  - default endpoint: OpenAI chat completions URL
- Uses UTF-8 for reading/writing text files.


