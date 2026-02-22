# Jisper feature list

A parity checklist for rebuilding Jisper in Go. This list reflects current behavior as implemented in `jisper.py` (and reinforced by `README.md` where consistent).

## CLI / commands

- [x] Single CLI entrypoint (urfave/cli) that runs an end-to-end edit pipeline by default.
- [x] Positional argument `routine` (optional): selects a named task override from config `routines`.
- Options:
  - [x] `--prompt/-p`: path to prompt config file (default `prompt.yaml`).
  - [x] `--new`: copy bundled `default_prompt.yaml` to CWD as `prompt.yaml`.
  - [x] `--undo/-u`: undo last git commit via hard reset to `HEAD~1`.
  - [x] `--redo`: redo undo by hard reset to `ORIG_HEAD` if present.
  - [x] `--debug`: print the exact user-message prompt content immediately before sending.
  - [x] `--no-model`: stop before API request; print payload/prompt for inspection.

## Prompt/config file loading

- [x] Loads config from YAML (`.yaml`/`.yml`).
- [ ] Loads config from JSON5 (uses strict JSON parser for non-YAML; JSON5 extensions not supported).
- Core config keys used by the runtime:
  - [x] `model` (required): model identifier passed to the API.
  - [x] `endpoint` (optional): OpenAI-compatible chat completions URL.
  - [x] `api_key_env_var` (optional): env var name for API key.
  - [x] `system_instruction` (optional): system role message content.
  - [x] `system_prompt` (required): included in the user message.
  - [x] `task` (required): default task (may be overridden by routine).
  - [x] `routines` (optional): mapping of routine name -> task string.
  - [x] `project` (optional): appended to `system_prompt`.
  - [x] `output_schema` (optional): overrides the JSON schema used for `response_format`.
  - [x] `language` (optional): override lexer selection for diff syntax highlighting.
  - [x] `model_prices_usd_per_1m` config override (merges with built-in static map).

## Routine task resolution

- [x] If a `routine` name is provided:
  - [x] Looks up `config.routines[routine]`.
  - [x] If missing, prints an error + list of available routine names and exits with code 2.
- [x] If found, routine task string replaces the default `task` for this run.

## Source material inclusion

- [x] Three inclusion levels (all read from disk, concatenated into one `SOURCE MATERIAL` block):
  - [x] `full_files`: included in full.
  - [x] `structural_level_files`: included in full for reading; contributes *summaries only* to the prompt.
  - [x] `input_level_files`: included in full for reading; contributes *INTENT-only summaries*.
- [x] Path resolution behavior:
  - [x] file paths
  - [x] directory paths (includes direct child files only, sorted by filename)
  - [x] glob patterns (expanded via `filepath.Glob`)
- [x] De-duplicates while preserving first-seen order.
- [x] Prevents duplicates across levels (full > structural > input).

## `[FILE SUMMARY]` extraction and compaction

- [x] Detects and parses YAML inside `[FILE SUMMARY] ... [/FILE SUMMARY]` blocks.
- [x] Summary selection rules:
  - [x] Structural level: includes `context.INTENT` and `context.STRUCTURAL` when present.
  - [x] Input level: includes only `context.INTENT` when present.
- [x] Emits selected summary content as YAML under headers like `--- FILENAME: path ---`.

## Jinja templating

- [x] Always renders `system_prompt` and `task` as Jinja templates using a context derived from config.
- [x] Optional: `render_source_files_as_jinja` enables rendering included source files.
- [x] Jinja context includes:
  - [x] all top-level config keys (shallow copy)
  - [x] `source_text`, `task`, `system_prompt`
  - [ ] build result fields (`build_stdout`, `build_stderr`, `success`, `error`) not injected into context.

## Prompt construction

- [x] Sends two chat messages:
  - [x] `system`: `system_instruction`
  - [x] `user`: a single string that contains:
    - [x] `SYSTEM PROMPT:` + rendered system prompt
    - [x] `TASK:` + rendered task
    - [x] `SOURCE MATERIAL:` + concatenated source material

## Model API call (OpenAI-compatible)

- [x] Makes a single HTTP POST to `endpoint` with JSON body.
- [x] Auth: `Authorization: Bearer <api key from env var>`.
- [x] Requests structured response using `response_format: { type: "json_schema", json_schema: { strict: true, schema: ... } }`.
- [x] Parses model response from `choices[0].message.content`.
  - [x] If content is wrapped in a ```json (or ```) code fence, strips it before JSON parsing.

## Structured response contract (expected)

- [x] Expects top-level JSON with:
  - [x] `edit.commit_message` (string)
  - [x] `edit.replacements` (array of `{ filename, old_string, new_string }`).
- [x] Commit message selection:
  - [x] Prefers `edit.commit_message`; falls back to default `"Apply model edits"`.

## Token usage + cost reporting

- [x] Extracts token usage from either:
  - [x] response JSON `usage.{prompt_tokens, completion_tokens, total_tokens}`.
  - [x] response headers `x-openai-prompt-tokens`, `x-openai-completion-tokens`.
- [x] Computes and prints an estimated USD cost line when tokens present.
  - [x] Uses a built-in per-model USD-per-1M-tokens map.
  - [x] Falls back to default prices when model is unknown.

## Replacement application semantics

- [x] Applies each replacement independently to disk (CWD as base directory).
- [x] Replacement fields required: `filename`, `old_string`, `new_string`.
- [x] Missing fields cause a skip with a message.
- [x] File creation behavior:
  - [x] If target file does not exist and `old_string` is blank, creates file with `new_string`.
  - [x] Creates parent directories as needed.
- [x] Match/replace behavior (deterministic substring replacement):
  - [x] Primary: `old_string` must appear exactly; replaces all occurrences.
  - [x] Fallback 1: tries `old_string` with whitespace trimmed.
  - [x] Fallback 2: compares within `original` with whitespace trimmed, preserves leading/trailing whitespace.
- [x] If no match is found, the replacement is skipped (no write).

## Diff preview rendering

- [x] Before writing each changed file, prints a unified diff preview with:
  - [x] context lines (n=2)
  - [x] stable line numbers for deletes/contexts and inserts.
  - [x] syntax highlighting with lexer guessed from filename extension or `language` config.
- [x] If generated updated content equals original, prints a message and skips writing.

## Git integration

- [x] Repository detection:
  - [x] Walks upward from CWD to find nearest directory containing `.git`.
  - [x] If none is found, initializes a new git repo in CWD.
- [x] After applying replacements:
  - [x] If no files changed, skips commit.
  - [x] Otherwise stages changed files and creates commit using the selected message.
- [x] Undo/redo:
  - [x] Undo: `git reset --hard HEAD~1`.
  - [x] Redo: `git reset --hard ORIG_HEAD` when `ORIG_HEAD` exists.

## Optional build step (post-commit)

- [x] If config key `build` is set:
  - [x] Runs the shell command via `/bin/sh -c <build>`.
  - [x] Streams stdout/stderr live to the terminal.
  - [x] Captures stdout/stderr and writes `build_stdout`/`build_stderr` back to prompt config.
  - [x] Writes `success: true` flag on exit code 0.
  - [x] Writes `error: "build failed (<code>)"` on non-zero exit.
  - [x] Removes existing `build_stdout`, `build_stderr`, `success`, `error` keys before rewriting.

## Misc / defaults

- [x] Default filenames and constants:
  - [x] default prompt file: `prompt.yaml`
  - [x] default endpoint: OpenAI chat completions URL
  - [x] default API key env var: `OPENAI_API_KEY` (auto-switches for OpenRouter)
- [x] Uses UTF-8 for reading/writing text files.


