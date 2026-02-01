# Jisper

Jisper is a small CLI that:

- Loads a prompt configuration (YAML or JSON5).
- Reads a set of project files as “source material”.
- Optionally extracts lightweight YAML context from `[FILE SUMMARY] ... [/FILE SUMMARY]` blocks.
- Sends the assembled prompt to a Chat Completions endpoint.
- Expects a structured JSON response describing exact string replacements.
- Applies those replacements locally with a diff preview.
- Optionally stages and commits the changes to git.

This repository’s functionality is implemented in a single script: `jisper.py`.

## Requirements

- Python 3.11+ (recommended).
- An API key in an environment variable (default: `OPENAI_API_KEY`).
- A prompt config file (default: `prompt.yaml`).

## Usage

Run with the default prompt file (`prompt.yaml`) and default task:

```bash
python jisper.py
```

Use a specific prompt/config file:

```bash
python jisper.py --prompt prompt.yaml
# or
python jisper.py -p prompt.json5
```

Run a named routine (a task override from `routines` in the config):

```bash
python jisper.py my_routine_name
```

Undo the last commit created (hard reset to `HEAD~1`):

```bash
python jisper.py --undo
```

Redo an undo (reset to `ORIG_HEAD` when available):

```bash
python jisper.py --redo
```

## How it works

1. Jisper loads your prompt config.
2. It builds the prompt content:
   - `SYSTEM PROMPT:` from `system_prompt` in the config.
   - `TASK:` from `task` (or a routine override).
   - `FILE SUMMARIES (YAML):` extracted from `[FILE SUMMARY]` blocks (optional; controlled by which files you include).
   - `SOURCE MATERIAL:` concatenated contents of the included files.
3. It calls the configured chat completions endpoint.
4. It parses the model response as JSON and reads `edit.replacements`.
5. For each replacement it:
   - Loads the target file.
   - Replaces `old_string` with `new_string`.
   - Prints a diff preview.
   - Writes the updated file.
6. If the current directory is a git repo and any files changed, it stages and commits them.

## Prompt config format

The config can be YAML (`.yaml`/`.yml`) or JSON5 (`.json`/`.json5`).

### Minimal YAML example

```yaml
system_prompt: >
  Provide edits to files by setting values in the replacements array.
  Update the file per the user's request.

task: >
  create a readme for the project

full_files:
  - jisper.py

# Optional: use a model/endpoint override
model: gpt-5.2
endpoint: https://api.openai.com/v1/chat/completions
api_key_env_var: OPENAI_API_KEY
```

### Routines (task overrides)

You can define named routines and invoke them by passing the routine name as the positional CLI argument.

```yaml
task: >
  Default task if no routine is provided

routines:
  docs:
    Write documentation updates
  refactor:
    Refactor the code to simplify the flow
```

## Included files and file summaries

You control which files are concatenated into `SOURCE MATERIAL` via these config keys:

- `full_files`: included in full.
- `structural_level_files`: included in full, and also scanned for `[FILE SUMMARY]` blocks; if present, both `INTENT` and `STRUCTURAL` context may be included in the `FILE SUMMARIES` section.
- `input_level_files`: included in full, and also scanned for `[FILE SUMMARY]` blocks; if present, only the `INTENT` context may be included in the `FILE SUMMARIES` section.

All three lists are concatenated, de-duplicated (keeping order), and read from disk.

### `[FILE SUMMARY]` blocks

If a file contains a block like:

```text
[FILE SUMMARY]
context:
  INTENT:
    purpose: >
      ...
  STRUCTURAL:
    responsibility: >
      ...
[/FILE SUMMARY]
```

Jisper can extract and forward that YAML as compact context.

## Structured response contract

Jisper expects the model to return JSON matching this shape:

```json
{
      "edit": {
        "explanation": "1-2 sentence explanation",
    "commit_message": "Commit subject",
    "replacements": [
          {
            "filename": "README.md",
        "old_string": "...",
        "new_string": "..."
      }
    ]
  }
}
```

Notes:

- Replacements are applied by exact substring match.
- If `old_string` is not found, Jisper prints a small preview and skips that replacement.
- A whitespace-tolerant fallback is attempted (e.g. trimming `old_string`).

## Git behavior

- If Jisper detects a git repo (walking upward from the current directory), it will:
  - stage any changed files from the replacements, and
  - create a commit using the model-provided `edit.commit_message` (or `Apply model edits` if missing).
- If not in a git repo, it will still apply the changes but skip committing.

## Cost reporting

Jisper prints a simple cost line derived from token usage when available, using configured model prices:

- Built-in defaults exist for a small set of model codes.
- You can override or add prices via `model_prices_usd_per_1m` in the config.

Example:

```yaml
model_prices_usd_per_1m:
  gpt-5.2: [5.0, 15.0]
  my-model: [2.0, 6.0]
```
