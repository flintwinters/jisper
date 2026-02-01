<!--
[FILE SUMMARY]
context:
  INTENT:
    purpose: >
      Project README documenting what Jisper is, how to configure it, and the structured response
      contract it expects so users can run repeatable, reviewable LLM-driven code edits.

  STRUCTURAL:
    responsibility: >
      Defines the user-facing behavior contract: prompt config keys, file-inclusion semantics,
      optional [FILE SUMMARY] context extraction rules, structured JSON response shape, and git
      behavior expectations.
    boundaries:
      owns:
        - High-level description of the CLI workflow (configure -> send files -> preview -> apply -> commit)
        - Prompt config schema surface (system_prompt/task/routines/file lists/model/endpoint/api key)
        - Explanation of [FILE SUMMARY] blocks and how they are used for context compression
        - Structured response JSON contract and replacement application rules
        - Git and cost-reporting behavior documentation at a user level
      does_not_own:
        - Implementation details of HTTP calls, diff rendering, or file mutation logic (owned by jisper.py)
        - Exact pricing tables and token extraction mechanics beyond user-facing knobs
    entrypoints:
      - name: README
    key_functions:
      - name: Prompt config
        purpose: >
          Documents the configuration surface area that jisper.py loads and enforces.
      - name: Structured response contract
        purpose: >
          Documents the JSON shape required for deterministic application of model edits.
    dependencies:
      - jisper.py
[/FILE SUMMARY]
-->
# Jisper

Jisper is a small CLI for making LLM-driven code edits **repeatable and reviewable**.

Motivation: ad-hoc “paste code into a chat and copy changes back” is hard to audit, easy to misapply, and difficult to reproduce. Jisper turns that workflow into a simple pipeline: load a prompt config, send the relevant files, get back an explicit list of string replacements, preview a diff, apply locally, and commit.

## What it does

- Loads a prompt/config file (YAML or JSON5).
- Reads a configured set of project files as source material.
- Optionally extracts compact context from `[FILE SUMMARY] ... [/FILE SUMMARY]` blocks.
- Calls a Chat Completions endpoint.
- Expects a **structured JSON** response that describes exact string replacements.
- Shows a diff preview, applies the edits, and creates a git commit.

## Motivation

Jisper is intentionally “dumb” about editing: the model does not return a patch, it returns a list of **(filename, old_string, new_string)** operations. That keeps the application step deterministic, makes failures obvious (can’t find `old_string`), and keeps the review loop tight (diff preview before write + git commit when available).

## Prompt config

The config can be YAML (`.yaml`/`.yml`) or JSON5 (`.json`/`.json5`). A minimal YAML example:

```yaml
system_prompt: >
  Provide edits to files by setting values in the replacements array.
  Update the file per the user's request.

task: >
  create a readme for the project

full_files:
  - jisper.py

model: gpt-5.2
endpoint: https://api.openai.com/v1/chat/completions
api_key_env_var: OPENAI_API_KEY
```

### Routines (task overrides)

Define named routines and invoke them by passing the routine name as the positional CLI argument:

```yaml
task: >
  Default task if no routine is provided

routines:
  docs: Write documentation updates
  refactor: Refactor the code to simplify the flow
```

## Included files and `[FILE SUMMARY]` context

You control what gets sent as `SOURCE MATERIAL` via these config keys:

- `full_files`: included in full.
- `structural_level_files`: included in full; if a `[FILE SUMMARY]` block exists, Jisper may include `INTENT` + `STRUCTURAL` context in a compact summaries section.
- `input_level_files`: included in full; if a `[FILE SUMMARY]` block exists, Jisper may include only `INTENT` context.

All lists are concatenated, de-duplicated (keeping order), and read from disk.

A summary block looks like:

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

## Structured response contract

Jisper expects JSON with this shape:

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

Behavior notes:

- Replacements are applied by substring match.
- If `old_string` is not found, Jisper prints a small preview and skips that replacement.
- A whitespace-tolerant fallback is attempted (e.g. trimming `old_string`).

## Git behavior

If Jisper detects a git repo (walking upward from the current directory), it stages changed files and creates a commit using `edit.commit_message` (or `Apply model edits` if missing). Outside a repo it still applies edits, but skips committing.

## Cost reporting

Jisper prints a simple cost line derived from token usage when available. Defaults exist for a small set of model codes, and you can override/add prices via `model_prices_usd_per_1m`:

```yaml
model_prices_usd_per_1m:
  gpt-5.2: [5.0, 15.0]
  my-model: [2.0, 6.0]
```

