# AI Coding Assistant

## Overview

This project is a command-line AI coding assistant built with Python, Typer, and Rich. It's designed to be a simple, interactive tool that maintains a conversational context in a static JSON file (`context.json`).

The core design philosophy is to keep the tool lightweight and allow for manual intervention. Users can directly view and edit the JSON context file between interactions, providing a transparent and highly controllable state management system.

## Key Features

- **CLI Interaction**: All interactions happen within the terminal.
- **JSON Context**: Conversation history is stored in a human-readable `context.json` file.
- **Manual Override**: The user has full control to modify the context file manually.
- **Rich & Typer**: Modern CLI libraries for a pleasant user experience.

## Design Preferences

- **Compact Output**: The output is intentionally kept minimal and compact. Rich's `Panel` feature is avoided to prevent unnecessary visual clutter. The goal is an information-dense, clean interface.
- **Simplicity**: The codebase is straightforward, prioritizing readability and ease of modification.
