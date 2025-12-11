# ai_coding_assistant/main_cli_tool.py

import typer
from rich.console import Console
from rich.text import Text
import json

# Initialize Typer app and Rich console
app = typer.Typer()
console = Console()

CONTEXT_FILE = "context.json"

def load_context():
    """Loads the conversation context from the JSON file."""
    try:
        with open(CONTEXT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_context(context):
    """Saves the conversation context to the JSON file."""
    with open(CONTEXT_FILE, "w") as f:
        json.dump(context, f, indent=2)

@app.command()
def chat(message: str):
    """Simulate a chat turn with the AI assistant."""
    context = load_context()

    # Add user message to context
    context.append({"role": "user", "content": message})
    save_context(context)

    context.append({"role": "assistant", "content": ai_response_content})
    save_context(context)

    console.print(Text(f"{ai_response_content}", style="bold green"))


@app.command()
def show_context():
    """Display the current conversation context."""
    context = load_context()
    if not context:
        console.print(Text("Context is empty.", style="italic yellow"))
        return

    context_text = Text()
    for entry in context:
        role = entry["role"]
        content = entry["content"]
        style = "bold blue" if role == "user" else "bold green"
        context_text.append(f"{role.capitalize()}: ", style=style)
        context_text.append(f"{content}\n")

    console.print(context_text)


@app.command()
def clear_context():
    """Clear the conversation context."""
    save_context([])
    console.print(Text("Context cleared.", style="bold red"))


if __name__ == "__main__":
    app()
