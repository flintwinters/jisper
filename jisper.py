import typer
from rich.console import Console
from rich.text import Text
import json
import os
import google.generativeai as genai
from jinja2 import Environment, FileSystemLoader

app = typer.Typer(no_args_is_help=True)
console = Console()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

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

env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('prompt.jinja')

@app.command()
def chat(message: str):
    """Simulate a chat turn with the AI assistant."""
    context = load_context()

    context.append({"role": "user", "content": message})
    
    # Prepare messages for rendering, including file contents
    rendered_messages = []
    for entry in context:
        if entry["role"] == "file":
            try:
                with open(entry["path"], "r") as f:
                    file_content = f.read()
                rendered_messages.append({"role": "file_content", "path": entry['path'], "content": file_content})
            except FileNotFoundError:
                console.print(Text(f"Warning: File not found: {entry['path']}", style="bold yellow"))
                rendered_messages.append({"role": "error_message", "content": f"Failed to load file: {entry['path']}"})
            except Exception as e:
                console.print(Text(f"Error reading file {entry['path']}: {e}", style="bold red"))
                rendered_messages.append({"role": "error_message", "content": f"Error reading file {entry['path']}: {e}"})
        else:
            rendered_messages.append(entry)

    prompt_text = template.render(messages=rendered_messages)

    response = model.generate_content(prompt_text)
    ai_response_content = response.text

    context.append({"role": "assistant", "content": ai_response_content})
    save_context(context)

    console.print(Text(f"{ai_response_content}", style="bold green"))

@app.command()
def add_file_to_context(file_path: str):
    """Adds a file path to the conversation context."""
    context = load_context()
    context.append({"role": "file", "path": file_path})
    save_context(context)
    console.print(Text(f"Added file '{file_path}' to context.", style="bold blue"))


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
        style = "bold blue" if role == "user" else "bold green"
        
        if role == "file":
            path = entry["path"]
            context_text.append(f"File Path: ", style="bold magenta")
            context_text.append(f"{path}\n")
        else:
            content = entry["content"]
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
