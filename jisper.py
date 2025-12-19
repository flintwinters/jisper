import typer
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
import json
import os
import google.generativeai as genai
from jinja2 import Environment, FileSystemLoader
from typing import Optional
from rich.live import Live
from rich.spinner import Spinner
import diff_match_patch as dmp_module
import subprocess

# Pricing for gemini-1.5-flash (as a placeholder for gemini-2.5-flash)
# Prices per 1 million tokens in USD
INPUT_PRICE_PER_1M_TOKENS = 0.35
OUTPUT_PRICE_PER_1M_TOKENS = 0.70

app = typer.Typer(no_args_is_help=False)
console = Console()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

RESPONSE_SCHEMA = json.loads(open("response_schema.json", "r").read())

model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config=genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA
    )
)

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

def replace_in_file(filename, old_string, new_string):
    """Replaces a string in a file."""
    with open(filename, "r") as f:
        content = f.read()
    
    if old_string not in content:
        raise ValueError("The specified old_string was not found in the file.")

    new_content = content.replace(old_string, new_string, 1)

    with open(filename, "w") as f:
        f.write(new_content)

def print_diff(old_string, new_string):
    """Prints a colorful, character-level diff of the changes."""
    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(old_string, new_string)
    dmp.diff_cleanupSemantic(diffs)

    text = Text()
    for op, data in diffs:
        if op == dmp.DIFF_INSERT:
            text.append(data, style="on #006600")
        elif op == dmp.DIFF_DELETE:
            text.append(data, style="on #660000")
        elif op == dmp.DIFF_EQUAL:
            text.append(data)
    
    console.print(text)


env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('prompt.jinja')

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    message: Optional[str] = typer.Argument(None, help="The message to send to the AI assistant."),
    add_file: Optional[str] = typer.Option(None, "--add-file", "-f", help="Path to the file to add to context."),
    show_context_flag: bool = typer.Option(False, "--show-context", "-s", help="Display the current conversation context.", is_flag=True),
    clear_context_flag: bool = typer.Option(False, "--clear-context", "-c", help="Clear the conversation context.", is_flag=True),
    undo_flag: bool = typer.Option(False, "--undo", "-u", help="Revert the last commit.", is_flag=True),
):
    """
    A CLI for interacting with a Gemini-powered AI assistant.
    """
    if show_context_flag:
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

            if role == "assistant" and "turn_cost" in entry:
                input_tokens = entry.get("input_tokens", "N/A")
                output_tokens = entry.get("output_tokens", "N/A")
                turn_cost = entry.get("turn_cost", 0)
                total_cost = entry.get("total_cost", 0)
                
                metadata_text = Text()
                metadata_text.append(f"    Tokens: {input_tokens} in, {output_tokens} out. ", style="italic dim")
                metadata_text.append(f"Cost: ${turn_cost:.6f} (Turn), ${total_cost:.6f} (Total)\n", style="italic dim")
                context_text.append(metadata_text)

        console.print(context_text)
    elif clear_context_flag:
        save_context([])
        console.print(Text("Context cleared.", style="bold red"))
    elif undo_flag:
        try:
            console.print(Text("Attempting to revert the last commit...", style="italic yellow"))
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
            console.print(Text("Successfully reverted the last commit.", style="bold green"))
        except subprocess.CalledProcessError as e:
            console.print(Text(f"Error reverting commit: {e}", style="bold red"))
    elif add_file:
        context = load_context()
        context.append({"role": "file", "path": add_file})
        save_context(context)
        console.print(Text(f"Added file '{add_file}' to context.", style="bold blue"))
    elif message:
        context = load_context()
        context.append({"role": "user", "content": message})
        
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
        
        with Live(console=console, screen=False, auto_refresh=True, transient=True) as live:
            live.update(Spinner("pipe", speed=10.0))
            response = model.generate_content(prompt_text)
        
        ai_response_json = json.loads(response.text)
        ai_response_content = ai_response_json.get("response_text", "")
        ai_edit = ai_response_json.get("edit")
        
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M_TOKENS
        output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M_TOKENS
        turn_cost = input_cost + output_cost

        # Find the last total_cost to calculate the new total_cost
        last_total_cost = 0
        for entry in reversed(context):
            if entry.get("role") == "assistant" and "total_cost" in entry:
                last_total_cost = entry["total_cost"]
                break
        
        total_cost = last_total_cost + turn_cost

        assistant_response = {
            "role": "assistant",
            "content": ai_response_content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "turn_cost": turn_cost,
            "total_cost": total_cost,
        }
        
        context.append(assistant_response)
        save_context(context)
        console.print(Markdown(ai_response_content))

        # Print usage metadata
        metadata_text = Text()
        metadata_text.append(f"Tokens: {input_tokens} in, {output_tokens} out. ", style="italic dim")
        metadata_text.append(f"Cost: ${turn_cost:.6f} (Turn), ${total_cost:.6f} (Total)", style="italic dim")
        console.print(metadata_text)
        
        if ai_edit:
            filename = ai_edit.get("filename")
            old_string = ai_edit.get("old_string")
            new_string = ai_edit.get("new_string")

            if filename and old_string and new_string:
                try:
                    replace_in_file(filename, old_string, new_string)
                    console.print(Text(f"Applied edit to {filename}:", style="bold blue"))
                    print_diff(old_string, new_string)
                except Exception as e:
                    console.print(Text(f"Error applying edit to {filename}: {e}", style="bold red"))
            else:
                console.print(Text("Warning: Malformed edit object received from AI.", style="bold red"))
    else:
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
