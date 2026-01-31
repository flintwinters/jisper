import json
import os
import amazon.ion.simpleion as ion
from rich.console import Console
from rich.syntax import Syntax


def load_ion(path):
    with open(path, "rb") as f:
        return ion.load(f)


ION_TO_JSON_TYPES = {
    "string": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "decimal": "number",
    "timestamp": "string",
}


def ion_type_to_json_type(ion_type):
    return ION_TO_JSON_TYPES.get(ion_type.text, "string")


def ion_schema_to_json_schema(ion_schema):
    properties = {}

    for field, ion_type in ion_schema["fields"].items():
        properties[field] = {"type": ion_type_to_json_type(ion_type)}

    return {
        "type": "object",
        "properties": properties,
        "required": ion_schema.get("required_fields", []),
        "additionalProperties": False,
    }


def main():
    schema_path = os.environ.get("SCHEMA_ION_PATH", "schema.ion")
    ion_data = load_ion(schema_path)
    rendered = json.dumps(ion_schema_to_json_schema(ion_data), indent=2)
    console = Console()
    console.print(Syntax(rendered, "json", theme="monokai", word_wrap=False))


if __name__ == "__main__":
    main()