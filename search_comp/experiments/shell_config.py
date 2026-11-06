"""Typed settings from independent Bash launchers; never evaluate shell text."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]


def decode(value, kind, name):
    if kind == "str":
        return value
    if kind == "path":
        result = Path(value).expanduser()
        return result if result.is_absolute() else ROOT / result
    if kind == "optional" and value == "":
        return None
    try:
        if kind == "bool":
            if value.lower() not in {"true", "false", "1", "0"}:
                raise ValueError("use true/false or 1/0")
            return value.lower() in {"true", "1"}
        parsed = json.loads(value)
    except (ValueError, TypeError) as error:
        if kind == "optional":
            return value
        raise ValueError(f"{name}: invalid {kind} value {value!r}") from error
    valid = {"int": type(parsed) is int, "float": type(parsed) in (int, float),
             "list": isinstance(parsed, list), "tuple": isinstance(parsed, list), "optional": True}
    if not valid.get(kind, False):
        raise ValueError(f"{name}: expected {kind}, got {value!r}")
    return tuple(parsed) if kind == "tuple" else float(parsed) if kind == "float" else parsed


def assemble(schema, values):
    result = {}
    for key, descriptor in schema.items():
        name = descriptor["env"]
        if name not in values:
            raise ValueError(f"Missing {name}; launch the matching sh file")
        cursor = result
        parts = key.split("__")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part.lower() if cursor is not result else part, {})
        # Top-level Python names remain uppercase; nested configuration keys remain lowercase.
        cursor[parts[-1] if len(parts) == 1 or parts[0] == "ENVIRONMENT" else parts[-1].lower()] = decode(
            values[name], descriptor["type"], name)
    return result


def load_settings(filename, schema):
    filename = Path(filename).resolve()
    identity = f"{filename.parent.parent.name}/{filename.stem}"
    if os.environ.get("SA_ENTRY") != identity:
        raise RuntimeError(f"Run scripts/{filename.parent.parent.name}/sh/{filename.stem}.sh; edit parameters there")
    return assemble(schema, os.environ)


def launcher_defaults(filename, schema):
    """Read literal exported defaults for static checks without executing Bash."""
    filename = Path(filename)
    values = {}
    for line in filename.read_text(encoding="utf-8").splitlines():
        if not line.startswith("export "):
            continue
        fields = shlex.split(line, comments=True)
        if len(fields) != 2 or "=" not in fields[1]:
            continue
        key, value = fields[1].split("=", 1)
        # Credential exports deliberately inherit values at run time; inspect with empty credentials.
        if value.startswith("${"):
            value = ""
        values[key] = value
    return assemble(schema, values)
