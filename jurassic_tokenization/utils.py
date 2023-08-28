import json
from pathlib import Path
from typing import Any, Dict

_NUMBER_DIGITS = set("0123456789")


def is_number(s):
    return s and all([c in _NUMBER_DIGITS for c in s])


def load_json(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r") as f:
        return json.load(f)


def load_binary(file_path: Path) -> bytes:
    with file_path.open("rb") as f:
        return f.read()
