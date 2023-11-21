import json
from pathlib import Path
from typing import Any, Dict, Union

_NUMBER_DIGITS = set("0123456789")
PathLike = Union[Path, str]


def is_number(s) -> bool:
    return s and all([c in _NUMBER_DIGITS for c in s])


def load_json(file_path: PathLike) -> Dict[str, Any]:
    file_path = _str_to_path(file_path)

    with file_path.open("r") as f:
        return json.load(f)


def load_binary(file_path: PathLike) -> bytes:
    file_path = _str_to_path(file_path)

    with file_path.open("rb") as f:
        return f.read()


def _str_to_path(path: PathLike) -> Path:
    if isinstance(path, str):
        return Path(path)

    return path
