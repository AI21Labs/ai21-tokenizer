import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Union, Optional, BinaryIO

from anyio import open_file

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


async def aload_binary(file_path: PathLike) -> bytes:
    async with await open_file(file_path, "rb") as f:
        return await f.read()


async def aread_file_handle(file_handle: Optional[BinaryIO]) -> bytes:
    if file_handle is None:
        raise ValueError("file_handle cannot be None")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, file_handle.read)


def _str_to_path(path: PathLike) -> Path:
    if isinstance(path, str):
        return Path(path)

    return path
