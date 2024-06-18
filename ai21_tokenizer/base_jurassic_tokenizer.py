from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, BinaryIO, Tuple, Union

import sentencepiece as spm

from abc import ABC
from ai21_tokenizer.file_utils import is_number, PathLike, load_json

_MODEL_EXTENSION = ".model"
_MODEL_CONFIG_FILENAME = "config.json"


@dataclass
class SpaceSymbol:
    tok_id: int
    count: int


class BaseJurassicTokenizer(ABC):
    _sp: spm.SentencePieceProcessor = None
    _id_to_token_map: Dict[int, str] = None
    _token_to_id_map: Dict[str, int] = None
    _no_show_tokens: set[str] = None
    _newline_id: int = 0
    _space_tokens: List[SpaceSymbol] = None

    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        config: Optional[Dict[str, Any]] = None,
        model_file_handle: Optional[BinaryIO] = None,
    ):
        self._model_path = model_path
        self._model_file_handle = model_file_handle
        self._validate_init(model_path=model_path, model_file_handle=model_file_handle)

        config = self._get_config(model_path=model_path, config=config)

        self.pad_id = config.get("pad_id")
        self.unk_id = config.get("unk_id")
        self.eop_id = config.get("eop_id")
        self.bos_id = config.get("bos_id")
        self.eos_id = config.get("eos_id")

        self._newline_piece = config.get("newline_piece")
        self._mask_pieces = config.get("mask_pieces", [])

        self._manual_add_dummy_prefix = not (config.get("add_dummy_prefix", True))

        self._sample_split = re.compile(r"▁*[^▁]+|▁")
        self._space_split = re.compile("( {2,})")  # Split by 2 or more consecutive spaces

        self._number_mode = config.get("number_mode")
        self._space_mode = config.get("space_mode")

    def _validate_init(self, model_path: Optional[PathLike], model_file_handle: Optional[BinaryIO]) -> None:
        if model_path is None and model_file_handle is None:
            raise ValueError("Must provide exactly one of model_path or model_file_handle. Got none.")

        if model_path is not None and model_file_handle is not None:
            raise ValueError("Must provide exactly one of model_path or model_file_handle. Got both.")

    def _get_model_file(self, model_path: PathLike) -> PathLike:
        model_path = Path(model_path)

        if model_path.is_dir():
            return model_path / f"{model_path.name}{_MODEL_EXTENSION}"

        return model_path

    def _get_config(self, model_path: Optional[PathLike], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if model_path and Path(model_path).is_dir():
            config_path = model_path / _MODEL_CONFIG_FILENAME
            return load_json(config_path)

        return config or {}

    def _map_space_tokens(self) -> List[SpaceSymbol]:
        res = []
        for count in range(32, 0, -1):
            tok_id = self._token_to_id("▁" * count)
            if tok_id != self.unk_id:
                res.append(SpaceSymbol(tok_id=tok_id, count=count))

        return res

    def _encode(self, text: str) -> List[int]:
        if self._space_mode is None:
            return self._sp.encode(text)

        res = []
        text = text.replace("\t", " ")
        remainder = ""

        for sub_text in self._space_split.split(text):
            if not sub_text:
                continue

            if not sub_text.startswith("  "):
                res.extend(self._sp.encode(remainder + sub_text))
                remainder = ""
                continue

            remaining = len(sub_text) - 1
            remainder = " "
            space_index = 0

            while remaining:
                while self._space_tokens[space_index].count > remaining:
                    space_index += 1

                remaining -= self._space_tokens[space_index].count
                res.append(self._space_tokens[space_index].tok_id)

        if remainder:
            res.extend(self._sp.encode(remainder))

        return res

    def _tokenize_number(self, num: str, mode: str) -> list[int]:
        if mode.endswith("_keep"):
            # If the full number is in the vocab in keep mode, just use it
            single_id = self._token_to_id(num)

            if single_id != self.unk_id:
                return [single_id]

            mode = mode.rstrip("_keep")

        if mode in ["left", "right"]:
            res = []

            if mode == "right":
                offset = len(num) % 3

                if offset:
                    res.append(self._token_to_id(num[:offset]))
                    num = num[offset:]

            res += [self._token_to_id(num[i : i + 3]) for i in range(0, len(num), 3)]
        else:
            raise ValueError(f"Invalid number mode: {mode}")

        return res

    def _encode_post_process(self, ids: List[int]) -> List[int]:
        if self._number_mode is None:
            return ids

        i = 0
        res = []

        while i < len(ids):
            token = self._id_to_token(ids[i])
            if not is_number(token):
                res.append(ids[i])
                i += 1
            else:
                num = ""

                while i < len(ids):
                    token = self._id_to_token(ids[i])
                    if not is_number(token):
                        break
                    num += token
                    i += 1

                res.extend(self._tokenize_number(num, self._number_mode))

        return res

    def _id_to_token(self, token_id: int) -> str:
        return self._id_to_token_map[token_id]

    def _convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return [self._id_to_token(token_id) for token_id in token_ids]

    def _token_to_id(self, token: str) -> int:
        return self._token_to_id_map.get(token, self.unk_id)

    def _encode_wrapper(self, text: str, **kwargs) -> List[int]:
        is_start = kwargs.get("is_start", True)
        lines = text.split("\n")
        toks = []

        for i, line in enumerate(lines):
            if i > 0:
                toks.append(self.newline_id)
            if not line:
                continue
            # We add the dummy prefix on every newline, and also for the 1st line if it's a 'start'
            if self._manual_add_dummy_prefix and (i > 0 or (i == 0 and is_start)):
                line = " " + line
            toks.extend(self._encode(line))

        return self._encode_post_process(toks)

    def _decode_wrapper(self, token_ids: List[int], **kwargs) -> str:
        res_text, _ = self._decode_with_offsets(token_ids, **kwargs)
        return res_text

    def _decode_with_offsets(self, token_ids: List[int], **kwargs) -> Tuple[str, List[Tuple[int, int]]]:
        start_of_line = kwargs.get("start_of_line", True)

        res_text = ""
        offsets = []
        tokens = self._convert_ids_to_tokens(token_ids)

        for token in tokens:
            if token not in self._no_show_tokens:
                text = token.replace("▁", " ")
            else:
                text = ""

            if start_of_line and text.startswith(" "):
                text = text[1:]

            if token == self._newline_piece:
                text = "\n"

            offsets.append((len(res_text), len(res_text) + len(text)))
            res_text += text

            start_of_line = token == self._newline_piece

        return res_text, offsets

    def _convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._token_to_id(tokens)

        return [self._token_to_id(token) for token in tokens]

    def _convert_ids_to_tokens_wrapper(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        if isinstance(token_ids, int):
            return self._id_to_token(token_ids)

        return [self._id_to_token(token_id) for token_id in token_ids]
