from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict

import sentencepiece as spm

from jurassic_tokenization.utils import load_json, with_extension, load_binary, is_number

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"
_PRETRAINED_TOKENIZERS = [
    "j2-tokenizer",
]


@dataclass
class SpaceSymbol:
    tok_id: int
    count: int


class JurassicTokenizer:
    def __init__(
        self,
        tokenizer_path: Path | str | None,
        tokenizer_name: str | None,
        config: Dict[str, Any] | None = None,
    ):
        self._vocab_path = Path(tokenizer_path)
        self._vocab_name = tokenizer_name

        config = config if config is not None else self._load_config()
        self.unk_id = config.get("unk_id")
        self.eop_id = config.get("eop_id")

        self._newline_piece = config.get("newline_piece")
        self._mask_pieces = config.get("mask_pieces", [])

        # noinspection PyArgumentList
        self._sp = spm.SentencePieceProcessor(model_proto=self._load_model())

        self._manual_add_dummy_prefix = not (config.get("add_dummy_prefix", True))

        self._id_to_token = {i: self._sp.id_to_piece(i) for i in range(self._vocab_size)}
        self._token_to_id = {self._sp.id_to_piece(i): i for i in range(self._vocab_size)}
        self._no_show_tokens = set(
            self._convert_ids_to_tokens([i for i in range(self._vocab_size) if self._sp.IsControl(i)])
        )

        self._newline_id = self._convert_tokens_to_ids(self._newline_piece)

        self._sample_split = re.compile(r"▁*[^▁]+|▁")
        self._space_split = re.compile("( {2,})")  # Split by 2 or more consecutive spaces

        self._number_mode = config.get("number_mode")
        self._space_mode = config.get("space_mode")
        self._space_tokens = self._map_space_tokens()

    def _map_space_tokens(self) -> List[SpaceSymbol]:
        res = []
        for count in range(32, 0, -1):
            tok_id = self._convert_tokens_to_ids("▁" * count)
            if tok_id != self.unk_id:
                res.append(SpaceSymbol(tok_id=tok_id, count=count))

        return res

    def _load_config(self) -> Dict[str, Any]:
        return load_json(with_extension(path=self._vocab_path / self._vocab_name, suffix=".args"))

    def _load_model(self) -> bytes:
        return load_binary(with_extension(path=self._vocab_path / self._vocab_name, suffix=".model"))

    @property
    def _vocab_size(self) -> int:
        return self._sp.vocab_size()

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
            single_id = self._convert_tokens_to_ids(num)

            if single_id != self.unk_id:
                return [single_id]

            mode = mode.rstrip("_keep")

        if mode in ["left", "right"]:
            res = []

            if mode == "right":
                offset = len(num) % 3

                if offset:
                    res.append(self._convert_tokens_to_ids(num[:offset]))
                    num = num[offset:]

            res += [self._convert_tokens_to_ids(num[i : i + 3]) for i in range(0, len(num), 3)]
        else:
            raise ValueError(f"Invalid number mode: {mode}")

        return res

    def _encode_post_process(self, ids: List[int]) -> List[int]:
        if self._number_mode is None:
            return ids

        i = 0
        res = []

        while i < len(ids):
            token = self._convert_ids_to_tokens(ids[i])
            if not is_number(token):
                res.append(ids[i])
                i += 1
            else:
                num = ""

                while i < len(ids):
                    token = self._convert_ids_to_tokens(ids[i])
                    if not is_number(token):
                        break
                    num += token
                    i += 1

                res.extend(self._tokenize_number(num, self._number_mode))

        return res

    def _convert_tokens_to_ids(self, tokens: List[str] | str) -> List[int] | int:
        if isinstance(tokens, list):
            return [self._token_to_id.get(x, self.unk_id) for x in tokens]
        else:
            return self._token_to_id.get(tokens, self.unk_id)

    def _convert_ids_to_tokens(self, ids: List[int] | int) -> List[int] | int:
        if isinstance(ids, list):
            return [self._id_to_token[x] for x in ids]
        else:
            return self._id_to_token[ids]

    def encode(self, text: str) -> List[int]:
        lines = text.split("\n")
        toks = []

        for i, line in enumerate(lines):
            if i > 0:
                toks.append(self._newline_id)
            if not line:
                continue
            # We add the dummy prefix on every newline, and also for the 1st line if it's a 'start'
            if self._manual_add_dummy_prefix and (i > 0 or i == 0):
                line = " " + line
            toks.extend(self._encode(line))

        return self._encode_post_process(toks)

    def decode(self, ids: List[int]) -> str:
        start_of_line = True

        res_text = ""
        offsets = []
        tokens = self._convert_ids_to_tokens(ids)

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

        return res_text

    @classmethod
    def create(cls, tokenizer_name: str = "j2-tokenizer") -> JurassicTokenizer:
        if tokenizer_name not in _PRETRAINED_TOKENIZERS:
            raise ValueError(f"Unknown tokenizer - {tokenizer_name}. Must be one of {cls.pretrained_tokenizers()}")

        return JurassicTokenizer(tokenizer_path=_LOCAL_RESOURCES_PATH, tokenizer_name=tokenizer_name)

    @classmethod
    def pretrained_tokenizers(cls) -> List[str]:
        return _PRETRAINED_TOKENIZERS
