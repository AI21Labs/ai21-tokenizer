from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any

import sentencepiece as spm

from ai21_tokenizer.base_tokenizer import BaseTokenizer
from ai21_tokenizer.utils import load_binary, is_number, PathLike


@dataclass
class SpaceSymbol:
    tok_id: int
    count: int


class JurassicTokenizer(BaseTokenizer):
    def __init__(
        self,
        model_path: PathLike,
        config: Optional[Dict[str, Any]] = None,
    ):
        # noinspection PyArgumentList
        self._sp = spm.SentencePieceProcessor(model_proto=load_binary(model_path))
        config = config or {}

        self.unk_id = config.get("unk_id")
        self.eop_id = config.get("eop_id")

        self._newline_piece = config.get("newline_piece")
        self._mask_pieces = config.get("mask_pieces", [])

        self._manual_add_dummy_prefix = not (config.get("add_dummy_prefix", True))

        self._id_to_token_map = {i: self._sp.id_to_piece(i) for i in range(self.vocab_size)}
        self._token_to_id_map = {self._sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self._no_show_tokens = set(
            self._convert_ids_to_tokens([i for i in range(self.vocab_size) if self._sp.IsControl(i)])
        )

        self._newline_id = self._token_to_id(self._newline_piece)

        self._sample_split = re.compile(r"▁*[^▁]+|▁")
        self._space_split = re.compile("( {2,})")  # Split by 2 or more consecutive spaces

        self._number_mode = config.get("number_mode")
        self._space_mode = config.get("space_mode")
        self._space_tokens = self._map_space_tokens()

    def _map_space_tokens(self) -> List[SpaceSymbol]:
        res = []
        for count in range(32, 0, -1):
            tok_id = self._token_to_id("▁" * count)
            if tok_id != self.unk_id:
                res.append(SpaceSymbol(tok_id=tok_id, count=count))

        return res

    @property
    def vocab_size(self) -> int:
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

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Tokenizes the input text and returns it's token ids
        """
        lines = text.split("\n")
        toks = []

        for i, line in enumerate(lines):
            if i > 0:
                toks.append(self._newline_id)
            if not line:
                continue
            # We add the dummy prefix on every newline, and also for the 1st line if it's a 'start'
            if self._manual_add_dummy_prefix and i >= 0:
                line = " " + line
            toks.extend(self._encode(line))

        return self._encode_post_process(toks)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        Transforms token ids into text
        """
        start_of_line = True

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

        return res_text

    def _id_to_token(self, token_id: int) -> str:
        return self._id_to_token_map[token_id]

    def _convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return [self._id_to_token(token_id) for token_id in token_ids]

    def _token_to_id(self, token: str) -> int:
        return self._token_to_id_map.get(token, self.unk_id)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._token_to_id(tokens)

        return [self._token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        if isinstance(token_ids, int):
            return self._id_to_token(token_ids)

        return [self._id_to_token(token_id) for token_id in token_ids]
