from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm

VOCAB_PATH = Path(__file__).parent / "resources"
VOCAB_NAME = "j2-dictionary"

_NUMBER_DIGITS = set("0123456789")


def is_number(s):
    return s and all([c in _NUMBER_DIGITS for c in s])


def with_extension(path: Path, suffix: str) -> Path:
    return path.parent / (path.name + suffix)


def load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r") as f:
        return json.load(f)


def load_binary(file_path: Path) -> bytes:
    with file_path.open("rb") as f:
        return f.read()


@dataclass
class SpaceSymbol:
    tok_id: int
    count: int


class JurassicTokenizer:
    def __init__(
        self,
        vocab_path: Path | None = VOCAB_PATH,
        vocab_name: str | None = VOCAB_NAME,
        config: dict[str, Any] | None = None,
    ):
        # assert (not (vocab_path.endswith(".model") or vocab_path.endswith(".args"))), \
        #     "Expected path to vocab without .args/.model extension"
        self._vocab_path = vocab_path
        self._vocab_name = vocab_name

        config = config if config is not None else self._load_config()
        self.unk_id = config.get("unk_id", None)
        self.eop_id = config.get("eop_id", None)

        self.newline_piece = config.get("newline_piece")
        self.mask_pieces = config.get("mask_pieces", [])

        # noinspection PyArgumentList
        self._sp = spm.SentencePieceProcessor(model_proto=self._load_model())

        self._manual_add_dummy_prefix = not (config.get("add_dummy_prefix", True))

        self._id_to_token = {i: self._sp.id_to_piece(i) for i in range(self.vocab_size)}
        self._token_to_id = {self._sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self.no_show_tokens = set(
            self.convert_ids_to_tokens([i for i in range(self.vocab_size) if self._sp.IsControl(i)])
        )

        self.newline_id = self.convert_tokens_to_ids(self.newline_piece)

        self._sample_split = re.compile(r"▁*[^▁]+|▁")
        self._space_split = re.compile("( {2,})")  # Split by 2 or more consecutive spaces

        self.number_mode = config.get("number_mode", None)
        self.space_mode = config.get("space_mode", None)
        self._space_tokens = self._map_space_tokens()

    def _map_space_tokens(self):
        res = []
        for count in range(32, 0, -1):
            tok_id = self.convert_tokens_to_ids("▁" * count)
            if tok_id != self.unk_id:
                res.append(SpaceSymbol(tok_id=tok_id, count=count))
        return res

    def _load_config(self) -> dict[str, Any]:
        return load_json(with_extension(path=self._vocab_path / self._vocab_name, suffix=".args"))

    def _load_model(self) -> bytes:
        return load_binary(with_extension(path=self._vocab_path / self._vocab_name, suffix=".model"))

    @property
    def vocab_size(self):
        return self._sp.vocab_size()

    def _encode(self, text):
        if self.space_mode is None:
            return self._sp.encode(text)
        assert self.space_mode == "left"
        assert self._manual_add_dummy_prefix, "Space Encoding is only supported with manual dummy prefix"
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
                    assert space_index < len(self._space_tokens)
                remaining -= self._space_tokens[space_index].count
                res.append(self._space_tokens[space_index].tok_id)
        if remainder:
            res.extend(self._sp.encode(remainder))
        return res

    def encode(self, text, is_start=True):
        if not self._manual_add_dummy_prefix and not is_start:
            # If it's not a 'start', but SPM is configured to auto-add a prefix space, we try to remove
            # the leading prefix if exists. This is not guaranteed to succeed, for example if the next sentence
            # doesn't start with a space but was sentence-split, we can't avoid SPM adding the leading space
            if text.startswith(" ") or text.startswith("\t"):
                text = text[1:]
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

    def _tokenize_number(self, num, mode):
        assert is_number(num)
        if mode.endswith("_keep"):
            # If the full number is in the vocab in keep mode, just use it
            single_id = self.convert_tokens_to_ids(num)
            if single_id != self.unk_id:
                return [single_id]
            mode = mode.rstrip("_keep")
        if mode in ["left", "right"]:
            res = []
            if mode == "right":
                offset = len(num) % 3
                if offset:
                    res.append(self.convert_tokens_to_ids(num[:offset]))
                    num = num[offset:]
            res += [self.convert_tokens_to_ids(num[i : i + 3]) for i in range(0, len(num), 3)]
        else:
            raise ValueError(f"Invalid number mode: {mode}")
        assert all([_id != self.unk_id for _id in res])
        return res

    def _encode_post_process(self, ids):
        if self.number_mode is None:
            return ids
        i = 0
        res = []
        while i < len(ids):
            token = self.convert_ids_to_tokens(ids[i])
            if not is_number(token):
                res.append(ids[i])
                i += 1
            else:
                num = ""
                while i < len(ids):
                    token = self.convert_ids_to_tokens(ids[i])
                    if not is_number(token):
                        break
                    num += token
                    i += 1
                res.extend(self._tokenize_number(num, self.number_mode))

        return res

    def decode(self, ids, start_of_line=True, return_offsets=False):
        # This decode function doesn't work exactly like SentencePiece's decode, as it removes leading whitespaces at
        # the beginning of lines. This is the behaviour we want, so that's what we implement here. This logic is what
        # was implemented in the make_text function in jurassic-serving.
        # It can also return the indices of tokens in resulting text
        res_text = ""
        offsets = []
        tokens = self.convert_ids_to_tokens(ids)
        for token in tokens:
            if token not in self.no_show_tokens:
                text = token.replace("▁", " ")
            else:
                text = ""
            if start_of_line and text.startswith(" "):
                text = text[1:]
            if token == self.newline_piece:
                text = "\n"

            offsets.append((len(res_text), len(res_text) + len(text)))
            res_text += text

            start_of_line = token == self.newline_piece

        if return_offsets:
            return res_text, offsets
        return res_text

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self._token_to_id.get(x, self.unk_id) for x in tokens]
        else:
            assert isinstance(tokens, str)
            return self._token_to_id.get(tokens, self.unk_id)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self._id_to_token[x] for x in ids]
        else:
            assert isinstance(ids, int)
            return self._id_to_token[ids]

    def get_mask_token_id(self, mask_index=0):
        assert len(self.mask_pieces) > mask_index, f"This vocab has no mask-piece for mask_index={mask_index}"
        return self.mask_pieces[mask_index]

    def encode_prompt_completion(self, prompt, completion):
        prompt_tokens = self.encode(prompt, is_start=True)
        # Add leading space only when prompt and completion are not in the same sentence
        if prompt.endswith("\n"):
            completion_tokens = self.encode(completion, is_start=True)
        else:
            completion_tokens = self.encode(completion, is_start=False)
        return prompt_tokens, completion_tokens

    def get_whitespace_token_ids(self):
        return [x.tok_id for x in self._space_tokens] + [self.newline_id]

    def get_numeric_token_ids(self):
        return [k for k, v in self._id_to_token.items() if v.isnumeric()]

    @classmethod
    def create(cls) -> JurassicTokenizer:
        return JurassicTokenizer(vocab_path=VOCAB_PATH, vocab_name=VOCAB_NAME)
