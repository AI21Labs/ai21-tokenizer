from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any, Tuple, BinaryIO

import sentencepiece as spm

from ai21_tokenizer.base_tokenizer import BaseTokenizer, AsyncBaseTokenizer
from ai21_tokenizer.base_jurassic_tokenizer import BaseJurassicTokenizer
from ai21_tokenizer.file_utils import PathLike, load_binary, aload_binary, aread_file_handle

_MODEL_EXTENSION = ".model"
_MODEL_CONFIG_FILENAME = "config.json"


@dataclass
class SpaceSymbol:
    tok_id: int
    count: int


class JurassicTokenizer(BaseJurassicTokenizer, BaseTokenizer):
    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        config: Optional[Dict[str, Any]] = None,
        model_file_handle: Optional[BinaryIO] = None,
    ):
        BaseJurassicTokenizer.__init__(self, model_path=model_path, config=config, model_file_handle=model_file_handle)
        model_proto = load_binary(self._get_model_file(model_path)) if model_path else model_file_handle.read()
        # noinspection PyArgumentList
        self._sp = spm.SentencePieceProcessor(model_proto=model_proto)
        self._id_to_token_map = {i: self._sp.id_to_piece(i) for i in range(self.vocab_size)}
        self._token_to_id_map = {self._sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self._no_show_tokens = set(
            self._convert_ids_to_tokens([i for i in range(self.vocab_size) if self._sp.IsControl(i)])
        )
        self.newline_id = self._token_to_id(self._newline_piece)
        self._space_tokens = self._map_space_tokens()

    @property
    def vocab_size(self) -> int:
        return self._sp.vocab_size()

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Tokenizes the input text and returns it's token ids
        """
        return self._encode_wrapper(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        Transforms token ids into text
        """
        return self._decode_wrapper(token_ids, **kwargs)

    def decode_with_offsets(self, token_ids: List[int], **kwargs) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Transforms token ids into text, and returns the offsets of each token as well
        """
        return self._decode_with_offsets(token_ids, **kwargs)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        return self._convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        return self._convert_ids_to_tokens_wrapper(token_ids, **kwargs)

    @classmethod
    def from_file_handle(
        cls, model_file_handle: BinaryIO, config: Optional[Dict[str, Any]] = None
    ) -> JurassicTokenizer:
        return cls(model_file_handle=model_file_handle, config=config)

    @classmethod
    def from_file_path(cls, model_path: PathLike, config: Optional[Dict[str, Any]] = None) -> JurassicTokenizer:
        return cls(model_path=model_path, config=config)


class AsyncJurassicTokenizer(BaseJurassicTokenizer, AsyncBaseTokenizer):
    def __init__(self):
        raise ValueError(
            "Do not create AsyncJurassicTokenizer directly.Use either AsyncJurassicTokenizer.create or "
            "Tokenizer.get_async_tokenizer"
        )

    @classmethod
    async def create(
        cls,
        model_path: Optional[PathLike] = None,
        config: Optional[Dict[str, Any]] = None,
        model_file_handle: Optional[BinaryIO] = None,
        model_proto: Optional[bytes] = None,
    ):
        self = cls.__new__(cls)
        BaseJurassicTokenizer.__init__(
            self,
            model_path=model_path,
            config=config,
            model_file_handle=model_file_handle,
        )
        if not model_proto:
            await self._aload_model_proto()
        else:
            self._set_model_proto_related_variables(model_proto)

        return self

    @property
    def vocab_size(self) -> int:
        if not self._sp:
            raise ValueError(
                "Tokenizer not properly initialized. Please do not initialize the tokenizer directly. Use "
                "Tokenizer.get_async_tokenizer instead."
            )
        return self._sp.vocab_size()

    async def encode(self, text: str, **kwargs) -> List[int]:
        """
        Tokenizes the input text and returns it's token ids
        """
        if not self._sp:
            await self._aload_model_proto()

        return await self._make_async_call(callback_func=self._encode_wrapper, text=text, **kwargs)

    async def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        Transforms token ids into text
        """
        if not self._sp:
            await self._aload_model_proto()

        return await self._make_async_call(callback_func=self._decode_wrapper, token_ids=token_ids, **kwargs)

    async def decode_with_offsets(self, token_ids: List[int], **kwargs) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Transforms token ids into text, and returns the offsets of each token as well
        """
        if not self._sp:
            await self._aload_model_proto()

        return await self._make_async_call(callback_func=self._decode_with_offsets, token_ids=token_ids, **kwargs)

    async def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if not self._sp:
            await self._aload_model_proto()

        return await self._make_async_call(callback_func=self._convert_tokens_to_ids, tokens=tokens)

    async def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        if not self._sp:
            await self._aload_model_proto()

        return await self._make_async_call(
            callback_func=self._convert_ids_to_tokens_wrapper, token_ids=token_ids, **kwargs
        )

    @classmethod
    async def from_file_handle(
        cls, model_file_handle: BinaryIO, config: Optional[Dict[str, Any]] = None
    ) -> AsyncJurassicTokenizer:
        model_proto = await aread_file_handle(model_file_handle)
        return await cls.create(model_file_handle=model_file_handle, config=config, model_proto=model_proto)

    @classmethod
    async def from_file_path(
        cls, model_path: PathLike, config: Optional[Dict[str, Any]] = None
    ) -> AsyncJurassicTokenizer:
        model_proto = await aload_binary(model_path)
        return await cls.create(model_path=model_path, config=config, model_proto=model_proto)

    def _load_model_proto(self):
        model_proto = (
            load_binary(self._get_model_file(self._model_path)) if self._model_path else self._model_file_handle.read()
        )

        self._set_model_proto_related_variables(model_proto)

    async def _aload_model_proto(self):
        model_proto = (
            await aload_binary(self._get_model_file(self._model_path))
            if self._model_path
            else await aread_file_handle(self._model_file_handle)
        )

        self._set_model_proto_related_variables(model_proto)

    def _set_model_proto_related_variables(self, model_proto: bytes):
        # noinspection PyArgumentList
        self._sp = spm.SentencePieceProcessor(model_proto=model_proto)
        self._id_to_token_map = {i: self._sp.id_to_piece(i) for i in range(self.vocab_size)}
        self._token_to_id_map = {self._sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self._no_show_tokens = set(
            self._convert_ids_to_tokens([i for i in range(self.vocab_size) if self._sp.IsControl(i)])
        )
        self.newline_id = self._token_to_id(self._newline_piece)
        self._space_tokens = self._map_space_tokens()
