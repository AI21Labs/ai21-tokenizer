from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Union, List, Optional, cast

from tokenizers import Tokenizer

from ai21_tokenizer import BaseTokenizer, AsyncBaseTokenizer
from ai21_tokenizer.file_utils import PathLike
from ai21_tokenizer.base_jamba_instruct_tokenizer import BaseJambaInstructTokenizer

_TOKENIZER_FILE = "tokenizer.json"
_DEFAULT_MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "jamba_instruct"


class JambaInstructTokenizer(BaseJambaInstructTokenizer, BaseTokenizer):
    def __init__(
        self,
        model_path: str,
        cache_dir: Optional[PathLike] = None,
    ):
        """
        Args:
            model_path: str
                The identifier of a Model on the Hugging Face Hub, that contains a tokenizer.json file
            cache_dir: Optional[PathLike]
                The directory to cache the tokenizer.json file.
                 If not provided, the default cache directory will be used
        """
        self._tokenizer = self._init_tokenizer(model_path=model_path, cache_dir=cache_dir or _DEFAULT_MODEL_CACHE_DIR)

    def _init_tokenizer(self, model_path: PathLike, cache_dir: PathLike) -> Tokenizer:
        if self._is_cached(cache_dir):
            return self._load_from_cache(cache_dir / _TOKENIZER_FILE)

        tokenizer = cast(
            Tokenizer,
            Tokenizer.from_pretrained(model_path),
        )
        self._cache_tokenizer(tokenizer, cache_dir)

        return tokenizer

    def _load_from_cache(self, cache_file: Path) -> Tokenizer:
        return cast(Tokenizer, Tokenizer.from_file(str(cache_file)))

    def encode(self, text: str, **kwargs) -> List[int]:
        return self._encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self._decode(token_ids, **kwargs)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        return self._convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        return self._convert_ids_to_tokens(token_ids)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()


class AsyncJambaInstructTokenizer(BaseJambaInstructTokenizer, AsyncBaseTokenizer):
    _model_path: str
    _tokenizer: Tokenizer = None
    _cache_dir: PathLike = None

    def __init__(self):
        raise ValueError(
            "Do not create AsyncJambaInstructTokenizer directly. Use either AsyncJambaInstructTokenizer.create or "
            "Tokenizer.get_async_tokenizer"
        )

    @classmethod
    async def create(
        cls,
        model_path: str,
        cache_dir: Optional[PathLike] = None,
    ):
        """
        Args:
            model_path: str
                The identifier of a Model on the Hugging Face Hub, that contains a tokenizer.json file
            cache_dir: Optional[PathLike]
                The directory to cache the tokenizer.json file.
                 If not provided, the default cache directory will be used
        """
        self = cls.__new__(cls)
        self._model_path = model_path
        self._cache_dir = cache_dir or _DEFAULT_MODEL_CACHE_DIR
        await self._init_tokenizer()
        return self

    async def encode(self, text: str, **kwargs) -> List[int]:
        if not self._tokenizer:
            await self._init_tokenizer()

        return await self._make_async_call(callback_func=self._encode, text=text, **kwargs)

    async def decode(self, token_ids: List[int], **kwargs) -> str:
        if not self._tokenizer:
            await self._init_tokenizer()

        return await self._make_async_call(callback_func=self._decode, token_ids=token_ids, **kwargs)

    async def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if not self._tokenizer:
            await self._init_tokenizer()

        return await self._make_async_call(callback_func=self._convert_tokens_to_ids, tokens=tokens)

    async def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        if not self._tokenizer:
            await self._init_tokenizer()

        return await self._make_async_call(callback_func=self._convert_ids_to_tokens, token_ids=token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        if not self._tokenizer:
            raise ValueError(
                "Tokenizer not properly initialized. Please do not initialize the tokenizer directly. Use "
                "Tokenizer.get_async_tokenizer instead."
            )
        return self._tokenizer.get_vocab_size()

    async def _init_tokenizer(self):
        if self._is_cached(self._cache_dir):
            self._tokenizer = await self._load_from_cache(self._cache_dir / _TOKENIZER_FILE)
        else:
            tokenizer_from_pretrained = await self._make_async_call(
                callback_func=Tokenizer.from_pretrained, identifier=self._model_path
            )

            tokenizer = cast(
                Tokenizer,
                tokenizer_from_pretrained,
            )
            self._cache_tokenizer(tokenizer, self._cache_dir)

            self._tokenizer = tokenizer

    async def _load_from_cache(self, cache_file: Path) -> Tokenizer:
        tokenizer_from_file = await self._make_async_call(callback_func=Tokenizer.from_file, path=str(cache_file))
        return cast(Tokenizer, tokenizer_from_file)
