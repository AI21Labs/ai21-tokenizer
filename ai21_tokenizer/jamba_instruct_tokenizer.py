from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union, List, cast, Optional

from tokenizers import Tokenizer

from ai21_tokenizer import BaseTokenizer
from ai21_tokenizer.utils import PathLike

_TOKENIZER_FILE = "tokenizer.json"
_DEFAULT_MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "jamba_instruct"


class JambaInstructTokenizer(BaseTokenizer):
    _tokenizer: Tokenizer

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

    def _is_cached(self, cache_dir: PathLike) -> bool:
        return Path(cache_dir).exists() and _TOKENIZER_FILE in os.listdir(cache_dir)

    def _load_from_cache(self, cache_file: Path) -> Tokenizer:
        return cast(Tokenizer, Tokenizer.from_file(str(cache_file)))

    def _cache_tokenizer(self, tokenizer: Tokenizer, cache_dir: PathLike) -> None:
        # create cache directory for caching the tokenizer and save it
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(cache_dir / _TOKENIZER_FILE))

    def encode(self, text: str, **kwargs) -> List[int]:
        return self._tokenizer.encode(text, **kwargs).ids

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self._tokenizer.decode(token_ids, **kwargs)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._tokenizer.token_to_id(tokens)

        return [self._tokenizer.token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        if isinstance(token_ids, int):
            return self._tokenizer.id_to_token(token_ids)

        return [self._tokenizer.id_to_token(token_id) for token_id in token_ids]

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
