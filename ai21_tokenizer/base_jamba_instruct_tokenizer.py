from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Union, Optional
from abc import ABC, abstractmethod

from tokenizers import Tokenizer

from ai21_tokenizer.file_utils import PathLike

_TOKENIZER_FILE = "tokenizer.json"
_DEFAULT_MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "jamba_instruct"


class BaseJambaInstructTokenizer(ABC):
    _tokenizer: Optional[Tokenizer] = None

    @abstractmethod
    def _load_from_cache(self, cache_file: Path) -> Tokenizer:
        pass

    def _is_cached(self, cache_dir: PathLike) -> bool:
        return Path(cache_dir).exists() and _TOKENIZER_FILE in os.listdir(cache_dir)

    def _cache_tokenizer(self, tokenizer: Tokenizer, cache_dir: PathLike) -> None:
        # create cache directory for caching the tokenizer and save it
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(cache_dir / _TOKENIZER_FILE))

    def _encode(self, text: str, **kwargs) -> List[int]:
        return self._tokenizer.encode(text, **kwargs).ids

    def _decode(self, token_ids: List[int], **kwargs) -> str:
        return self._tokenizer.decode(token_ids, **kwargs)

    def _convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._tokenizer.token_to_id(tokens)

        return [self._tokenizer.token_to_id(token) for token in tokens]

    def _convert_ids_to_tokens(self, token_ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(token_ids, int):
            return self._tokenizer.id_to_token(token_ids)

        return [self._tokenizer.id_to_token(token_id) for token_id in token_ids]
