from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Any, Dict

from tokenizer.utils import load_json

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"

MODEL_EXTENSION = ".model"
MODEL_CONFIG_FILENAME = "config.json"


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], **kwargs) -> str:
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, tokenizer_name: str) -> Tokenizer:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @classmethod
    def _tokenizer_dir(cls, tokenizer_name: str) -> Path:
        return _LOCAL_RESOURCES_PATH / tokenizer_name

    @classmethod
    def _model_path(cls, tokenizer_name: str) -> Path:
        return cls._tokenizer_dir(tokenizer_name) / f"{tokenizer_name}.model"

    @classmethod
    def _config(cls, tokenizer_name: str) -> Dict[str, Any]:
        config_path = cls._tokenizer_dir(tokenizer_name) / MODEL_CONFIG_FILENAME
        return load_json(config_path)
