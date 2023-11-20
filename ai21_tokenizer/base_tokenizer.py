from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union


class BaseTokenizer(ABC):
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

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass
