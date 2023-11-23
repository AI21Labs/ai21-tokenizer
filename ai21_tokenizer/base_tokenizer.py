from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union


class BaseTokenizer(ABC):
    """
    Base class for tokenizers.

    This class defines the interface for tokenization operations such as encoding, decoding,
    converting tokens to IDs, and converting IDs to tokens.
    """

    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Encodes the given text into a list of token IDs.

        Args:
            text (str): The input text to be encoded.
            **kwargs: Additional keyword arguments for encoding.

        Returns:
            List[int]: The list of token IDs representing the encoded text.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        Decodes the given list of token IDs into a string.

        Args:
            token_ids (List[int]): The list of token IDs to be decoded.
            **kwargs: Additional keyword arguments for decoding.

        Returns:
            str: The decoded string.
        """
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts the given tokens into token IDs.

        Args:
            tokens (Union[str, List[str]]): The input tokens to be converted.

        Returns:
            Union[int, List[int]]: The token IDs representing the input tokens.
        """
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        """
        Converts the given token IDs into tokens.

        Args:
            token_ids (Union[int, List[int]]): The input token IDs to be converted.
            **kwargs: Additional keyword arguments for conversion.

        Returns:
            Union[str, List[str]]: The tokens representing the input token IDs.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabs.

        Returns:
            int: The size of the vocabs.
        """
        pass
