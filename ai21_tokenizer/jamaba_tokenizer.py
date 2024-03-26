from typing import Union, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ai21_tokenizer import BaseTokenizer
from ai21_tokenizer.utils import PathLike


class JambaTokenizer(BaseTokenizer):
    _tokenizer: PreTrainedTokenizerFast

    def __init__(
        self,
        model_path: Optional[PathLike] = None,
        cache_dir: Optional[PathLike] = None,
    ):
        self._name_or_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, cache_dir=cache_dir)

    def encode(self, text: str, **kwargs) -> List[int]:
        return self._tokenizer.encode(text=text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self._tokenizer.decode(token_ids=token_ids, **kwargs)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], **kwargs) -> Union[str, List[str]]:
        return self._tokenizer.convert_ids_to_tokens(ids=token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
