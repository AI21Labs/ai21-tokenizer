from enum import Enum

from tokenizer.tokenizer import Tokenizer
from tokenizer.jurassic_tokenizer import JurassicTokenizer


class PreTrainedTokenizers(str, Enum):
    J2Tokenizer = "j2-tokenizer"


class TokenizerFactory:
    @classmethod
    def get_tokenizer(cls, tokenizer_name: PreTrainedTokenizers = PreTrainedTokenizers.J2Tokenizer) -> Tokenizer:
        if tokenizer_name not in PreTrainedTokenizers:
            raise ValueError(f"Unknown tokenizer - {tokenizer_name}. Must be one of {PreTrainedTokenizers.__members__}")

        if tokenizer_name == PreTrainedTokenizers.J2Tokenizer:
            return JurassicTokenizer.from_pretrained(tokenizer_name.value)
