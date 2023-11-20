from tokenizer.jurassic_tokenizer import JurassicTokenizer
from tokenizer.tokenizer import Tokenizer


class PreTrainedTokenizers:
    J2_TOKENIZER = "j2-tokenizer"


_PRETRAINED_MODEL_NAMES = [
    PreTrainedTokenizers.J2_TOKENIZER,
]


class TokenizerFactory:
    @classmethod
    def get_tokenizer(cls, tokenizer_name: str = "j2-tokenizer") -> Tokenizer:
        if tokenizer_name not in _PRETRAINED_MODEL_NAMES:
            raise ValueError(f"Unknown tokenizer - {tokenizer_name}. Must be one of {_PRETRAINED_MODEL_NAMES}")

        if tokenizer_name == PreTrainedTokenizers.J2_TOKENIZER:
            return JurassicTokenizer.from_pretrained(tokenizer_name)
