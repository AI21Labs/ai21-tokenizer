import os
from pathlib import Path

from ai21_tokenizer.base_tokenizer import BaseTokenizer
from ai21_tokenizer.jamba_instruct_tokenizer import JambaInstructTokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer
from ai21_tokenizer.utils import PathLike

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"
_MODEL_CACHE_DIR = _LOCAL_RESOURCES_PATH / "cache"


class PreTrainedTokenizers:
    J2_TOKENIZER = "j2-tokenizer"
    JAMBA_TOKENIZER = "jamba-tokenizer"


class TokenizerFactory:
    """
    Factory class to create AI21 tokenizer
    Currently supports only J2-Tokenizer
    """

    @classmethod
    def get_tokenizer(cls, tokenizer_name: str = PreTrainedTokenizers.J2_TOKENIZER) -> BaseTokenizer:
        if tokenizer_name == PreTrainedTokenizers.JAMBA_TOKENIZER:
            return cls._create_jamaba_tokenizer("<huggingface-tokenizer-url-placeholder>")

        if tokenizer_name == PreTrainedTokenizers.J2_TOKENIZER:
            return cls._create_jurassic_tokenizer(_LOCAL_RESOURCES_PATH / PreTrainedTokenizers.J2_TOKENIZER)

        raise ValueError(f"Tokenizer {tokenizer_name} is not supported")

    @classmethod
    def _create_jamaba_tokenizer(cls, model_path: str) -> JambaInstructTokenizer:
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"  # Disable Huggingface advice warning

        return JambaInstructTokenizer(model_path=model_path, cache_dir=_MODEL_CACHE_DIR)

    @classmethod
    def _create_jurassic_tokenizer(cls, model_path: PathLike) -> JurassicTokenizer:
        return JurassicTokenizer(model_path=model_path)
