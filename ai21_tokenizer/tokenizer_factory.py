import os
from pathlib import Path

from ai21_tokenizer.base_tokenizer import BaseTokenizer
from ai21_tokenizer.jamaba_tokenizer import JambaTokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer
from ai21_tokenizer.utils import PathLike

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"
_MODEL_CACHE_DIR = _LOCAL_RESOURCES_PATH / "cache"


class PreTrainedTokenizers:
    J2_TOKENIZER = "j2-tokenizer"
    JAMBA_TOKENIZER = "jamba-tokenizer"


_TOKENIZER_MODEL_MAP = {
    PreTrainedTokenizers.JAMBA_TOKENIZER: "huggingface-tokenizer-url-placeholder",
    PreTrainedTokenizers.J2_TOKENIZER: _LOCAL_RESOURCES_PATH / PreTrainedTokenizers.J2_TOKENIZER,
}


class TokenizerFactory:
    """
    Factory class to create AI21 tokenizer
    Currently supports only J2-Tokenizer
    """

    @classmethod
    def get_tokenizer(cls, tokenizer_name: str = PreTrainedTokenizers.J2_TOKENIZER) -> BaseTokenizer:
        model_path = cls._model_path(tokenizer_name)

        if tokenizer_name == PreTrainedTokenizers.JAMBA_TOKENIZER:
            return cls._create_jamaba_tokenizer(model_path)

        return cls._create_jurassic_tokenizer(model_path)

    @classmethod
    def _create_jamaba_tokenizer(cls, model_path: str) -> JambaTokenizer:
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"  # Disable Huggingface advice warning

        return JambaTokenizer(model_path=model_path, cache_dir=_MODEL_CACHE_DIR)

    @classmethod
    def _create_jurassic_tokenizer(cls, model_path: str) -> JurassicTokenizer:
        return JurassicTokenizer(model_path=model_path)

    @classmethod
    def _model_path(cls, tokenizer_name: str) -> PathLike:
        return _TOKENIZER_MODEL_MAP[tokenizer_name]
