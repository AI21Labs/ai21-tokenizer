import os
from pathlib import Path
from typing import Dict, Any

from ai21_tokenizer.base_tokenizer import BaseTokenizer
from ai21_tokenizer.jamaba_tokenizer import JambaTokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer
from ai21_tokenizer.utils import load_json

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"
_MODEL_EXTENSION = ".model"
_MODEL_CONFIG_FILENAME = "config.json"
_MODEL_CACHE_DIR = _LOCAL_RESOURCES_PATH / "cache"


class PreTrainedTokenizers:
    J2_TOKENIZER = "j2-tokenizer"
    JAMBA_TOKENIZER = "jamba-tokenizer"


_PRETRAINED_MODEL_NAMES = [
    PreTrainedTokenizers.J2_TOKENIZER,
    PreTrainedTokenizers.JAMBA_TOKENIZER,
]


class TokenizerFactory:
    """
    Factory class to create AI21 tokenizer
    Currently supports only J2-Tokenizer
    """

    @classmethod
    def get_tokenizer(cls, tokenizer_name: str = PreTrainedTokenizers.J2_TOKENIZER) -> BaseTokenizer:
        model_path = cls._model_path(tokenizer_name)

        if tokenizer_name == PreTrainedTokenizers.JAMBA_TOKENIZER:
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
            return JambaTokenizer(model_path=model_path, cache_dir=_MODEL_CACHE_DIR)

        config = cls._get_config(tokenizer_name)
        return JurassicTokenizer(model_path=model_path, config=config)

    @classmethod
    def _tokenizer_dir(cls, tokenizer_name: str) -> Path:
        return _LOCAL_RESOURCES_PATH / tokenizer_name

    @classmethod
    def _model_path(cls, tokenizer_name: str) -> Path:
        return cls._tokenizer_dir(tokenizer_name) / f"{tokenizer_name}{_MODEL_EXTENSION}"

    @classmethod
    def _get_config(cls, tokenizer_name: str) -> Dict[str, Any]:
        config_path = cls._tokenizer_dir(tokenizer_name) / _MODEL_CONFIG_FILENAME
        return load_json(config_path)
