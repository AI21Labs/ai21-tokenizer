from pathlib import Path
from typing import Dict, Any

from ai21_tokenizer.base_tokenizer import BaseTokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer
from ai21_tokenizer.utils import load_json
from .version import VERSION

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"

MODEL_EXTENSION = ".model"
MODEL_CONFIG_FILENAME = "config.json"

__version__ = VERSION


class PreTrainedTokenizers:
    J2_TOKENIZER = "j2-tokenizer"


_PRETRAINED_MODEL_NAMES = [
    PreTrainedTokenizers.J2_TOKENIZER,
]


class Tokenizer:
    """
    Factory class to create AI21 tokenizer
    Currently supports only J2-Tokenizer
    """

    _tokenizer_name = PreTrainedTokenizers.J2_TOKENIZER

    @classmethod
    def get_tokenizer(cls) -> BaseTokenizer:
        config = cls._get_config(cls._tokenizer_name)
        model_path = cls._model_path(cls._tokenizer_name)
        return JurassicTokenizer(model_path=model_path, config=config)

    @classmethod
    def _tokenizer_dir(cls, tokenizer_name: str) -> Path:
        return _LOCAL_RESOURCES_PATH / tokenizer_name

    @classmethod
    def _model_path(cls, tokenizer_name: str) -> Path:
        return cls._tokenizer_dir(tokenizer_name) / f"{tokenizer_name}.model"

    @classmethod
    def _get_config(cls, tokenizer_name: str) -> Dict[str, Any]:
        config_path = cls._tokenizer_dir(tokenizer_name) / MODEL_CONFIG_FILENAME
        return load_json(config_path)
