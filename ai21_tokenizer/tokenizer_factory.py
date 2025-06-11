import os
import tempfile

from pathlib import Path

from ai21_tokenizer.base_tokenizer import AsyncBaseTokenizer, BaseTokenizer
from ai21_tokenizer.jamba_1_5_tokenizer import AsyncJambaTokenizer, SyncJambaTokenizer


_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"
_ENV_CACHE_DIR_KEY = "AI21_TOKENIZER_CACHE_DIR"
JAMBA_TOKENIZER_HF_PATH = "ai21labs/Jamba-v0.1"
JAMBA_MINI_1_6_TOKENIZER_HF_PATH = "ai21labs/AI21-Jamba-Mini-1.6"
JAMBA_LARGE_1_6_TOKENIZER_HF_PATH = "ai21labs/AI21-Jamba-Large-1.6"


def _get_cache_dir(tokenizer_name: str) -> Path:
    tokenizer_name_as_path = tokenizer_name.replace(".", "_")
    tokenizer_name_as_path = tokenizer_name_as_path.replace("-", "_")
    default_tokenizer_cache_dir = Path(tempfile.gettempdir()) / tokenizer_name_as_path
    env_cache_from_env = os.getenv(_ENV_CACHE_DIR_KEY)

    if env_cache_from_env is not None:
        return Path(env_cache_from_env)

    return default_tokenizer_cache_dir


class PreTrainedTokenizers:
    # deprecated tokenizers
    JAMBA_INSTRUCT_TOKENIZER = "jamba-instruct-tokenizer"
    JAMBA_TOKENIZER = "jamba-tokenizer"
    JAMBA_1_5_MINI_TOKENIZER = "jamba-1.5-mini-tokenizer"
    JAMBA_1_5_LARGE_TOKENIZER = "jamba-1.5-large-tokenizer"

    # active tokenizers
    JAMBA_MINI_1_6_TOKENIZER = "jamba-mini-1.6-tokenizer"
    JAMBA_LARGE_1_6_TOKENIZER = "jamba-large-1.6-tokenizer"
    JAMBA_MINI_TOKENIZER = "jamba-mini-tokenizer"
    JAMBA_LARGE_TOKENIZER = "jamba-large-tokenizer"


_TOKENIZER_NAME_TO_MODEL_PATH = {
    PreTrainedTokenizers.JAMBA_MINI_TOKENIZER: JAMBA_MINI_1_6_TOKENIZER_HF_PATH,
    PreTrainedTokenizers.JAMBA_LARGE_TOKENIZER: JAMBA_LARGE_1_6_TOKENIZER_HF_PATH,
    PreTrainedTokenizers.JAMBA_1_5_MINI_TOKENIZER: JAMBA_MINI_1_6_TOKENIZER_HF_PATH,
    PreTrainedTokenizers.JAMBA_1_5_LARGE_TOKENIZER: JAMBA_LARGE_1_6_TOKENIZER_HF_PATH,
    PreTrainedTokenizers.JAMBA_MINI_1_6_TOKENIZER: JAMBA_MINI_1_6_TOKENIZER_HF_PATH,
    PreTrainedTokenizers.JAMBA_LARGE_1_6_TOKENIZER: JAMBA_LARGE_1_6_TOKENIZER_HF_PATH,
    PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER: JAMBA_TOKENIZER_HF_PATH,
    PreTrainedTokenizers.JAMBA_TOKENIZER: JAMBA_TOKENIZER_HF_PATH,
}


class TokenizerFactory:
    """
    Factory class to create AI21 tokenizer
    Currently supports only J2-Tokenizer
    """

    @classmethod
    def get_tokenizer(
        cls,
        tokenizer_name: str = PreTrainedTokenizers.JAMBA_MINI_TOKENIZER,
    ) -> BaseTokenizer:
        cache_dir = _get_cache_dir(tokenizer_name=tokenizer_name)

        model_path = _TOKENIZER_NAME_TO_MODEL_PATH.get(tokenizer_name)

        if model_path is None:
            raise ValueError(f"Tokenizer {tokenizer_name} is not supported")

        return SyncJambaTokenizer(model_path=model_path, cache_dir=cache_dir)

    @classmethod
    async def get_async_tokenizer(
        cls,
        tokenizer_name: str = PreTrainedTokenizers.JAMBA_MINI_TOKENIZER,
    ) -> AsyncBaseTokenizer:
        cache_dir = _get_cache_dir(tokenizer_name=tokenizer_name)

        model_path = _TOKENIZER_NAME_TO_MODEL_PATH.get(tokenizer_name)

        if model_path is None:
            raise ValueError(f"Tokenizer {tokenizer_name} is not supported")

        return await AsyncJambaTokenizer.create(model_path=model_path, cache_dir=cache_dir)
