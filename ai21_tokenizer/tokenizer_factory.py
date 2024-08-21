import os
import tempfile
from pathlib import Path

from ai21_tokenizer.base_tokenizer import BaseTokenizer, AsyncBaseTokenizer
from ai21_tokenizer.jamba_instruct_tokenizer import JambaInstructTokenizer, AsyncJambaInstructTokenizer
from ai21_tokenizer.jamba_1_5_tokenizer import Jamba1_5Tokenizer, AsyncJamba1_5Tokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer, AsyncJurassicTokenizer

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"
_ENV_CACHE_DIR_KEY = "AI21_TOKENIZER_CACHE_DIR"
JAMBA_TOKENIZER_HF_PATH = "ai21labs/Jamba-v0.1"
JAMBA_1_5_MINI_TOKENIZER_HF_PATH = "ai21labs/AI21-Jamba-1.5-Mini"
JAMBA_1_5_LARGE_TOKENIZER_HF_PATH = "ai21labs/AI21-Jamba-1.5-Large"


def _get_cache_dir(tokenizer_name: str) -> Path:
    tokenizer_name_as_path = tokenizer_name.replace(".", "_")
    tokenizer_name_as_path = tokenizer_name_as_path.replace("-", "_")
    default_tokenizer_cache_dir = Path(tempfile.gettempdir()) / tokenizer_name_as_path
    env_cache_from_env = os.getenv(_ENV_CACHE_DIR_KEY)

    if env_cache_from_env is not None:
        return Path(env_cache_from_env)

    return default_tokenizer_cache_dir


class PreTrainedTokenizers:
    J2_TOKENIZER = "j2-tokenizer"
    JAMBA_INSTRUCT_TOKENIZER = "jamba-instruct-tokenizer"
    JAMBA_TOKENIZER = "jamba-tokenizer"
    JAMBA_1_5_MINI_TOKENIZER = "jamba-1.5-mini-tokenizer"
    JAMBA_1_5_LARGE_TOKENIZER = "jamba-1.5-large-tokenizer"


class TokenizerFactory:
    """
    Factory class to create AI21 tokenizer
    Currently supports only J2-Tokenizer
    """

    @classmethod
    def get_tokenizer(
        cls,
        tokenizer_name: str = PreTrainedTokenizers.J2_TOKENIZER,
    ) -> BaseTokenizer:
        cache_dir = _get_cache_dir(tokenizer_name=tokenizer_name)

        if tokenizer_name == PreTrainedTokenizers.JAMBA_1_5_MINI_TOKENIZER:
            return Jamba1_5Tokenizer(model_path=JAMBA_1_5_MINI_TOKENIZER_HF_PATH, cache_dir=cache_dir)

        if tokenizer_name == PreTrainedTokenizers.JAMBA_1_5_LARGE_TOKENIZER:
            return Jamba1_5Tokenizer(model_path=JAMBA_1_5_LARGE_TOKENIZER_HF_PATH, cache_dir=cache_dir)

        if (
            tokenizer_name == PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER
            or tokenizer_name == PreTrainedTokenizers.JAMBA_TOKENIZER
        ):
            return JambaInstructTokenizer(model_path=JAMBA_TOKENIZER_HF_PATH, cache_dir=os.getenv(_ENV_CACHE_DIR_KEY))

        if tokenizer_name == PreTrainedTokenizers.J2_TOKENIZER:
            return JurassicTokenizer(_LOCAL_RESOURCES_PATH / PreTrainedTokenizers.J2_TOKENIZER)

        raise ValueError(f"Tokenizer {tokenizer_name} is not supported")

    @classmethod
    async def get_async_tokenizer(
        cls,
        tokenizer_name: str = PreTrainedTokenizers.J2_TOKENIZER,
    ) -> AsyncBaseTokenizer:
        cache_dir = _get_cache_dir(tokenizer_name=tokenizer_name)

        if tokenizer_name == PreTrainedTokenizers.JAMBA_1_5_MINI_TOKENIZER:
            return await AsyncJamba1_5Tokenizer.create(model_path=JAMBA_1_5_MINI_TOKENIZER_HF_PATH, cache_dir=cache_dir)

        if tokenizer_name == PreTrainedTokenizers.JAMBA_1_5_LARGE_TOKENIZER:
            return await AsyncJamba1_5Tokenizer.create(
                model_path=JAMBA_1_5_LARGE_TOKENIZER_HF_PATH, cache_dir=cache_dir
            )

        if (
            tokenizer_name == PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER
            or tokenizer_name == PreTrainedTokenizers.JAMBA_TOKENIZER
        ):
            return await AsyncJambaInstructTokenizer.create(
                model_path=JAMBA_TOKENIZER_HF_PATH, cache_dir=os.getenv(_ENV_CACHE_DIR_KEY)
            )

        if tokenizer_name == PreTrainedTokenizers.J2_TOKENIZER:
            return await AsyncJurassicTokenizer.create(
                model_path=_LOCAL_RESOURCES_PATH / PreTrainedTokenizers.J2_TOKENIZER
            )

        raise ValueError(f"Tokenizer {tokenizer_name} is not supported")
