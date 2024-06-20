import os
from pathlib import Path

from ai21_tokenizer.base_tokenizer import BaseTokenizer, AsyncBaseTokenizer
from ai21_tokenizer.jamba_instruct_tokenizer import JambaInstructTokenizer, AsyncJambaInstructTokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer, AsyncJurassicTokenizer

_LOCAL_RESOURCES_PATH = Path(__file__).parent / "resources"
_ENV_CACHE_DIR_KEY = "AI21_TOKENIZER_CACHE_DIR"
JAMBA_TOKENIZER_HF_PATH = "ai21labs/Jamba-v0.1"


class PreTrainedTokenizers:
    J2_TOKENIZER = "j2-tokenizer"
    JAMBA_INSTRUCT_TOKENIZER = "jamba-instruct-tokenizer"
    JAMBA_TOKENIZER = "jamba-tokenizer"


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
