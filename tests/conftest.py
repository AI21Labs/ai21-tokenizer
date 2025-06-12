from pathlib import Path

import pytest
import pytest_asyncio

from ai21_tokenizer import PreTrainedTokenizers, Tokenizer
from ai21_tokenizer.jamba_tokenizer import AsyncJambaTokenizer, SyncJambaTokenizer


@pytest.fixture
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def jamba_mini_tokenizer() -> SyncJambaTokenizer:
    jamba_mini_tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_MINI_TOKENIZER)

    if isinstance(jamba_mini_tokenizer, SyncJambaTokenizer):
        return jamba_mini_tokenizer

    raise ValueError("SyncJambaTokenizer not found")


@pytest_asyncio.fixture(scope="session")
async def async_jamba_mini_tokenizer() -> AsyncJambaTokenizer:
    jamba_mini_tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_MINI_TOKENIZER)

    if isinstance(jamba_mini_tokenizer, AsyncJambaTokenizer):
        return jamba_mini_tokenizer

    raise ValueError("AsyncJambaTokenizer not found")


@pytest.fixture(scope="session")
def jamba_large_tokenizer() -> SyncJambaTokenizer:
    jamba_large_tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_LARGE_TOKENIZER)

    if isinstance(jamba_large_tokenizer, SyncJambaTokenizer):
        return jamba_large_tokenizer

    raise ValueError("SyncJambaTokenizer not found")


@pytest_asyncio.fixture(scope="session")
async def async_jamba_large_tokenizer() -> AsyncJambaTokenizer:
    jamba_large_tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_LARGE_TOKENIZER)

    if isinstance(jamba_large_tokenizer, AsyncJambaTokenizer):
        return jamba_large_tokenizer

    raise ValueError("AsyncJambaTokenizer not found")
