from pathlib import Path

import pytest

from pytest_mock import MockerFixture

from ai21_tokenizer import (
    AsyncJamba1_5Tokenizer,
    AsyncJambaInstructTokenizer,
    Jamba1_5Tokenizer,
    JambaInstructTokenizer,
    PreTrainedTokenizers,
    Tokenizer,
)
from ai21_tokenizer.jamba_1_5_tokenizer import AsyncJambaTokenizer, SyncJambaTokenizer


@pytest.fixture
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def jamba_instruct_tokenizer() -> JambaInstructTokenizer:
    jamba_tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER)

    if isinstance(jamba_tokenizer, JambaInstructTokenizer):
        return jamba_tokenizer

    raise ValueError("JambaInstructTokenizer not found")


@pytest.fixture
async def async_jamba_instruct_tokenizer() -> AsyncJambaInstructTokenizer:
    jamba_tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER)

    if isinstance(jamba_tokenizer, AsyncJambaInstructTokenizer):
        return jamba_tokenizer

    raise ValueError("AsyncJambaInstructTokenizer not found")


@pytest.fixture
def mock_async_jamba_instruct_tokenizer(mocker: MockerFixture) -> AsyncJambaInstructTokenizer:
    return mocker.MagicMock(spec=AsyncJambaInstructTokenizer)


@pytest.fixture(scope="session")
def jamba_1_5_mini_tokenizer() -> Jamba1_5Tokenizer:
    jamba_1_5_mini_tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_1_5_MINI_TOKENIZER)

    if isinstance(jamba_1_5_mini_tokenizer, Jamba1_5Tokenizer):
        return jamba_1_5_mini_tokenizer

    raise ValueError("Jamba1_5Tokenizer not found")


@pytest.fixture
async def async_jamba_1_5_mini_tokenizer() -> AsyncJamba1_5Tokenizer:
    jamba_1_5_mini_tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_1_5_MINI_TOKENIZER)

    if isinstance(jamba_1_5_mini_tokenizer, AsyncJamba1_5Tokenizer):
        return jamba_1_5_mini_tokenizer

    raise ValueError("AsyncJamba1_5Tokenizer not found")


@pytest.fixture(scope="session")
def jamba_1_5_large_tokenizer() -> SyncJambaTokenizer:
    jamba_1_5_large_tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_1_5_LARGE_TOKENIZER)

    if isinstance(jamba_1_5_large_tokenizer, SyncJambaTokenizer):
        return jamba_1_5_large_tokenizer

    raise ValueError("Jamba1_5Tokenizer not found")


@pytest.fixture
async def async_jamba_1_5_large_tokenizer() -> AsyncJambaTokenizer:
    jamba_1_5_large_tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_1_5_LARGE_TOKENIZER)

    if isinstance(jamba_1_5_large_tokenizer, AsyncJambaTokenizer):
        return jamba_1_5_large_tokenizer

    raise ValueError("AsyncJamba1_5Tokenizer not found")
