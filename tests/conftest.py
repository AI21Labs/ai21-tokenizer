from pathlib import Path
from pytest_mock import MockerFixture


import pytest

from ai21_tokenizer import (
    Tokenizer,
    PreTrainedTokenizers,
    JambaInstructTokenizer,
    AsyncJambaInstructTokenizer,
    AsyncJurassicTokenizer,
)
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer


@pytest.fixture
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def tokenizer() -> JurassicTokenizer:
    jurassic_tokenizer = Tokenizer.get_tokenizer(tokenizer_name=PreTrainedTokenizers.J2_TOKENIZER)

    if isinstance(jurassic_tokenizer, JurassicTokenizer):
        return jurassic_tokenizer

    raise ValueError("JurassicTokenizer not found")


@pytest.fixture()
async def async_tokenizer() -> AsyncJurassicTokenizer:
    jurassic_tokenizer = await Tokenizer.get_async_tokenizer(tokenizer_name=PreTrainedTokenizers.J2_TOKENIZER)

    if isinstance(jurassic_tokenizer, AsyncJurassicTokenizer):
        return jurassic_tokenizer

    raise ValueError("AsyncJurassicTokenizer not found")


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
