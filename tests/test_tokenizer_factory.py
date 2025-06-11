from typing import Type

import pytest

from ai21_tokenizer import BaseTokenizer, Tokenizer
from ai21_tokenizer.jamba_1_5_tokenizer import AsyncJambaTokenizer, SyncJambaTokenizer
from ai21_tokenizer.jamba_instruct_tokenizer import (
    AsyncJambaInstructTokenizer,
    JambaInstructTokenizer,
)


@pytest.mark.parametrize(
    ids=[
        "when_tokenizer_name_is_jamba_tokenizer__should_return_jamba_tokenizer",
        "when_tokenizer_name_is_jamba_mini_1_6_tokenizer__should_return_jamba_mini_1_6_tokenizer",
        "when_tokenizer_name_is_jamba_large_1_6_tokenizer__should_return_jamba_large_1_6_tokenizer",
    ],
    argnames=["tokenizer_name", "expected_tokenizer_instance"],
    argvalues=[
        pytest.param(
            "jamba-tokenizer",
            JambaInstructTokenizer,
        ),
        pytest.param(
            "jamba-mini-1.6-tokenizer",
            SyncJambaTokenizer,
        ),
        pytest.param(
            "jamba-large-1.6-tokenizer",
            SyncJambaTokenizer,
        ),
    ],
)
def test_tokenizer_factory__get_tokenizer(
    tokenizer_name: str, expected_tokenizer_instance: Type[BaseTokenizer]
) -> None:
    tokenizer = Tokenizer.get_tokenizer(tokenizer_name)

    assert tokenizer is not None
    assert isinstance(tokenizer, expected_tokenizer_instance)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=[
        "when_tokenizer_name_is_jamba_tokenizer__should_return_async_jamba_tokenizer",
        "when_tokenizer_name_is_jamba_mini_1_6_tokenizer__should_return_async_jamba_mini_1_6_tokenizer",
        "when_tokenizer_name_is_jamba_large_1_6_tokenizer__should_return_async_jamba_large_1_6_tokenizer",
    ],
    argnames=["tokenizer_name", "expected_tokenizer_instance"],
    argvalues=[
        pytest.param(
            "jamba-tokenizer",
            AsyncJambaInstructTokenizer,
        ),
        pytest.param(
            "jamba-mini-1.6-tokenizer",
            AsyncJambaTokenizer,
        ),
        pytest.param(
            "jamba-large-1.6-tokenizer",
            AsyncJambaTokenizer,
        ),
    ],
)
async def test_tokenizer_factory__get_async_tokenizer(
    tokenizer_name: str, expected_tokenizer_instance: Type[BaseTokenizer]
) -> None:
    tokenizer = await Tokenizer.get_async_tokenizer(tokenizer_name)

    assert tokenizer is not None
    assert isinstance(tokenizer, expected_tokenizer_instance)


def test_tokenizer__when_tokenizer_name_is_not_supported__should_raise_value_error() -> None:
    with pytest.raises(ValueError):
        Tokenizer.get_tokenizer(tokenizer_name="unsupported")
