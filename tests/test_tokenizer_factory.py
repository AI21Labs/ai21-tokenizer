from typing import Type

import pytest
from ai21_tokenizer import Tokenizer, BaseTokenizer
from ai21_tokenizer.jamba_instruct_tokenizer import JambaInstructTokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer


@pytest.mark.parametrize(
    ids=[
        "when_tokenizer_name_is_jurassic_tokenizer__should_return_jurassic_tokenizer",
        "when_tokenizer_name_is_jamba_tokenizer__should_return_jamba_tokenizer",
    ],
    argnames=["tokenizer_name", "expected_tokenizer_instance"],
    argvalues=[
        ("j2-tokenizer", JurassicTokenizer),
        ("jamba-tokenizer", JambaInstructTokenizer),
    ],
)
def test_tokenizer_factory__get_tokenizer(
    tokenizer_name: str, expected_tokenizer_instance: Type[BaseTokenizer]
) -> None:
    tokenizer = Tokenizer.get_tokenizer(tokenizer_name)

    assert tokenizer is not None
    assert isinstance(tokenizer, expected_tokenizer_instance)
