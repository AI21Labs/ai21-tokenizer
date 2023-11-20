from typing import Optional, Type

import pytest

from tokenizer import TokenizerFactory, Tokenizer
from tokenizer.jurassic_tokenizer import JurassicTokenizer


def test_tokenizer_factory__get_tokenizer__when_receives_unknown_name__should_raise():
    with pytest.raises(ValueError):
        TokenizerFactory.get_tokenizer("unknown_name")


@pytest.mark.parametrize(
    ids=[
        "when_tokenizer_name_is_known_string__should_return_tokenizer",
        "when_tokenizer_name_is_None__should_return_tokenizer",
    ],
    argnames=["tokenizer_name", "tokenizer_instance"],
    argvalues=[
        ("j2-tokenizer", JurassicTokenizer),
        (None, JurassicTokenizer),
    ],
)
def test_tokenizer_factory__get_tokenizer__when_receives_j2_tokenizer__should_return_jurassic_tokenizer(
    tokenizer_name: Optional[str], tokenizer_instance: Type[Tokenizer]
):
    if tokenizer_name is None:
        tokenizer = TokenizerFactory.get_tokenizer()
    else:
        tokenizer = TokenizerFactory.get_tokenizer(tokenizer_name)

    assert tokenizer is not None
    assert isinstance(tokenizer, tokenizer_instance)
