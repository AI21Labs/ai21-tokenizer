from pathlib import Path

import pytest

from tokenizer import Tokenizer
from tokenizer.jurassic_tokenizer import JurassicTokenizer


@pytest.fixture
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def tokenizer() -> JurassicTokenizer:
    jurassic_tokenizer = Tokenizer.get_tokenizer()

    if isinstance(jurassic_tokenizer, JurassicTokenizer):
        return jurassic_tokenizer

    raise ValueError("JurassicTokenizer not found")
