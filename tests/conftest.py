from pathlib import Path

import pytest

from jurassic_tokenization import JurassicTokenizer


@pytest.fixture
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def tokenizer() -> JurassicTokenizer:
    return JurassicTokenizer.from_pretrained("j2-tokenizer")