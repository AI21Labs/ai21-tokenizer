from .version import VERSION
from ai21_tokenizer.tokenizer_factory import TokenizerFactory as Tokenizer
from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer

__version__ = VERSION

__all__ = ["Tokenizer", "JurassicTokenizer", "__version__"]
