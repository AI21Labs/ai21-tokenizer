# Jurassic Tokenization

---

## Installation

---

### pip

```bash
pip install ai21-tokenizer
```

### poetry

```bash
poetry add ai21-tokenizer
```

## Usage

---

```python
from ai21_tokenizer import Tokenizer

tokenizer = Tokenizer.get_tokenizer()
# Your code here
```

Another way would be to use our Jurassic model directly:

```python
from ai21_tokenizer import JurassicTokenizer

model_path = "<Path to your vocabs file. This is usually a binary file that end with .model>"
config = {} # "dictionary object of your config.json file"
tokenizer = JurassicTokenizer(model_path=model_path, config=config)
```

For more examples, please see our [examples](examples) folder.
