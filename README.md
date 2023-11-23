# AI21 Labs Tokenizer

## Installation

### pip

```bash
pip install ai21-tokenizer
```

### poetry

```bash
poetry add ai21-tokenizer
```

## Usage

### Tokenizer Creation

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

### Functions

#### Encode and Decode

These functions allow you to encode your text to a list of token ids and back to plaintext

```python
text_to_encode = "apple orange banana"
encoded_text = tokenizer.encode(text_to_encode)
print(f"Encoded text: {encoded_text}")

decoded_text = tokenizer.decode(encoded_text)
print(f"Decoded text: {encoded_text}")
```

#### What if you had wanted to convert your tokens to ids or vice versa?

```python
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
print(f"IDs corresponds to Tokens: {tokens}")

ids = tokenizer.convert_tokens_to_ids(tokens)
```

**For more examples, please see our [examples](examples) folder.**
