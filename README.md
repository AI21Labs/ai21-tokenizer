<h1 align="center">
    <a href="https://github.com/AI21Labs/ai21-tokenizer">AI21 Labs Tokenizer</a>
</h1>

<p align="center">
    <em>A SentencePiece based tokenizer for production uses with AI21's models</em>
</p>

<p align="center">
<a href="https://github.com/AI21Labs/ai21-tokenizer/actions?query=workflow%3ATest+event%3Apush+branch%3Amain"><img src="https://github.com/AI21Labs/ai21-tokenizer/actions/workflows/test.yaml/badge.svg" alt="Test"></a>
<a href="https://pypi.org/project/ai21-tokenizer" target="_blank"><img src="https://img.shields.io/pypi/v/ai21-tokenizer?color=%2334D058&label=pypi%20package" alt="Package version"></a>
<a href="https://pypi.org/project/ai21-tokenizer" target="_blank"><img src="https://img.shields.io/pypi/pyversions/ai21-tokenizer?color=%2334D058" alt="Supported Python versions"></a>
<a href="https://python-poetry.org/" target="_blank"><img src="https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json" alt="Poetry"></a>
<a href="https://github.com/semantic-release/semantic-release" target="_blank"><img src="https://img.shields.io/badge/semantic--release-python-e10079?logo=semantic-release" alt="Supported Python versions"></a>
<a href="https://opensource.org/licenses/Apache-2.0" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

---

## Prerequisites

- If you wish to use the tokenizers for `Jamba 1.5 Mini` or `Jamba 1.5 Large`, you will need to request access to the relevant model's HuggingFace repo:
  - [Jamba 1.5 Mini](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini)
  - [Jamba 1.5 Large](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large)

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

### Jamba 1.5 Mini Tokenizer

```python
from ai21_tokenizer import Tokenizer, PreTrainedTokenizers

tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_1_5_MINI_TOKENIZER)
# Your code here
```

Another way would be to use our Jamba 1.5 Mini tokenizer directly:

```python
from ai21_tokenizer import Jamba1_5Tokenizer

model_path = "<Path to your vocabs file>"
tokenizer = Jamba1_5Tokenizer(model_path=model_path)
# Your code here
```

#### Async usage

```python
from ai21_tokenizer import Tokenizer, PreTrainedTokenizers

tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_1_5_MINI_TOKENIZER)
# Your code here
```

### Jamba 1.5 Large Tokenizer

```python
from ai21_tokenizer import Tokenizer, PreTrainedTokenizers

tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_1_5_LARGE_TOKENIZER)
# Your code here
```

Another way would be to use our Jamba 1.5 Large tokenizer directly:

```python
from ai21_tokenizer import Jamba1_5Tokenizer

model_path = "<Path to your vocabs file>"
tokenizer = Jamba1_5Tokenizer(model_path=model_path)
# Your code here
```

#### Async usage

```python
from ai21_tokenizer import Tokenizer, PreTrainedTokenizers

tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_1_5_LARGE_TOKENIZER)
# Your code here
```

### Jamba Instruct Tokenizer

```python
from ai21_tokenizer import Tokenizer, PreTrainedTokenizers

tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER)
# Your code here
```

Another way would be to use our Jamba tokenizer directly:

```python
from ai21_tokenizer import JambaInstructTokenizer

model_path = "<Path to your vocabs file>"
tokenizer = JambaInstructTokenizer(model_path=model_path)
# Your code here
```

#### Async usage

```python
from ai21_tokenizer import Tokenizer, PreTrainedTokenizers

tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER)
# Your code here
```

Another way would be to use our async Jamba tokenizer class method create:

```python
from ai21_tokenizer import AsyncJambaInstructTokenizer

model_path = "<Path to your vocabs file>"
tokenizer = AsyncJambaInstructTokenizer.create(model_path=model_path)
# Your code here
```

### J2 Tokenizer

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

#### Async usage

```python
from ai21_tokenizer import Tokenizer

tokenizer = await Tokenizer.get_async_tokenizer()
# Your code here
```

Another way would be to use our async Jamba tokenizer class method create:

```python
from ai21_tokenizer import AsyncJurassicTokenizer

model_path = "<Path to your vocabs file. This is usually a binary file that end with .model>"
config = {} # "dictionary object of your config.json file"
tokenizer = AsyncJurassicTokenizer.create(model_path=model_path, config=config)
# Your code here
```

### Functions

#### Encode and Decode

These functions allow you to encode your text to a list of token ids and back to plaintext

```python
text_to_encode = "apple orange banana"
encoded_text = tokenizer.encode(text_to_encode)
print(f"Encoded text: {encoded_text}")

decoded_text = tokenizer.decode(encoded_text)
print(f"Decoded text: {decoded_text}")
```

#### Async

```python
# Assuming you have created an async tokenizer
text_to_encode = "apple orange banana"
encoded_text = await tokenizer.encode(text_to_encode)
print(f"Encoded text: {encoded_text}")

decoded_text = await tokenizer.decode(encoded_text)
print(f"Decoded text: {decoded_text}")
```

#### What if you had wanted to convert your tokens to ids or vice versa?

```python
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
print(f"IDs corresponds to Tokens: {tokens}")

ids = tokenizer.convert_tokens_to_ids(tokens)
```

#### Async

```python
# Assuming you have created an async tokenizer
tokens = await tokenizer.convert_ids_to_tokens(encoded_text)
print(f"IDs corresponds to Tokens: {tokens}")

ids = tokenizer.convert_tokens_to_ids(tokens)
```

**For more examples, please see our [examples](examples) folder.**
