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

- If you wish to use the tokenizers for `Jamba Mini` or `Jamba Large`, you will need to request access to the relevant model's HuggingFace repo:
  - [Jamba Mini](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6)
  - [Jamba Large](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6)

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

### Basic Usage

```python
from ai21_tokenizer import Tokenizer

# Create tokenizer (defaults to Jamba Mini)
tokenizer = Tokenizer.get_tokenizer()

# Encode text to token IDs
text = "Hello, world!"
encoded = tokenizer.encode(text)
print(f"Encoded: {encoded}")

# Decode token IDs back to text
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
```

### Specific Tokenizer Selection

```python
from ai21_tokenizer import Tokenizer, PreTrainedTokenizers

# Jamba Mini tokenizer
tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_MINI_TOKENIZER)

# Jamba Large tokenizer
tokenizer = Tokenizer.get_tokenizer(PreTrainedTokenizers.JAMBA_LARGE_TOKENIZER)
```

### Async Usage

```python
import asyncio
from ai21_tokenizer import Tokenizer

async def main():
    tokenizer = await Tokenizer.get_async_tokenizer()

    text = "Hello, world!"
    encoded = await tokenizer.encode(text)
    decoded = await tokenizer.decode(encoded)

    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

asyncio.run(main())
```

### Advanced Token Operations

```python
# Convert between tokens and IDs
tokens = tokenizer.convert_ids_to_tokens(encoded)
print(f"Tokens: {tokens}")

ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"IDs: {ids}")
```

### Direct Class Usage

```python
from ai21_tokenizer import SyncJambaTokenizer

# Using local model file
model_path = "/path/to/your/tokenizer.model"
tokenizer = SyncJambaTokenizer(model_path=model_path)

text = "Hello, world!"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
```

### Async Direct Class Usage

```python
from ai21_tokenizer import AsyncJambaTokenizer

async def main():
    model_path = "/path/to/your/tokenizer.model"
    tokenizer = await AsyncJambaTokenizer.create(model_path=model_path)

    text = "Hello, world!"
    encoded = await tokenizer.encode(text)
    decoded = await tokenizer.decode(encoded)

asyncio.run(main())
```

**For more examples, please see our [examples](examples) folder.**
