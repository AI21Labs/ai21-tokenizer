# Jurassic Tokenization

---

## Installation

---

### pip

```bash
pip install ai21_tokenizer
```

### poetry

```bash
poetry add ai21_tokenizer
```

## Usage

---

```python
from jurassic_tokenization import JurassicTokenizer

tokenizer = JurassicTokenizer.from_pretrained('j2-tokenizer')
# Your code here
```

## Contribute

---

### Prerequisites

- [pyenv](https://github.com/pyenv/pyenv)
- [poetry](https://python-poetry.org/)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/AI21Labs/ai21_tokenizer.git
   ```

2. Set up a virtual environment with the `init` script

   ```bash
   source ./init.sh
   ```

3. Run validation by leveraging [pre-commit](https://pre-commit.com)
   1. Install `pre-commit install --install-hooks -t pre-commit -t commit-msg`
   2. To run on-demand `pre-commit run -a`
4. Submit a pull-request

### Run CI tasks locally

```bash
$ inv --list
Available tasks:

  clean          clean (remove) packages
  lint           python lint
  outdated       outdated packages
  test           Run unit tests
  update         update packages
  audit          run safety checks on project dependencies
```

## Publish

Package will be published to our _internal python registry_ defined by `_AR_PYTHON_REPO`.

- In order to publish from a side branch a release candidate, update your branch you should checkout to a branch named `dev` and commit to upstream. Please note that in order to be [PEP compliant](https://peps.python.org/pep-0440/#pre-releases) pre-release branches should be one of the following: `dev`, `alpha` or `beta`.
- Once merging to `master` a new version will be published by [semantic-release](https://github.com/semantic-release/semantic-release)
