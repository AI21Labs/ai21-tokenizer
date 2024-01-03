#!/usr/bin/env bash

# create .git folder
if [[ ! -d .git ]]; then
  git init
fi

# install the python version specified in .python-version, if not already installed
if pyenv --version; then
  pyenv install --skip-existing
fi

{ [[ -d .venv ]] || {
  echo 'creating virtualenv...'
  python -m venv .venv
}; } && {
  # shellcheck disable=SC1091
  . .venv/bin/activate
} && {
  # install poetry if not already installed
  poetry --version || brew install poetry
} && {
  # install keyring
  poetry self add keyrings-google-artifactregistry-auth@1.1.2
} && {
  # update lock file
  poetry lock --no-update
} && {
  # install dependencies
  poetry install --no-root --sync
}

{
  # install pre-commit if not already installed
  pre-commit --version || brew install pre-commit
} && {
  # install pre-commit hooks
  pre-commit install --install-hooks -t pre-commit -t pre-push -t commit-msg
}
