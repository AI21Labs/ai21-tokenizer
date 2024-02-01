#!/usr/bin/env bash

set -u -e -o pipefail

if [[ $# -gt 0 && $1 == "--fix" ]]; then
  inv formatter --fix
else
  inv formatter
fi

inv audit
inv lint
inv test
inv isort

set +e +u
