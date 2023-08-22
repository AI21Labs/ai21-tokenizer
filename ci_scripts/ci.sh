#!/usr/bin/env bash

set -u -o pipefail

CWD=$(dirname "$0")
pushd "$CWD" > /dev/null || exit

cd ..
inv audit
inv formatter
inv lint
inv test
inv isort

popd > /dev/null || exit

set +e +u
