#!/usr/bin/env sh

set -e

VERSION=${1:-MISSING}

echo "publishing version ${VERSION}"
poetry config repositories.repo https://"${PYTHON_ARTIFACT_REGISTRY}"/
poetry publish --build --repository repo
