#!/usr/bin/env bash
set -e

INSTALL_OPTION=${1:-"dev"}

PIP=pip

echo 'Uninstalling stuff'
${PIP} uninstall -y nemo_text_processing

${PIP} install -U setuptools

echo 'Installing nemo'
if [[ "$INSTALL_OPTION" == "dev" ]]; then
    ${PIP} install --editable ".[all]"
else
    rm -rf dist/
    ${PIP} install build
    python -m build --no-isolation --wheel
    DIST_FILE=$(find ./dist -name "*.whl" | head -n 1)
    ${PIP} install "${DIST_FILE}[all]"
fi

echo 'All done!'
