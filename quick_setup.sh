#!/usr/bin/env sh

python3 -m venv venv
. venv/bin/activate
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install pytest pylint
pip install -e . --no-build-isolation
