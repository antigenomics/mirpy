python3 -m venv venv
. venv/bin/activate.fish
export CMAKE_POLICY_VERSION_MINIMUM=3.5 && pip install .
pip install pytest pylint