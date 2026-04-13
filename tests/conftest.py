import os

import pytest


RUN_BENCHMARKS = os.getenv("RUN_BENCHMARKS") == "1"
RUN_INTEGRATION = os.getenv("RUN_INTEGRATION") == "1"

skip_benchmarks = pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="set RUN_BENCHMARKS=1 to run benchmark tests",
)

skip_integration = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set RUN_INTEGRATION=1 to run integration tests",
)
