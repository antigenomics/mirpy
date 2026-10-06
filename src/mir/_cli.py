"""Set signature kernel limits before importing the command implementation."""
import os
import sys


def main():
    if sys.argv[1:2] == ['signature']:
        # Keep this bootstrap independent of newer vdjtools entry points.
        for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'RAYON_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[key] = '1'
    from .cli import main as run

    run()
