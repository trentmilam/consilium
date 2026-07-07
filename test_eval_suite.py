"""Thin pytest wrapper around ``eval/run_all.py``.

The eval suite is a set of narrative print-and-return-int scripts (wired into
CI via ``python eval/run_all.py``), which ``pytest`` does not discover on its
own. This wrapper shells out to the same aggregate entrypoint and asserts on
its exit code, so both commands work:

    python eval/run_all.py
    pytest
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def test_eval_suite_passes():
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "eval", "run_all.py")],
        cwd=ROOT,
    )
    assert result.returncode == 0, "eval/run_all.py did not exit 0 -- see stdout above for which eval failed"
