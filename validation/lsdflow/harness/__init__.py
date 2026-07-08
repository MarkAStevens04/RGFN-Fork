"""LSD-Flow analysis harness (proposal §3b). ``run.py`` drives one (model x reward); the
matrix driver (``matrix.py``, phase 2) composes the full sweep."""

from validation.lsdflow.harness.config import LSDFlowRunConfig  # noqa: F401
from validation.lsdflow.harness.run import run  # noqa: F401
