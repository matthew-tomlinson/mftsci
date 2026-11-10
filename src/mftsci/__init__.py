"""Top-level package for mftsci."""

__author__ = """Matthew Frank Tomlinson"""
__email__ = "mtthw.tomlinson@gmail.com"
__version__ = "0.0.1.dev20260824"

from . import (    # noqa: F401
                        utils,
                        linalg, plot,
                        htest, point_proc, tdist,
                        )
from .utils import (  # noqa: F401
                            sklearn,
                            cls,
                            )
