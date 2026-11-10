"""
Defaults module for Ljung--Box test.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional

# External packages


# Internal packages
from mftsci.utils.meta import set_default
from mftsci.htest.confidence_interval.defaults import alpha, tail     # noqa: F401
from mftsci.htest.htests.defaults import N, set_res     # noqa: F401
from mftsci.htest.htests.dist_tests.defaults import app_PIT     # noqa: F401


# =====================================
# DEFAULTS
# =====================================

def h(
        h: Optional[int] = None,
        ) -> int:

    return set_default(val=h, default=1)


def H_max(
        H_max: Optional[int] = None,
        ) -> int:

    return set_default(val=H_max, default=20)


def h_test(
        h_test: Optional[int] = None,
        ) -> int:

    return set_default(val=h_test, default=5)


def H_plot_skip(
        H_plot_skip: Optional[int] = None,
        ) -> int:

    return set_default(val=H_plot_skip, default=1)
