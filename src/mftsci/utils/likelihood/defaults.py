"""
Defaults module for likelihood utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional


# External packages


# Internal packages
from mftsci.utils.meta.core import set_default


# =====================================
# DEFAULTS
# =====================================


def opt(
        opt: Optional[bool] = None,
        ) -> bool:

    return set_default(val=opt, default=True)


def free_only(
        free_only: Optional[bool] = None,
        ) -> bool:

    return set_default(val=free_only, default=False)


def frac(
        frac: Optional[float] = None,
        ) -> float:

    return set_default(val=frac, default=1E-5)


def min_step(
        min_step: Optional[float] = None,
        ) -> float:

    return set_default(val=min_step, default=1E-8)


def N_data(
        N_data: Optional[int] = None,
        ) -> int:

    return set_default(val=N_data, default=0)
