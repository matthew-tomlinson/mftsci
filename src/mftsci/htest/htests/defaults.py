"""
Defaults module for `htests` subpackage.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional

# External packages


# Internal packages
from mftsci.utils.meta import set_default


# =====================================
# DEFAULTS
# =====================================


def set_res(
        set_res: Optional[bool] = None,
        ) -> bool:

    return set_default(val=set_res, default=True)


def N(
        N: Optional[int] = None,
        ) -> bool:

    return set_default(val=N, default=100)
