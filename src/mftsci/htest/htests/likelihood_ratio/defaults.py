"""
Defaults module for Likelihood--Ratio test.
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


def ascending(
                ascending: Optional[bool] = None,
                ) -> bool:

    return set_default(val=ascending, default=False)


def N_mods(
                N_mods: Optional[int] = None,
                ) -> int:

    return set_default(val=N_mods, default=2)
