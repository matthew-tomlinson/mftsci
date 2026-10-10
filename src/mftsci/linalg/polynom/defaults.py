"""
Defaults module for linalg.polynom module.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional


# External packages


# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401

# =====================================
# DEFAULTS
# =====================================


def min_degree(
        min_degree: Optional[int] = None,
        ) -> int:

    return set_default(min_degree, 0)


def max_degree(
        max_degree: Optional[int] = None,
        ) -> int:

    return set_default(max_degree, 1)
