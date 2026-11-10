"""
Defaults module for Jupyter notebook utilities.
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


def counter(
        counter: Optional[int] = None,
        ) -> int:

    return set_default(val=counter, default=1)


def caption(
        caption: Optional[str] = None,
        ) -> str:

    return set_default(val=caption, default="")


def id(
        id: Optional[str] = None,
        ) -> str:

    return set_default(val=id, default="")
