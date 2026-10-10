"""
Defaults module for file utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional
import os


# External packages


# Internal packages
from mftsci.utils.meta.core import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401


# =====================================
# DEFAULTS
# =====================================


def mkdir(
        mkdir: Optional[bool] = None,
        ) -> bool:

    return set_default(val=mkdir, default=True)


def root(
        root: Optional[str] = None,
        ) -> str:

    return os.path.normpath(set_default(val=root, default=__root__))


def subdir(
        root: Optional[str] = None,
        ) -> str:

    return os.path.normpath(set_default(val=root, default=""))


def file(
        root: Optional[str] = None,
        ) -> str:

    return set_default(val=root, default="")


# =====================================
# VARIABLES
# =====================================

__root__ = "."
