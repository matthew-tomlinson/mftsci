"""
Module for file utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional
import os

# External packages

# Internal packages


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def mkdir_new(
                path: str,
                mkdir: Optional[bool] = None,
                ) -> str:

    if mkdir is None:
        mkdir = True
    if mkdir:
        if not os.path.exists(path):
            os.mkdir(path)
    return path


def calc_root(
                root: Optional[str] = None,
                mkdir: Optional[bool] = None,
                ) -> str:

    if mkdir is None:
        mkdir = True

    if root is None:
        root = "./"

    return mkdir_new(path=root)
