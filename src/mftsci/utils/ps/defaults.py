"""
Defaults module for string prefix and suffix utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional


# External packages


# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.str.core import *     # noqa: F401, F403
from mftsci.utils.str.defaults import *     # noqa: F401, F403


# =====================================
# DEFAULTS
# =====================================


def phantom(
        phantom: Optional[bool] = None,
        ) -> bool:

    return set_default(val=phantom, default=False)


def null_phantom(
        null_phantom: Optional[bool] = None,
        ) -> bool:

    return set_default(val=null_phantom, default=False)


def brac_type(
        brac_type: Optional[str] = None,
        ) -> str:

    return set_default(val=brac_type, default="()")


def repeat(
        repeat: Optional[int] = None,
        ) -> int:

    return set_default(val=repeat, default=1)


def app_null(
        app_null: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_null, default=False)


def tex_fn(
        tex_fn: Optional[str] = None,
        ) -> str:

    return set_default(val=tex_fn, default="")


def tex_num(
        tex_num: Optional[str] = None,
        ) -> str:

    return set_default(val=tex_num, default="hphantom")


def tex_hphantom(
        tex_hphantom: Optional[str] = None,
        ) -> str:

    return set_default(val=tex_hphantom, default="hphantom")
