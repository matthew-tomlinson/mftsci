"""
Defaults module for unit utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional


# External packages


# Internal packages
from mftsci.utils.meta.core import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401


# =====================================
# DEFAULTS
# =====================================


def loc(
        loc: Optional[object] = None,
        ) -> object:

    return set_default(val=loc, default=0)


def scale(
        scale: Optional[object] = None,
        ) -> object:

    return set_default(val=scale, default=1)


def symbol(
        symbol: Optional[str] = None,
        ) -> str:

    return set_default(val=symbol, default="")


def do(
        do: Optional[bool] = None,
        ) -> bool:

    return set_default(val=do, default=True)


def skip_loc(
        skip_loc: Optional[bool] = None,
        ) -> bool:

    return set_default(val=skip_loc, default=False)


def inv(
        inv: Optional[bool] = None,
        ) -> bool:

    return set_default(val=inv, default=False)


def inv_op(
        inv_op: Optional[bool] = None,
        ) -> bool:

    return set_default(val=inv_op, default=False)


def app_unit_suff(
        app_unit_suff: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_unit_suff, default=True)
