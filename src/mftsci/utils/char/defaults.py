"""
Defaults module for character utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union


# External packages
import numpy as np


# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.arr import iterarr as _iterarr
from mftsci.utils.char.core import *    # noqa: F401, F403
from mftsci.utils.str.defaults import *  # noqa: F401, F403


# =====================================
# DEFAULTS
# =====================================


def app_tex_vect(
        app_tex_vect: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_tex_vect, default=False)


def app_mathrm(
        app_mathrm: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_mathrm, default=False)


def capital(
        capital: Optional[bool] = None,
        ) -> bool:

    return set_default(val=capital, default=False)


def ow_omicron(
        ow_omicron: Optional[bool] = None,
        ) -> bool:

    return set_default(val=ow_omicron, default=False)


def app_omicron(
        app_omicron: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_omicron, default=False)


def blank(
        blank: Optional[bool] = None,
        ) -> bool:

    return set_default(val=blank, default=False)


def blank_str(
        blank_str: Optional[str] = None,
        ) -> str:

    return set_default(val=blank_str, default="")


def var(
        var: Optional[bool] = None,
        ) -> bool:

    return set_default(val=var, default=False)


def order_arg(
        order_arg: Optional[Union[np.ndarray, list, str,]] = None,
        ) -> np.ndarray:

    return _iterarr(set_default(val=order_arg, default=["arab", "roman", "greek"]))
