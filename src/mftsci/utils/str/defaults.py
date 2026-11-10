"""
Defaults module for string utilities.
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
from mftsci.utils.arr.defaults import a     # noqa: F401
from mftsci.utils.str.core import *     # noqa: F401, F403
from mftsci.utils.str.core import get_matplotlib_use_tex

# =====================================
# DEFAULTS
# =====================================


def null(
            null: Optional[Union[np.ndarray, list, str]] = None,
            ) -> str:

    return set_default(val=null, default="")


def s(
        s: Optional[Union[np.ndarray, list, str]] = None,
        ) -> str:

    return set_default(val=s, default="")


def join_str(
            join_str: Optional[str] = None,
            ) -> str:

    return set_default(val=join_str, default="")


def mode(
            mode: Optional[str] = None,
            ) -> str:

    return set_default(val=mode, default="constant")


def app_null(
        app_null: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_null, default=True)


def app_pre(
        app_pre: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_pre, default=True)


def app_suff(
        app_suff: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_suff, default=True)


def rm_pre(
        rm_pre: Optional[bool] = None,
        ) -> bool:

    return set_default(val=rm_pre, default=True)


def rm_suff(
        rm_suff: Optional[bool] = None,
        ) -> bool:

    return set_default(val=rm_suff, default=True)


def require_pre(
        require_pre: Optional[bool] = None,
        ) -> bool:

    return set_default(val=require_pre, default=False)


def require_suff(
        require_suff: Optional[bool] = None,
        ) -> bool:

    return set_default(val=require_suff, default=False)


def pltf(
        pltf: Optional[bool] = None,
        ) -> bool:

    return set_default(val=pltf, default=False)


def use_tex(
            use_tex: Optional[bool] = None,
            ) -> bool:

    return set_default(val=use_tex, default=get_matplotlib_use_tex())
