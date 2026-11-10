"""
Defaults module for colour utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union


# External packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs


# Internal packages
from mftsci.utils.meta import set_default as _set_default


# =====================================
# FUNCTIONS
# =====================================


def is_dark_theme(
                    c_back: Optional[Union[np.ndarray, list, str,]] = None,
                    ) -> bool:

    c_back = _c_back(c_back=c_back, app_to_rgba=True)
    return np.mean(c_back[:3]) < 0.5


def capp_to_rgba(
                    c: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_rgba: Optional[bool] = None,
                    ) -> np.ndarray:

    app_to_rgba = _app_to_rgba(app_to_rgba=app_to_rgba)
    if app_to_rgba:
        return clrs.to_rgba_array(c)
    else:
        return c


def capp_to_hex(
                    c: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_hex: Optional[bool] = None,
                    ) -> np.ndarray:

    app_to_hex = _app_to_hex(app_to_hex=app_to_hex)
    if app_to_hex:
        return clrs.to_hex(c)
    else:
        return c


def set_default(
                    c: Optional[Union[np.ndarray, list, str,]] = None,
                    default: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_rgba: Optional[bool] = None,
                    ) -> np.ndarray:

    c = _set_default(val=c, default=default)

    return capp_to_rgba(c=c, app_to_rgba=app_to_rgba)


# =====================================
# DEFAULTS
# =====================================


def _app_to_rgba(
                        app_to_rgba: Optional[bool] = None,
                        ) -> bool:

    return _set_default(val=app_to_rgba, default=True)


def _app_to_hex(
                        app_to_hex: Optional[bool] = None,
                        ) -> bool:

    return _set_default(val=app_to_hex, default=False)


def _c_back(
                    c_back: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_rgba: Optional[bool] = None,
                    ) -> np.ndarray:

    return set_default(c=c_back, default=plt.rcParams["axes.facecolor"],
                       app_to_rgba=app_to_rgba)[0]
