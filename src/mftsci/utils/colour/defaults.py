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


# Internal packages
from mftsci.utils.meta import set_default as _set_default
from mftsci.utils.colour.core import set_default, _app_to_rgba, _app_to_hex, \
                                        _c_back as c_back

# =====================================
# DEFAULTS
# =====================================


def invert_with_theme(
        invert_with_theme: Optional[int] = None,
        ) -> int:

    return _set_default(val=invert_with_theme, default=0)


def invert_colors(
        invert_colors: Optional[bool] = None,
        ) -> bool:

    return _set_default(val=invert_colors, default=True)


def invert_bad(
        invert_bad: Optional[bool] = None,
        ) -> bool:

    return _set_default(val=invert_bad, default=False)


def app_to_rgba(
                        app_to_rgba: Optional[bool] = None,
                        ) -> bool:

    return _app_to_rgba(app_to_rgba=app_to_rgba)


def app_to_hex(
                        app_to_hex: Optional[bool] = None,
                        ) -> bool:

    return _app_to_hex(app_to_hex=app_to_hex)


def c_fore(
                    c_fore: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_rgba: Optional[bool] = None,
                    ) -> np.ndarray:

    return set_default(c=c_fore, default=plt.rcParams["axes.edgecolor"],
                       app_to_rgba=app_to_rgba)[0]


def cs(
                cs: Optional[np.ndarray] = None,
                app_to_rgba: Optional[bool] = None,
                ) -> np.ndarray:

    return set_default(c=cs, default=np.array([c_back(), c_fore()]),
                       app_to_rgba=app_to_rgba)


def cmap(
        cmap: Optional[str] = None,
        ) -> str:

    return _set_default(val=cmap, default="viridis")


def v(
        v: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return _set_default(val=v, default=1.0)


def max_contrast_0(
        max_contrast_0: Optional[bool] = None,
        ) -> bool:

    return _set_default(val=max_contrast_0, default=True)


def N_c(
        N_c: Optional[int] = None,
        ) -> int:

    return _set_default(val=N_c, default=1)


def i_0(
        i_0: Optional[int] = None,
        ) -> int:

    return _set_default(val=i_0, default=1)


def reverse(
        reverse: Optional[bool] = None,
        ) -> bool:

    return _set_default(val=reverse, default=False)


def invert_colour(
        invert_colour: Optional[bool] = None,
        ) -> bool:

    return _set_default(val=invert_colour, default=False)


def cmode(
        cmode: Optional[str] = None,
        ) -> str:

    return _set_default(val=cmode, default="background")
