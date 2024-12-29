"""
Module for colour utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union

# External packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

# Internal packages


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def is_dark_theme(
                    c_back: Optional[Union[np.ndarray, list, str,]] = None,
                    ) -> bool:

    c_back = default_c_back(c_back=c_back, app_to_rgba=True)
    return np.mean(c_back[:3]) < 0.5


def capp_to_rgba(
                    c: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_rgba: Optional[bool] = None,
                    ) -> np.ndarray:

    app_to_rgba = default_app_to_rgba(app_to_rgba=app_to_rgba)
    if app_to_rgba:
        return clrs.to_rgba_array(c)
    else:
        return c


def c_diff(
            cs: Optional[np.ndarray] = None,
            ) -> np.ndarray:

    cs = default_cs(cs=cs, app_to_rgba=True)
    return np.diff(cs, axis=0)


def c_vadd_diff(
                cs: Optional[np.ndarray] = None,
                v: Optional[Union[float, int,]] = None,
                ) -> np.ndarray:

    if v is None:
        v = 1.0

    cs = default_cs(cs=cs, app_to_rgba=True)
    diff = c_diff(cs=cs)

    v_diff = np.multiply.outer(v, diff)

    return v_diff + cs[:-1, ...]


def c_vadd_diff_back_fore(
                            v: Optional[Union[float, int,]] = None,
                            ) -> np.ndarray:

    return c_vadd_diff(cs=default_cs(app_to_rgba=True), v=v)


def c_grad_cycle(
                    cs: Optional[np.ndarray] = None,
                    N_c: Optional[int] = None,
                    N_grad: Optional[int] = None,
                    step: Optional[Union[float, int,]] = None,
                    i_0: Optional[int] = None,
                    max_contrast_0: Optional[bool] = None,
                    ) -> np.ndarray:

    if max_contrast_0 is None:
        max_contrast_0 = True

    if N_c is None:
        N_c = 1

    if N_grad is None:
        N_grad = N_c

    if i_0 is None:
        i_0 = 1

    if cs is None:
        cs = default_cs(cs=cs, app_to_rgba=True)

    if step is None:
        if np.mean(cs[0, :3]) < 0.5:
            # step = 30/255
            step = 40/255
        else:
            step = 20/255

    c_grad = c_vadd_diff(v=(i_0+np.arange(N_grad))*step)[:, 0, ...]
    if max_contrast_0:
        c_grad = np.flip(c_grad, axis=0)

    N_cycle = 1+((N_c-1)//N_grad)

    c_grad_cyc = np.concatenate([c_grad for k in range(N_cycle)])[:N_c]

    if False:
        _c_grad_cyc = np.concatenate([c_grad for k in range(N_cycle)])
        if max_contrast_0:
            c_grad_cyc = np.flip(_c_grad_cyc[-N_c:], axis=0)
        else:
            c_grad_cyc = _c_grad_cyc[:N_c]

    return c_grad_cyc


# Defaults
# -------------------------------------


def default_app_to_rgba(
                        app_to_rgba: Optional[bool] = None,
                        ) -> bool:
    if app_to_rgba is None:
        app_to_rgba = True
    return app_to_rgba


def default_c_fore(
                    c_fore: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_rgba: Optional[bool] = None,
                    ) -> np.ndarray:

    if c_fore is None:
        c_fore = plt.rcParams["axes.edgecolor"]

    return capp_to_rgba(c=c_fore, app_to_rgba=app_to_rgba)[0]


def default_c_back(
                    c_back: Optional[Union[np.ndarray, list, str,]] = None,
                    app_to_rgba: Optional[bool] = None,
                    ) -> np.ndarray:

    if c_back is None:
        c_back = plt.rcParams["axes.facecolor"]
    return capp_to_rgba(c=c_back, app_to_rgba=app_to_rgba)[0]


def default_cs(
                cs: Optional[np.ndarray] = None,
                app_to_rgba: Optional[bool] = None,
                ) -> np.ndarray:

    if cs is None:
        cs = np.array([default_c_back(), default_c_fore()])
    return capp_to_rgba(c=cs, app_to_rgba=app_to_rgba)
