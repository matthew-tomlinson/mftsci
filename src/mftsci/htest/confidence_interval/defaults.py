"""
Defaults module for confidence interval
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal

# External packages
import numpy as np


# Internal packages
from mftsci.utils.meta import set_default


# =====================================
# DEFAULTS
# =====================================


def alpha(
        alpha: Optional[Union[np.ndarray, list, float,]] = None,
        ) -> Union[np.ndarray, float,]:

    if alpha is None:
        alpha = 0.05
    elif isinstance(alpha, list):
        alpha = np.array(alpha)
    return alpha


def alphas(
        alpha: Optional[Union[np.ndarray, list, float,]] = None,
        ) -> Union[np.ndarray, float,]:

    return set_default(val=alpha, default=np.array([0.01, 0.05]))


def q(
        q: Optional[Union[np.ndarray, list, float,]] = None,
        ) -> Union[np.ndarray, float,]:

    if q is None:
        q = 0.5
    elif isinstance(q, list):
        q = np.array(q)
    return q


def tail(
            tail: Optional[Literal["both", "left", "right",]] = None,
            ) -> str:

    return set_default(val=tail, default="both")


def CI_color(
                color: Optional[Union[np.ndarray, list, str,]] = None,
                ) -> Union[np.ndarray, str,]:

    return set_default(val=color, default=["#d7d7d7", "#ebebeb"])
