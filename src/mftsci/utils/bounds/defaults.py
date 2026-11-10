"""
Defaults module for bounds utilities.
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
from mftsci.utils.meta.defaults import reset    # noqa: F401
from mftsci.utils.dict import keys


# =====================================
# DEFAULTS
# =====================================

def a(
        a: Optional[Union[float, int,]] = None
        ) -> Union[float, int,]:

    return set_default(val=a, default=0)


def b(
        b: Optional[Union[float, int,]] = None
        ) -> Union[float, int,]:

    return set_default(val=b, default=1)


def Delta(
        Delta: Optional[Union[float, int,]] = None
        ) -> Union[float, int,]:

    return set_default(val=a, default=b()-a())


def bounds_mode(
                        bounds_mode: Optional[str] = None,
                        ) -> str:

    return set_default(val=bounds_mode, default="unit")


def bounds(
                    bounds_mode: Optional[str] = None,
                    ) -> np.ndarray:

    bounds_mode = bounds_mode(bounds_mode=bounds_mode)

    if bounds_mode.lower()[:4] == "real":
        bounds = np.array([-np.inf, np.inf])
    else:
        bounds = np.array([0, 1])

    uscore = bounds.find("_")
    if uscore >= 0:
        postscore = bounds[1+uscore:]

        sign_str = {
                        "-": ["-", "neg"],
                        "+": ["+", "pos"],
                    }
        signs = keys(sign_str)

        for s in range(signs.size):
            _issign = np.any([  # noqa: F841
                                postscore.lower()[:3] == sign_str[1-s]
                                    ])

        if postscore.lower()[:3] == "pos" or \
                postscore.lower()[:6] == "nonneg" or \
                postscore.lower()[1:4] == "neg":
            bounds[0] = 0
        elif postscore.lower()[:8] == "negative" or \
                postscore.lower()[:11] == "nonpositive" or \
                postscore.lower()[1:9] == "positive":
            bounds[1] = 0


def dedim(
            dedim: Optional[bool] = None,
            ) -> bool:

    return set_default(val=dedim, default=False)


def num(
            num: Optional[int] = None,
            ) -> int:

    return set_default(val=num, default=1001)


def delta(
            num: Optional[float] = None,
            ) -> float:

    return set_default(val=num, default=1.0)
