"""
Defaults module for point processes in continuous time.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import scipy

# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401
from mftsci.utils.bounds.defaults import num, delta     # noqa: F401, F403
from mftsci.point_proc.cont.core import gen_lambda_const as _gen_lambda_const


# =====================================
# DEFAULTS
# =====================================


def a_lambda(
        a_lambda: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=a_lambda, default=1.0)


def t(
        t: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=t, default=0.0)


def t_0(
        t_0: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=t_0, default=0.0)


def t_1(
        t_1: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=t_1, default=1.0)


def lambda_fn(
        lambda_fn: Optional[Callable] = None,
        ):

    return set_default(val=lambda_fn,
                       default_call=lambda: _gen_lambda_const(a_lambda=a_lambda()))


def mark_dist(
        mark_dist: Optional[scipy.stats._distn_infrastructure.rv_continuous_frozen] = None,     # noqa: E501
        ) -> scipy.stats._distn_infrastructure.rv_continuous_frozen:

    return set_default(val=mark_dist, default_call=lambda: scipy.stats.expon())


def ylog(
        ylog: Optional[bool] = None,
        ) -> bool:

    return set_default(val=ylog, default=False)


def mm_N(
            mm_N: Optional[int] = None,
            ):

    return set_default(val=mm_N, default=100)


def lambda_mm(
            lambda_mm: Optional[float] = None,
            ):

    return set_default(val=lambda_mm, default=1.0)


def t_dedim(
        t_dedim: Optional[bool] = None,
        ) -> bool:

    return set_default(val=t_dedim, default=True)


def resid(
        resid: Optional[bool] = None,
        ) -> bool:

    return set_default(val=resid, default=False)
