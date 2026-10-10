"""
Defaults module for Kolmogorov--Smirnov test.
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
from mftsci.htest.confidence_interval.defaults import alpha, tail   # noqa: F401
from mftsci.htest.htests.defaults import N, set_res     # noqa: F401
from mftsci.htest.htests.dist_tests.defaults import app_PIT     # noqa: F401


# =====================================
# DEFAULTS
# =====================================


def N_0(
        N_0: Optional[int] = None,
        ) -> int:

    return set_default(val=N_0, default=0)


def a_lambda_T(
        a_lambda_T: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=a_lambda_T, default=1.0)


def t_0_T(
        a_lambda_T: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=a_lambda_T, default=0.0)


def point_proc(
        point_proc: Optional[bool] = None,
        ) -> bool:

    return set_default(val=point_proc, default=False)


def norm_point(
        norm_point: Optional[bool] = None,
        ) -> bool:

    return set_default(val=norm_point, default=True)


def loc(
        loc: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=loc, default=0.0)


def scale(
        scale: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=scale, default=1.0)


def cdf(
        cdf: Optional[Callable] = None,
        ) -> Callable:

    return set_default(val=alpha, default_call=scipy.stats.norm().cdf)


def method(
        method: Optional[str] = None,
        ) -> str:

    return set_default(val=method, default="auto")


def t_dedim(
        t_dedim: Optional[bool] = None,
        ) -> bool:

    return set_default(val=t_dedim, default=True)


def y_norm(
        y_norm: Optional[bool] = None,
        ) -> bool:

    return set_default(val=y_norm, default=False)


def y_scale(
        y_scale: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=y_scale, default=1.0)


def y_0(
        y_0: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=y_0, default=0.0)


def N_x(
        N_x: Optional[int] = None,
        ) -> int:

    return set_default(val=N_x, default=401)
