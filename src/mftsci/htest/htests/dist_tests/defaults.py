"""
Defaults module for `dist_tests` subpackage.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union

# External packages

# Internal packages
from mftsci.utils.meta import set_default


# =====================================
# DEFAULTS
# =====================================


def a_lambda(
        a_lambda: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=a_lambda, default=1.0)


def excl_inf(
        excl_inf: Optional[bool] = None,
        ) -> bool:

    return set_default(val=excl_inf, default=False)


def app_PIT(
         app_PIT: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_PIT, default=True)


def inv(
         inv: Optional[bool] = None,
        ) -> bool:

    return set_default(val=inv, default=False)


def inf_alpha(
         inf_alpha: Optional[float] = None,
        ) -> float:

    return set_default(val=inf_alpha, default=1E-8)
