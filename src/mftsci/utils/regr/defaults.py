"""
Defaults module for regr utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional


# External packages


# Internal packages
from mftsci.utils.meta.core import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401
from mftsci.utils.meta import default_kwargs as kwargs  # noqa: F401

# =====================================
# DEFAULTS
# =====================================


def app_bias(
        app_bias: Optional[bool] = None,
        ) -> bool:

    return set_default(app_bias, True)


def check_bias(
        check_bias: Optional[bool] = None,
        ) -> bool:

    return set_default(check_bias, True)


def bias_name(
        bias_name: Optional[str] = None,
        ) -> str:

    return set_default(bias_name, "_bias")


def safe_alpha(
        safe_alpha: Optional[float] = None,
        ) -> float:

    return set_default(val=safe_alpha, default=0.01)


def N_samp(
        N_samp: Optional[int] = None,
        ) -> int:

    return set_default(val=N_samp, default=1000)


def N_cline(
        N_cline: Optional[int] = None,
        ) -> int:

    return set_default(val=N_cline, default=0)


def stack_H(
        stack_H: Optional[bool] = None,
        ) -> bool:

    return set_default(val=stack_H, default=False)


def app_merge(
        app_merge: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_merge, default=True)


def expand_X(
        expand_X: Optional[bool] = None,
        ) -> bool:

    return set_default(val=expand_X, default=False)


def match_head(
        match_head: Optional[bool] = None,
        ) -> bool:

    return set_default(val=match_head, default=False)
