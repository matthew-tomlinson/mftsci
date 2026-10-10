"""
Defaults module for scientific notation utilities.
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
from mftsci.utils.ps.defaults import *  # noqa: F401, F403


# =====================================
# DEFAULTS
# =====================================


def val(
        val: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=val, default=0)


def est(
        est: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return val(val=est)


def se(
        se: Optional[float] = None,
        ) -> float:

    return set_default(val=se, default=np.nan)


def fsf(
        fsf: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:
    """
    Position of first signficant figure
    """

    return set_default(val=fsf, default=0)


def pow10(
        pow10: Optional[int] = None,
        ) -> int:

    return set_default(val=pow10, default=0)


def dp(
        dp: Optional[int] = None,
        ) -> int:

    return set_default(val=dp, default=0)


def vec_unif(
        vec_unif: Optional[bool] = None,
        ) -> bool:

    return set_default(val=vec_unif, default=False)


def vec_unif_mode(
        vec_unif_mode: Optional[int] = None,
        ) -> int:

    return set_default(val=vec_unif_mode, default=0)


def min_sf(
        min_sf: Optional[int] = None,
        ) -> int:

    return set_default(val=min_sf, default=1)


def max_sf(
        max_sf: Optional[int] = None,
        ) -> int:

    return set_default(val=max_sf, default=10)


def max_consec_0(
        max_consec_0: Optional[int] = None,
        ) -> int:

    return set_default(val=max_consec_0, default=4)


def min_dp(
        min_dp: Optional[int] = None,
        ) -> int:

    return set_default(val=min_dp, default=0)


def extra_dp_se_base(
        extra_dp_se_base: Optional[float] = None,
        ) -> float:

    return set_default(val=extra_dp_se_base, default=2.0)


def scif_format(
        scif_format: Optional[str] = None,
        ) -> str:

    return set_default(val=scif_format, default="e")


def pltf(
        pltf: Optional[bool] = None,
        ) -> bool:

    return set_default(val=pltf, default=False)


def tex(
        tex: Optional[bool] = None,
        ) -> bool:

    return set_default(val=tex, default=False)


def mathmode(
        mathmode: Optional[bool] = None,
        ) -> bool:

    return set_default(val=mathmode, default=True)


def val_plus(
        val_plus: Optional[str] = None,
        ) -> str:

    return set_default(val=val_plus, default="")


def val_zero(
        val_zero: Optional[str] = None,
        ) -> str:

    return set_default(val=val_zero, default="")


def force_se(
        force_se: Optional[bool] = None,
        ) -> bool:

    return set_default(val=force_se, default=False)


def null_se_mode(
        null_se_mode: Optional[str] = None,
        ) -> str:

    return set_default(val=null_se_mode, default="rm")


def pow10_0_mode(
        pow10_0_mode: Optional[str] = None,
        ) -> str:

    return set_default(val=pow10_0_mode, default="rm")


def exp_plus(
        exp_plus: Optional[bool] = None,
        ) -> bool:

    return set_default(val=exp_plus, default=True)


def exp_zeros_min(
        exp_zeros_min: Optional[int] = None,
        ) -> int:

    return set_default(val=exp_zeros_min, default=2)


def brac_mode(
        pow10_0_mode: Optional[str] = None,
        ) -> str:

    return set_default(val=pow10_0_mode, default="rm")


def sign_inside_brac(
        sign_inside_brac: Optional[bool] = None,
        ) -> bool:

    return set_default(val=sign_inside_brac, default=False)


def app_dlrs(
        app_dlrs: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_dlrs, default=False)


def app_tex_num(
        app_tex_num: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_tex_num, default=False)
