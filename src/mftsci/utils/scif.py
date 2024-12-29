"""
Module for scientific notation utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Tuple

# External packages
import numpy as np

# Internal packages
from mftsci.utils.str import calc_brac, capp_dlrs_tex_num
from mftsci.utils.dict import filter_keys


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def clean_fsf(
                fsf: Optional[Union[np.ndarray, list, float, int,]] = None,
                ) -> Union[np.ndarray, int,]:

    _default = 0

    if fsf is None:
        fsf = _default

    edge = np.logical_or(np.isnan(fsf), np.isinf(fsf))
    if np.isscalar(fsf):
        if edge:
            fsf = _default
        fsf = np.int_(fsf)
    else:
        fsf[edge] = _default
        fsf = np.array(fsf, dtype=int)

    return fsf


def calc_fsf(
                val: Optional[Union[np.ndarray, float, int,]] = None,
                ) -> Union[np.ndarray, float,]:
    val = default_val(val)
    with np.errstate(divide='ignore'):
        fsf = np.floor(np.log(np.abs(val)) / np.log(10))
    return fsf


def calc_pow10(
                val: Optional[Union[np.ndarray, float, int,]] = None,
                pow10: Optional[Union[np.ndarray, int,]] = None,
                vec_unif: Optional[bool] = None,
                ) -> Union[np.ndarray, int,]:
    """
    Return scientific float exponent
    """
    if vec_unif is None:
        vec_unif = False

    if pow10 is None:
        pow10 = clean_fsf(fsf=calc_fsf(val=val))

    if vec_unif:
        return np.max(pow10)
    else:
        return pow10


def _calc_base_scalar(
                        val: Optional[Union[float, int,]] = None,
                        max_sf: Optional[int] = None,
                        max_consec_0: Optional[int] = None,
                        ret_str: Optional[bool] = None,
                        ) -> Union[float, int,]:
    """
    Return scientific float base of a scalar value
    """

    if max_sf is None:
        max_sf = 10

    if max_consec_0 is None:
        max_consec_0 = 4

    if ret_str is None:
        ret_str = False

    val = default_val(val=val)

    if np.isnan(val):
        s = "nan"
    else:
        s = f"{val:+.{max_sf-1}e}"
        s = s[:s.find("e")]

    if max_consec_0 > 0:
        null_term = "0"*max_consec_0
        where_null = s[3:].find(null_term)
        if where_null >= 0:
            s = s[:3+where_null]

    if ret_str:
        return s
    else:
        return float(s)


_calc_base_vector = np.vectorize(_calc_base_scalar)


def calc_base(
                val: Optional[Union[np.ndarray, float, int,]] = None,
                pow10: Optional[Union[np.ndarray, int,]] = None,
                max_sf: Optional[Union[np.ndarray, int,]] = None,
                max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                vec_unif: Optional[bool] = None,
                ) -> Union[np.ndarray, float, int,]:
    """
    Return scientific float base
    """
    if vec_unif is None:
        vec_unif = False

    val = default_val(val=val)
    pow10 = calc_pow10(val=val, pow10=pow10, vec_unif=vec_unif)
    base = _calc_base_vector(val=val, max_sf=max_sf, max_consec_0=max_consec_0)

    if vec_unif:
        _pow10 = calc_pow10(val=val, vec_unif=False)
        base *= 10.0**(_pow10 - pow10)

    return base


def _calc_sf_scalar(
                    val: Optional[Union[float, int,]] = None,
                    max_sf: Optional[int] = None,
                    max_consec_0: Optional[int] = None,
                    ) -> int:
    """
    Return number of significant figures of a scalar value
    """
    s = _calc_base_scalar(val=val, max_sf=max_sf, max_consec_0=max_consec_0,
                          ret_str=True)
    return len(s) - 2


_calc_sf_vector = np.vectorize(_calc_sf_scalar)


def _calc_sf(
                val: Optional[Union[np.ndarray, float, int,]] = None,
                max_sf: Optional[Union[np.ndarray, int,]] = None,
                max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                ) -> Union[np.ndarray, int,]:
    """
    Return number of significant figures
    """

    return _calc_sf_vector(**locals())


def _calc_dp(
                val: Optional[Union[np.ndarray, float, int,]] = None,
                max_sf: Optional[Union[np.ndarray, int,]] = None,
                max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                pow10: Optional[Union[np.ndarray, int,]] = None,
                vec_unif: Optional[bool] = None,
                scif: Optional[bool] = None,
                ) -> Union[np.ndarray, int,]:
    """
    Return number of decimal places
    """

    if scif is None:
        if pow10 is not None or vec_unif is not None:
            scif = True
        else:
            scif = False

    if vec_unif is None:
        vec_unif = False

    sf = _calc_sf(**filter_keys(locals(), keys_rm=["pow10", "vec_unif", "scif"]))

    _pow10 = calc_pow10(val=val, vec_unif=False)
    if scif:
        pow10_0 = calc_pow10(val=val, pow10=pow10, vec_unif=vec_unif)
    else:
        pow10_0 = 0

    return sf - (1 + (_pow10 - pow10_0))


def force_min_sf(
                    sf: Optional[Union[np.ndarray, int,]] = None,
                    min_sf: Optional[Union[np.ndarray, int,]] = None,
                    vec_unif: Optional[bool] = None,
                    vec_unif_mode: Optional[int] = None,
                    ) -> Union[np.ndarray, int,]:

    if vec_unif is None:
        vec_unif = False

    if min_sf is None:
        min_sf = 1

    if sf is None:
        sf = min_sf

    if vec_unif:
        if vec_unif_mode is None:
            vec_unif_mode = 0

        if vec_unif_mode > 0:
            sf = np.max(sf)
        elif vec_unif_mode < 0:
            sf = np.min(sf)

    edge = sf < min_sf
    if np.isscalar(sf):
        if np.isscalar(min_sf):
            if edge:
                sf = min_sf
        else:
            if np.hstack(edge)[0]:
                sf = np.hstack(min_sf)[0]
        sf = np.int_(sf)
    else:
        if np.isscalar(min_sf):
            sf[edge] = min_sf
        else:
            sf[edge] = min_sf[edge]

        sf = np.array(sf, dtype=int)

    return sf


def force_min_dp(
                    dp: Optional[Union[np.ndarray, int,]] = None,
                    min_dp: Optional[Union[np.ndarray, int,]] = None,
                    vec_unif: Optional[bool] = None,
                    vec_unif_mode: Optional[int] = None,
                    ) -> Union[np.ndarray, int,]:

    if min_dp is None:
        min_dp = 0

    return force_min_sf(sf=dp, min_sf=min_dp, vec_unif=vec_unif,
                        vec_unif_mode=vec_unif_mode)


def calc_dp(
            dp: Optional[Union[np.ndarray, int,]] = None,
            min_dp: Optional[Union[np.ndarray, int,]] = None,
            val: Optional[Union[np.ndarray, float, int,]] = None,
            se: Optional[Union[np.ndarray, float, int,]] = None,
            max_sf: Optional[Union[np.ndarray, int,]] = None,
            max_consec_0: Optional[Union[np.ndarray, int,]] = None,
            pow10: Optional[Union[np.ndarray, int,]] = None,
            pow10_se: Optional[Union[np.ndarray, int,]] = None,
            vec_unif: Optional[bool] = None,
            vec_unif_mode: Optional[int] = None,
            ) -> Union[np.ndarray, int,]:

    if vec_unif is None:
        vec_unif = False

    if dp is None:
        pow10 = calc_pow10(val=val, vec_unif=vec_unif, pow10=pow10)
        dp = _calc_dp(val=val, max_sf=max_sf, max_consec_0=max_consec_0, pow10=pow10)
        if se is not None:
            _dp = -(calc_pow10(val=se, pow10=pow10_se) - pow10)
            _dp += calc_base(val=se) < 1.5

            if np.isscalar(se):
                if not np.isnan(se):
                    dp = _dp
            else:
                if np.isscalar(dp):
                    dp = np.full(np.shape(se), dp)
                dp[~np.isnan(se)] = _dp[~np.isnan(se)]

    return force_min_dp(dp=dp, min_dp=min_dp, vec_unif=vec_unif,
                        vec_unif_mode=vec_unif_mode)


def calc_convert(
                    val: Optional[Union[np.ndarray, float, int,]] = None,
                    se: Optional[Union[np.ndarray, float, int,]] = None,
                    pow10: Optional[Union[np.ndarray, int,]] = None,
                    dp: Optional[Union[np.ndarray, int,]] = None,
                    max_sf: Optional[Union[np.ndarray, int,]] = None,
                    max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                    vec_unif: Optional[bool] = None,
                    vec_unif_mode: Optional[int] = None,
                    min_dp: Optional[Union[np.ndarray, int,]] = None,
                    ) -> Tuple[
                                Union[np.ndarray, float, int,],
                                Union[np.ndarray, float, int,],
                                Union[np.ndarray, int,],
                                Union[np.ndarray, int,],
                                ]:

    if vec_unif is None:
        vec_unif = False

    pow10 = calc_pow10(val=val, pow10=pow10, vec_unif=vec_unif)
    base_val = calc_base(val=val, pow10=pow10, max_sf=max_sf, max_consec_0=max_consec_0,
                         vec_unif=vec_unif)

    if se is not None:
        base_se = calc_base(val=se, pow10=pow10, max_sf=max_sf,
                            max_consec_0=max_consec_0, vec_unif=False)
        _pow10_se = calc_pow10(val=se, vec_unif=False)
        pow10_se_rel = _pow10_se - pow10
        base_se *= 10.0**(pow10_se_rel)
    else:
        base_se = np.full(np.shape(base_val), np.nan)
        _pow10_se = None
        pow10_se_rel = None

    if False:
        dp = calc_dp(
                        dp=dp, min_dp=min_dp,
                        val=val, se=se, vec_unif=vec_unif, vec_unif_mode=vec_unif_mode,
                        pow10=pow10, pow10_se=_pow10_se,
                        )
    else:

        dp = calc_dp(
                        dp=dp, min_dp=min_dp,
                        val=base_val, se=base_se, vec_unif=vec_unif,
                        vec_unif_mode=vec_unif_mode,
                        pow10=0, pow10_se=pow10_se_rel,
                        )

    return base_val, base_se, pow10, dp


def _calc_scif_default_meta_scalar(
                                    plot: Optional[bool] = None,
                                    mathmode: Optional[bool] = None,
                                    ) -> Tuple[bool, bool]:

    if plot is None:
        plot = False

    if mathmode is None:
        mathmode = True

    return plot, mathmode


def _calc_scif_default_convert_scalar(
                                        base_val: Optional[Union[float, int,]] = None,
                                        base_se: Optional[Union[float, int,]] = None,
                                        pow10: Optional[int] = None,
                                        dp: Optional[int] = None,
                                        ) -> Tuple[
                                                    Union[float, int,],
                                                    Union[float, int,],
                                                    int,
                                                    int,
                                                    ]:

    if base_val is None:
        base_val = 0
    if base_se is None:
        base_se = np.nan
    if pow10 is None:
        pow10 = 0
    if dp is None:
        dp = 0

    return base_val, base_se, pow10, dp


def _calc_scif_str_abs_val_scalar(
                                    base_val: Optional[Union[float, int,]] = None,
                                    base_se: Optional[Union[float, int,]] = None,
                                    pow10: Optional[int] = None,
                                    dp: Optional[int] = None,
                                    plot: Optional[bool] = None,
                                    mathmode: Optional[bool] = None,
                                    ) -> str:

    plot, mathmode = _calc_scif_default_meta_scalar(plot=plot, mathmode=mathmode)
    base_val, base_se, pow10, dp = _calc_scif_default_convert_scalar(base_val=base_val,
                                                                     base_se=base_se,
                                                                     pow10=pow10,
                                                                     dp=dp)

    str_abs_val = f"{np.abs(base_val):.{dp}f}"

    return str_abs_val


def _calc_scif_str_val_sign_scalar(
                                    base_val: Union[float, int,],
                                    base_se: Union[float, int,],
                                    pow10: int,
                                    dp: int,
                                    plot: Optional[bool] = None,
                                    mathmode: Optional[bool] = None,
                                    val_plus: Optional[bool] = None,
                                    phantom_sign: Optional[bool] = None,
                                    phantom_sign_str: Optional[str] = None,
                                    ) -> str:

    plot, mathmode = _calc_scif_default_meta_scalar(plot=plot, mathmode=mathmode)
    base_val, base_se, pow10, dp = _calc_scif_default_convert_scalar(base_val=base_val,
                                                                     base_se=base_se,
                                                                     pow10=pow10,
                                                                     dp=dp)

    if val_plus is None:
        val_plus = False
    if val_plus:
        phantom_sign = False

    if phantom_sign is None:
        phantom_sign = False

    if phantom_sign_str is None:
        if plot:
            phantom_sign_str = "\\:\\:"
        else:
            phantom_sign_str = f"\\phantom{{+}}"    # noqa: F541

    if base_val < 0:
        val_sign = "-"
    else:
        if val_plus:
            val_sign = "+"
        elif phantom_sign:
            val_sign = phantom_sign_str
        else:
            val_sign = ""

    return val_sign


def _calc_scif_show_se_scalar(
                                base_val: Optional[Union[float, int,]] = None,
                                base_se: Optional[Union[float, int,]] = None,
                                pow10: Optional[int] = None,
                                dp: Optional[int] = None,
                                force_se: Optional[bool] = None,
                                ) -> bool:

    if force_se is None:
        force_se = False

    show_se = True
    if base_se is None:
        show_se = False
    elif np.isnan(base_se):
        if not force_se or np.isnan(base_val):
            show_se = False

    return show_se


def _calc_scif_str_se_scalar(
                                base_val: Optional[Union[float, int,]] = None,
                                base_se: Optional[Union[float, int,]] = None,
                                pow10: Optional[int] = None,
                                dp: Optional[int] = None,
                                plot: Optional[bool] = None,
                                mathmode: Optional[bool] = None,
                                force_se: Optional[bool] = None,
                                se_space: Optional[str] = None,
                                ) -> str:

    plot, mathmode = _calc_scif_default_meta_scalar(plot=plot, mathmode=mathmode)
    base_val, base_se, pow10, dp = _calc_scif_default_convert_scalar(base_val=base_val,
                                                                     base_se=base_se,
                                                                     pow10=pow10,
                                                                     dp=dp)

    show_se = _calc_scif_show_se_scalar(base_val=base_val, base_se=base_se,
                                        force_se=force_se)

    if not show_se:
        str_se = ""
    else:
        if se_space is None:
            if mathmode:
                se_space = " "
            else:
                se_space = ""
        if se_space == "":
            post_se_pm = f"{{}}"    # noqa: F541
        else:
            post_se_pm = ""

        if mathmode:
            se_pm = "\\pm"
        else:
            se_pm = "\\textpm"

        str_se = f"{se_space}{se_pm}{post_se_pm}{se_space}{base_se:.{dp}f}"

    return str_se


def _calc_scif_str_exp_scalar(
                                base_val: Optional[Union[float, int,]] = None,
                                base_se: Optional[Union[float, int,]] = None,
                                pow10: Optional[int] = None,
                                dp: Optional[int] = None,
                                plot: Optional[bool] = None,
                                mathmode: Optional[bool] = None,
                                scif_format: Optional[str] = None,
                                exp_plus: Optional[bool] = None,
                                exp_zeros: Optional[int] = None,
                                times_symb: Optional[str] = None,
                                ) -> str:

    plot, mathmode = _calc_scif_default_meta_scalar(plot=plot, mathmode=mathmode)
    base_val, base_se, pow10, dp = _calc_scif_default_convert_scalar(base_val=base_val,
                                                                     base_se=base_se,
                                                                     pow10=pow10,
                                                                     dp=dp)

    if scif_format == "":
        str_exp = ""
    else:
        if exp_plus is None:
            exp_plus = True

        if exp_zeros is None:
            if np.isin(scif_format, ["e", "E"]):
                exp_zeros = 2
            else:
                exp_zeros = 1

        if pow10 == 0:
            exp_sign = ""
            str_exp = ""
        else:
            if pow10 < 0:
                exp_sign = "-"
            else:
                if exp_plus:
                    exp_sign = "+"
                else:
                    exp_sign = ""

            abs_pow10 = np.abs(pow10)
            if np.isin(scif_format, ["e", "E"]):
                if mathmode:
                    str_exp = f"\\mathrm{{{scif_format}{exp_sign}}}{{{abs_pow10:0{exp_zeros}d}}}"   # noqa: E501
                else:
                    str_exp = f"{scif_format}{exp_sign}{{{abs_pow10:0{exp_zeros}d}}}"
            else:
                if times_symb is None:
                    times_symb = "\\times"

                str_exp = f" {times_symb} 10^{{{exp_sign}{abs_pow10:0{exp_zeros}d}}}"

    return str_exp


def _calc_scif_str_brac_scalar(
                                base_val: Optional[Union[float, int,]] = None,
                                base_se: Optional[Union[float, int,]] = None,
                                pow10: Optional[int] = None,
                                dp: Optional[int] = None,
                                plot: Optional[bool] = None,
                                mathmode: Optional[bool] = None,
                                force_se: Optional[bool] = None,
                                phantom_brac: Optional[bool] = None,
                                brac_type: Optional[str] = None,
                                add_lr: Optional[bool] = None,
                                ) -> np.ndarray:

    plot, mathmode = _calc_scif_default_meta_scalar(plot=plot, mathmode=mathmode)
    base_val, base_se, pow10, dp = _calc_scif_default_convert_scalar(base_val=base_val,
                                                                     base_se=base_se,
                                                                     pow10=pow10,
                                                                     dp=dp)

    show_se = _calc_scif_show_se_scalar(base_val=base_val, base_se=base_se,
                                        force_se=force_se)

    if phantom_brac is None:
        phantom_brac = False

    if add_lr is None:
        if mathmode:
            add_lr = True
        else:
            add_lr = False

    if brac_type is None:
        brac_type = "()"

    if pow10 == 0 or not show_se:
        if phantom_brac:
            str_brac = calc_brac(brac_type=brac_type, add_lr=add_lr,
                                 phantom=phantom_brac, plot=plot)
        else:
            str_brac = np.repeat("", 2)
    else:
        str_brac = calc_brac(brac_type=brac_type, add_lr=add_lr)

    return str_brac


def _build_scif_str_scif_scalar(
                                str_abs_val: str,
                                str_val_sign: str,
                                str_se: str,
                                str_exp: str,
                                str_brac: Union[np.ndarray, list],
                                mathmode: bool,
                                phantom_sign: bool,
                                sign_inside_brac: Optional[bool] = None,
                                app_dlrs: Optional[bool] = None,
                                app_tex_num: Optional[bool] = None,
                                ) -> str:

    if sign_inside_brac is None:
        if phantom_sign:
            sign_inside_brac = False
        else:
            sign_inside_brac = False

    if app_dlrs is None:
        app_dlrs = False

    if mathmode:
        app_tex_num = False
    if app_tex_num is None:
        app_tex_num = False

    if sign_inside_brac:
        str_scif = f"{str_brac[0]}{str_val_sign}{str_abs_val}{str_se}{str_brac[1]}{str_exp}"    # noqa: E501
    else:
        str_scif = f"{str_val_sign}{str_brac[0]}{str_abs_val}{str_se}{str_brac[1]}{str_exp}"    # noqa: E501

    return capp_dlrs_tex_num(s=str_scif,  app_dlrs=app_dlrs, app_tex_num=app_tex_num)


def _calc_scif_scalar(
                        base_val: Union[float, int,],
                        base_se: Union[float, int,],
                        pow10: int,
                        dp: int,
                        plot: Optional[bool] = None,
                        mathmode: Optional[bool] = None,
                        scif_format: Optional[str] = None,
                        val_plus: Optional[bool] = None,
                        phantom_sign: Optional[bool] = None,
                        phantom_sign_str: Optional[str] = None,
                        force_se: Optional[bool] = None,
                        se_space: Optional[str] = None,
                        exp_plus: Optional[bool] = None,
                        exp_zeros: Optional[int] = None,
                        times_symb: Optional[str] = None,
                        phantom_brac: Optional[bool] = None,
                        brac_type: Optional[str] = None,
                        add_lr: Optional[bool] = None,
                        sign_inside_brac: Optional[bool] = None,
                        app_dlrs: Optional[bool] = None,
                        app_tex_num: Optional[bool] = None,
                        ) -> str:

    convert = _calc_scif_default_convert_scalar(base_val=base_val, base_se=base_se,
                                                pow10=pow10, dp=dp)
    meta = _calc_scif_default_meta_scalar(plot=plot, mathmode=mathmode)

    cm = (*convert, *meta)

    str_abs_val = _calc_scif_str_abs_val_scalar(
                            *cm,
                            )
    str_val_sign = _calc_scif_str_val_sign_scalar(
                            *cm,
                            val_plus=val_plus, phantom_sign=phantom_sign,
                            phantom_sign_str=phantom_sign_str,
                            )
    str_se = _calc_scif_str_se_scalar(
                            *cm,
                            force_se=force_se, se_space=se_space,
                            )
    str_exp = _calc_scif_str_exp_scalar(
                            *cm,
                            scif_format=scif_format,
                            exp_plus=exp_plus, exp_zeros=exp_zeros,
                            times_symb=times_symb,
                            )
    str_brac = _calc_scif_str_brac_scalar(
                            *cm,
                            force_se=force_se,
                            phantom_brac=phantom_brac, brac_type=brac_type,
                            add_lr=add_lr,
                            )

    return _build_scif_str_scif_scalar(
                                            str_abs_val=str_abs_val,
                                            str_val_sign=str_val_sign,
                                            str_se=str_se,
                                            str_exp=str_exp,
                                            str_brac=str_brac,
                                            mathmode=mathmode,
                                            phantom_sign=phantom_sign,
                                            sign_inside_brac=sign_inside_brac,
                                            app_dlrs=app_dlrs,
                                            app_tex_num=app_tex_num,
                                        )


_calc_scif_str_abs_val_vector = np.vectorize(_calc_scif_str_abs_val_scalar)
_calc_scif_str_se_vector = np.vectorize(_calc_scif_str_se_scalar)
_calc_scif_str_exp_vector = np.vectorize(_calc_scif_str_exp_scalar)
_calc_scif_str_brac_vector = np.vectorize(_calc_scif_str_brac_scalar)
_build_scif_str_scif_vector = np.vectorize(_build_scif_str_scif_scalar)
_calc_scif_vector = np.vectorize(_calc_scif_scalar)


def calc_scif(
                val: Optional[Union[np.ndarray, float, int,]] = None,
                se: Optional[Union[np.ndarray, float, int,]] = None,
                pow10: Optional[Union[np.ndarray, int,]] = None,
                dp: Optional[Union[np.ndarray, int,]] = None,
                max_sf: Optional[Union[np.ndarray, int,]] = None,
                max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                vec_unif: Optional[Union[np.ndarray, bool,]] = None,
                vec_unif_mode: Optional[Union[np.ndarray, int,]] = None,
                scif_format: Optional[Union[np.ndarray, str,]] = None,
                plot: Optional[Union[np.ndarray, bool,]] = None,
                mathmode: Optional[Union[np.ndarray, bool,]] = None,
                val_plus: Optional[Union[np.ndarray, bool,]] = None,
                phantom_sign: Optional[Union[np.ndarray, bool,]] = None,
                phantom_sign_str: Optional[Union[np.ndarray, str,]] = None,
                force_se: Optional[Union[np.ndarray, bool,]] = None,
                se_space: Optional[Union[np.ndarray, str,]] = None,
                exp_plus: Optional[Union[np.ndarray, bool,]] = None,
                exp_zeros: Optional[Union[np.ndarray, int,]] = None,
                times_symb: Optional[Union[np.ndarray, str,]] = None,
                phantom_brac: Optional[Union[np.ndarray, bool,]] = None,
                brac_type: Optional[Union[np.ndarray, str,]] = None,
                add_lr: Optional[Union[np.ndarray, bool,]] = None,
                sign_inside_brac: Optional[Union[np.ndarray, bool,]] = None,
                app_dlrs: Optional[Union[np.ndarray, bool,]] = None,
                app_tex_num: Optional[Union[np.ndarray, bool,]] = None,
                ) -> Union[np.ndarray, str]:

    _locals = locals()

    convert_args = ["val", "se", "pow10", "dp",
                    "max_sf", "max_consec_0", "vec_unif", "vec_unif_mode", "min_dp"]

    if np.size(val) == 0:
        return np.full(np.shape(val), "")
    else:
        convert = calc_convert(**filter_keys(d=_locals, keys_in=convert_args))

        return _calc_scif_vector(*convert, **filter_keys(d=_locals,
                                                         keys_rm=convert_args))

# Defaults
# -------------------------------------


def default_val(
                val: Optional[Union[float, int,]] = None,
                ) -> Union[float, int,]:

    if val is None:
        val = 0

    return val
