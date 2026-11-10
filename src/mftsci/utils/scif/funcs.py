"""
Functions module for scientific notation utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils.meta import default_args_kwargs
from mftsci.utils.arr import fmap, overwrite_a_bool, np_round, round_str, lead_zeros
from mftsci.utils.ps import calc_brac, capp_dlrs_tex_num, capp_tex_hphantom, capp_tex_fn
from mftsci.utils.scif import defaults


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def clean_fsf(
                fsf: Optional[Union[np.ndarray, list, float, int,]] = None,
                ) -> Union[np.ndarray, int,]:
    """
    Position of first signficant figure
    """

    _default = defaults.fsf()
    fsf = defaults.fsf(fsf)

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
    val = defaults.val(val)
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
    vec_unif = defaults.vec_unif(vec_unif)

    if pow10 is None:
        pow10 = clean_fsf(fsf=calc_fsf(val=val))

    if vec_unif:
        return np.max(pow10)
    else:
        return pow10


def _calc_base_pow10_scalar(
                                val: Union[float, int,],
                                max_sf: Optional[int] = None,
                                ):

    max_sf = defaults.max_sf(max_sf)
    base_pow10 = f"{val:+.{max_sf-1}e}".split("e")
    if len(base_pow10) == 1:
        base_pow10.append("0")

    return base_pow10


def _calc_base_pow10_sf(
                        val: Union[float, int,],
                        max_sf: Optional[int] = None,
                        max_consec_0: Optional[int] = None,
                        ):

    res = {vs: None for vs in ["base", "pow10", "sf"]}

    max_sf = defaults.max_sf(max_sf)
    max_consec_0 = defaults.max_sf(max_consec_0)

    bp_split = fmap(val, lambda v: _calc_base_pow10_scalar(val=v, max_sf=max_sf))

    res["base"] = bp_split[..., 0]
    if max_consec_0 > 0:
        null_term = "0"*max_consec_0
        res["base"] = fmap(res["base"], lambda v: v[:3] + v[3:].split(null_term)[0])
    res["base"] = res["base"].astype(float)
    res["pow10"] = bp_split[..., 1].astype(int)

    res["sf"] = fmap(res["base"], lambda v: len(f"{v:+}".removesuffix("0")) - 2)

    for vs in res:
        if np.isscalar(val):
            res[vs] = res[vs][0]
        elif isinstance(val, pd.Series):
            res[vs] = pd.Series(res[vs], index=val.index)
        elif isinstance(val, pd.DataFrame):
            res[vs] = pd.DataFrame(res[vs], index=val.index, columns=val.columns)

    return (res[vs] for vs in res)


def calc_base_pow10_dp_sf(
                            val: Optional[Union[np.ndarray, float, int,]] = None,
                            pow10: Optional[Union[np.ndarray, int,]] = None,
                            max_sf: Optional[Union[np.ndarray, int,]] = None,
                            max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                            vec_unif: Optional[bool] = None,
                            est: Optional[Union[np.ndarray, float, int,]] = None,
                            ):

    vec_unif = defaults.vec_unif(vec_unif)
    if val is None:
        val = est
    val = defaults.val(val)
    base, _pow10, sf = _calc_base_pow10_sf(
                                            val, max_sf=max_sf,
                                            max_consec_0=max_consec_0,
                                            )

    if pow10 is None:
        if vec_unif:
            pow10 = int(np.max(_pow10))
        else:
            pow10 = _pow10

    pow_10_diff = _pow10 - pow10

    base *= 10.0**pow_10_diff

    dp = sf - (1+pow_10_diff)
    if np.isscalar(dp):
        if dp < 0:
            dp = 0
    else:
        dp[dp < 0] = 0

    return base, pow10, dp, sf


def calc_convert(
                    val: Optional[Union[np.ndarray, float, int,]] = None,
                    se: Optional[Union[np.ndarray, float, int,]] = None,
                    pow10: Optional[Union[np.ndarray, int,]] = None,
                    dp: Optional[Union[np.ndarray, int,]] = None,
                    min_dp: Optional[int] = None,
                    max_sf: Optional[Union[np.ndarray, int,]] = None,
                    max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                    vec_unif: Optional[bool] = None,
                    vec_unif_mode: Optional[int] = None,
                    est: Optional[Union[np.ndarray, float, int,]] = None,
                    extra_dp_se_base: Optional[float] = None,
                    ):

    vec_unif = defaults.vec_unif(vec_unif)
    vec_unif_mode = defaults.vec_unif_mode(vec_unif_mode)

    extra_dp_se_base = defaults.extra_dp_se_base(extra_dp_se_base)
    if pow10 is not None:
        spec_pow10 = True
    else:
        spec_pow10 = False

    if dp is not None:
        spec_dp = True
    else:
        spec_dp = False

    if min_dp is None:
        if spec_dp:
            min_dp = 0
        else:
            min_dp = defaults.min_dp(min_dp)

    if min_dp < 0:
        min_dp = 0

    base, pow10, _dp, sf = calc_base_pow10_dp_sf(
                            val=val,
                            pow10=pow10,
                            max_sf=max_sf,
                            max_consec_0=max_consec_0,
                            vec_unif=vec_unif,
                            est=est,
                            )

    if se is None:
        se = np.nan * base
    _se_base, se_pow10, se_dp, se_sf = calc_base_pow10_dp_sf(
                                                            val=se,
                                                            max_sf=max_sf,
                                                            max_consec_0=max_consec_0,
                                                            est=est,
                                                            )

    se_pow10_diff = se_pow10 - pow10

    se_base = _se_base*10.0**(se_pow10_diff)

    is_se = ~np.isnan(se_base)
    if dp is None:
        dp = _dp

        if np.isscalar(se_base):
            if is_se:
                dp = -se_pow10_diff
        else:
            dp[is_se] = -se_pow10_diff[is_se]

        if extra_dp_se_base > 1:
            extra_dp = _se_base < extra_dp_se_base
            if np.isscalar(extra_dp):
                if extra_dp:
                    dp += 1
            else:
                dp[extra_dp] += 1

    dp = overwrite_a_bool(
                    a=dp,
                    a1=min_dp,
                    a_bool=dp < min_dp,
                    )

    if vec_unif:
        vec_unif_mode = defaults.vec_unif_mode(vec_unif_mode)
        if vec_unif_mode == 0:
            dp = dp - se_pow10_diff
        else:
            if not np.isscalar(dp):
                if vec_unif_mode < 0:
                    f = np.min
                elif vec_unif_mode > 0:
                    f = np.max
                if np.isscalar(is_se) or not np.any(is_se):
                    dp = int(f(dp))
                else:
                    dp = int(f(dp[is_se]))

    if not spec_pow10:
        base_round = np_round(base, decimals=dp)
        is_10 = base_round == 10.0

        if np.any(is_10):
            if vec_unif:
                pow10 += 1
                base /= 10.0
                se_base /= 10.0

                if not spec_dp:
                    if np.isscalar(dp):
                        if vec_unif:
                            if vec_unif_mode >= 0:
                                dp += 1
                            else:
                                dp = 0
                        else:
                            dp = dp + is_10.astype(int)
                    else:
                        dp[is_10] += 1

            elif np.isscalar(is_10):
                pow10 += 1
                base /= 10.0
                se_base /= 10.0
                if not spec_dp:
                    dp += 1
            else:
                base[is_10] /= 10.0
                if np.isscalar(pow10):
                    pow10 = pow10 + is_10.astype(int)
                else:
                    pow10[is_10] += 1

                if np.isscalar(se_base):
                    se_base = se_base / 10.0**(is_10.astype(int))
                else:
                    se_base[is_10] /= 10.0

                if not spec_dp:
                    if np.isscalar(dp):
                        dp += 1
                    else:
                        dp[is_10] += 1

    return base, se_base, pow10, dp, is_se


def update_brac(
                brac_str: Optional[list] = None,
                brac_a1s: Optional[dict] = None,
                brac_a_bool: Optional[Union[np.ndarray, bool]] = None,
                brac_mode: Optional[str] = None,
                brac_mode_default: Optional[str] = None,
                brac_args: Optional[tuple] = None,
                brac_kwargs: Optional[dict] = None,
                ):

    if brac_str is None:
        brac_args, brac_kwargs = default_args_kwargs(
                                            brac_args, brac_kwargs,
                                            )
        brac_str = [*calc_brac(*brac_args, **brac_kwargs)]

    if brac_a_bool is None:
        brac_a_bool = False

    if np.any(brac_a_bool):

        if brac_a1s is None:
            brac_a1s = {"rm": np.repeat("", 2)}

        brac_mode_default = next(iter(brac_a1s.keys()))
        brac_mode = defaults.set_default(brac_mode, brac_mode_default)

        brac_a1 = brac_a1s.get(brac_mode, None)
        if brac_a1 is not None:
            for k in range(2):
                brac_str[k] = overwrite_a_bool(
                                            a=brac_str[k],
                                            a1=brac_a1[k],
                                            a_bool=brac_a_bool,
                                            )
    return brac_str


def calc_scif(
                    val: Optional[Union[np.ndarray, float, int,]] = None,
                    se: Optional[Union[np.ndarray, float, int,]] = None,
                    pow10: Optional[Union[np.ndarray, int,]] = None,
                    dp: Optional[Union[np.ndarray, int,]] = None,
                    min_dp: Optional[int] = None,
                    max_sf: Optional[Union[np.ndarray, int,]] = None,
                    max_consec_0: Optional[Union[np.ndarray, int,]] = None,
                    vec_unif: Optional[bool] = None,
                    vec_unif_mode: Optional[int] = None,
                    est: Optional[Union[np.ndarray, float, int,]] = None,
                    extra_dp_se_base: Optional[float] = None,
                    scif_format: Optional[str] = None,
                    pltf: Optional[bool] = None,
                    tex: Optional[bool] = None,
                    mathmode: Optional[bool] = None,
                    val_plus: Optional[str] = None,
                    val_zero: Optional[str] = None,
                    force_se: Optional[bool] = None,
                    se_pm: Optional[str] = None,
                    se_space: Optional[str] = None,
                    null_se_mode: Optional[str] = None,
                    vec_unif_mode_exp: Optional[int] = None,
                    times_symb: Optional[str] = None,
                    times_space: Optional[str] = None,
                    exp_supfun_ps: Optional[Union[np.ndarray]] = None,
                    exp_plus: Optional[bool] = None,
                    exp_zeros: Optional[Union[np.ndarray, int,]] = None,
                    exp_zeros_min: Optional[int] = None,
                    pow10_0_mode: Optional[str] = None,
                    brac_type: Optional[str] = None,
                    add_lr: Optional[bool] = None,
                    brac_mode: Optional[str] = None,
                    null_se_brac_mode: Optional[str] = None,
                    pow10_0_brac_mode: Optional[str] = None,
                    null_se_pow10_0_brac_mode: Optional[str] = None,
                    sign_inside_brac: Optional[bool] = None,
                    app_dlrs: Optional[bool] = None,
                    app_tex_num: Optional[bool] = None,
                    ):

    scif_format = defaults.scif_format(scif_format)

    pltf = defaults.pltf(pltf)
    if pltf:
        tex = True
        mathmode = True

    if tex is None:
        if mathmode is True:
            tex = True
        else:
            tex = defaults.tex(tex)
    mathmode = defaults.mathmode(mathmode)

    if isinstance(val_plus, bool):
        if val_plus:
            val_plus = "+"
        else:
            val_plus = ""
    elif val_plus == "phantom":
        val_plus = capp_tex_hphantom("+")
    val_plus = defaults.val_plus(val_plus)

    if se_pm is None:
        if tex:
            if mathmode:
                se_pm = "\\pm"
            else:
                se_pm = "\\textpm"
        else:
            se_pm = "±"

    if isinstance(val_zero, bool):
        if val_zero:
            val_zero = se_pm
        else:
            val_zero = ""
    elif val_zero == "phantom":
        val_zero = capp_tex_hphantom(se_pm)
    val_zero = defaults.val_zero(val_zero)

    force_se = defaults.force_se(force_se)

    exp_plus = defaults.exp_plus(exp_plus)
    if exp_zeros_min is None:
        if exp_zeros is None:
            exp_zeros_min = defaults.exp_zeros_min(exp_zeros_min)
        else:
            exp_zeros_min = 0

    brac_mode = defaults.brac_mode(brac_mode)
    sign_inside_brac = defaults.sign_inside_brac(sign_inside_brac)

    app_dlrs = defaults.app_dlrs(app_dlrs)
    app_tex_num = defaults.app_tex_num(app_tex_num)

    if tex:
        tex_bracs = np.array(["{", "}"])
    else:
        tex_bracs = np.array(["", ""])

    base, se_base, pow10, dp, is_se = calc_convert(
                                        val=val,
                                        se=se,
                                        pow10=pow10,
                                        dp=dp,
                                        min_dp=min_dp,
                                        max_sf=max_sf,
                                        max_consec_0=max_consec_0,
                                        vec_unif=vec_unif,
                                        vec_unif_mode=vec_unif_mode,
                                        est=est,
                                        extra_dp_se_base=extra_dp_se_base
                                        )
    pow10_0 = pow10 == 0

    signs_srs = pd.Series({-2: "", -1: "-", 0: val_zero, 1: val_plus})
    sbase = np.sign(base)

    if isinstance(sbase, pd.Series) or isinstance(sbase, pd.DataFrame):
        sign_str = sbase.fillna(-2).astype(int).astype(str)\
                        .replace("-1", signs_srs[-1]).replace("1", signs_srs[1])
    else:
        if np.isscalar(sbase):
            if np.isnan(sbase):
                sbase = -2
            sign_str = signs_srs.loc[int(sbase)]
        else:
            sbase[np.isnan(sbase)] = -2
            sign_str = signs_srs.loc[sbase.astype(int)].values

    est_val_base_str = round_str(val=base, decimals=dp, rm_sign=True)
    if force_se:
        se_base = overwrite_a_bool(
                                    a=se_base,
                                    a1=0.0,
                                    a_bool=~is_se,
                                    )
        is_se = overwrite_a_bool(
                            a=is_se,
                            a1=True,
                            a_bool=~is_se,
                            )
    se_val_base_str = round_str(val=se_base, decimals=dp, rm_sign=True)
    null_se = ~is_se

    if se_space is None:
        if tex:
            if mathmode:
                se_space = " "
            else:
                se_space = ""
        else:
            se_space = ""

    se_base_str = se_space + se_pm + se_space + se_val_base_str
    ow_null_se = False
    if np.any(null_se):
        null_se_mode = defaults.null_se_mode(null_se_mode)
        if null_se_mode == "rm":
            ow_null_se = True
            se_base_str_a1 = ""
        elif null_se_mode[:7].lower() == "phantom":
            ow_null_se = True
            se_base_str_phantom = round_str(
                                            val=np.zeros(np.shape(se_base)),
                                            decimals=dp, rm_sign=True,
                                            )
            se_base_str_a1 = capp_tex_hphantom(
                                                se_space + se_pm +
                                                se_space + se_base_str_phantom
                                                )
            if isinstance(se_base_str, pd.DataFrame):
                se_base_str_a1 = pd.DataFrame(se_base_str_a1)

        if ow_null_se:
            se_base_str = overwrite_a_bool(
                                                a=se_base_str,
                                                a1=se_base_str_a1,
                                                a_bool=null_se,
                                                )
    base_str = est_val_base_str + se_base_str

    if scif_format == "10":
        exp_base = "10"
        if times_symb is None:
            if tex:
                if mathmode:
                    times_symb = "\\cdot"
                    # times_symb = "\\times"
                else:
                    times_symb = "$\\cdot$"
                    # times_symb = "\\texttimes"
            else:
                times_symb = "×"

        if times_space is None:
            if tex:
                if mathmode:
                    times_space = " "
                else:
                    times_space = ""
            else:
                times_space = ""

        if exp_supfun_ps is None:
            if tex:
                if mathmode:
                    exp_supfun_ps = np.array(["^{", "}"])
                else:
                    exp_supfun_ps = np.array(["\\textsuperscript{", "}"])
            else:
                exp_supfun_ps = np.array(["^{", "}"])
    else:
        if scif_format.lower() == "e":
            s = scif_format
        else:
            s = "e"
        app_tex_fn = tex * mathmode
        exp_base = capp_tex_fn(
                                s=s, tex_fn="mathrm", app_null=False,
                                app_tex_fn=app_tex_fn,
                                )
        times_symb = ""
        times_space = ""
        exp_supfun_ps = tex_bracs

    times_str = times_space + times_symb + times_space

    if exp_zeros is None:
        exp_zeros = calc_pow10(pow10)
        if vec_unif_mode_exp is True:
            exp_zeros = int(np.max(exp_zeros))

    if exp_zeros_min > 0:
        exp_zeros = overwrite_a_bool(
                                    a=exp_zeros,
                                    a1=exp_zeros_min,
                                    a_bool=exp_zeros < exp_zeros_min,
                                    )

    exp_num_str = lead_zeros(val=pow10, zeros=exp_zeros, val_plus=exp_plus)
    exp_str = times_str + exp_base + exp_supfun_ps[..., 0] + exp_num_str + exp_supfun_ps[..., 1]    # noqa: E501

    ow_pow10_0 = False
    if np.any(pow10_0):
        pow10_0_mode = defaults.null_se_mode(pow10_0_mode)
        if pow10_0_mode == "rm":
            ow_pow10_0 = True
            exp_str_a1 = ""
        elif pow10_0_mode == "phantom":
            ow_pow10_0 = True
            exp_str_a1 = capp_tex_hphantom(exp_str)

        if ow_pow10_0:
            exp_str = overwrite_a_bool(
                                        a=exp_str,
                                        a1=exp_str_a1,
                                        a_bool=pow10_0,
                                        )

    val_se_brac = calc_brac(brac_type=brac_type, add_lr=add_lr)
    brac_a1s = {
                    "rm": np.repeat("", 2),
                    "phantom": calc_brac(
                                            brac_type=brac_type,
                                            add_lr=add_lr,
                                            phantom=True,
                                            )
                    }
    brac_str = [*val_se_brac]

    brac_str = update_brac(
                            brac_str=brac_str,
                            brac_a1s=brac_a1s,
                            brac_a_bool=null_se,
                            brac_mode=null_se_brac_mode,
                            brac_mode_default=brac_mode,
                            )
    brac_str = update_brac(
                        brac_str=brac_str,
                        brac_a1s=brac_a1s,
                        brac_a_bool=pow10_0,
                        brac_mode=pow10_0_brac_mode,
                        brac_mode_default=brac_mode,
                        )
    brac_str = update_brac(
                        brac_str=brac_str,
                        brac_a1s=brac_a1s,
                        brac_a_bool=null_se * pow10_0,
                        brac_mode=null_se_pow10_0_brac_mode,
                        brac_mode_default=brac_mode,
                        )

    if sign_inside_brac:
        scif_out = brac_str[0] + sign_str + base_str + brac_str[1] + exp_str
    else:
        scif_out = sign_str + brac_str[0] + base_str + brac_str[1] + exp_str

    return capp_dlrs_tex_num(scif_out, app_tex_num=app_tex_num, app_dlrs=app_dlrs)
