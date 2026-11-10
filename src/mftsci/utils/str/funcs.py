"""
Functions module for string utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union
import hashlib

# External packages
import numpy as np

# Internal packages
from mftsci.utils.meta import vectorize_pd
from mftsci.utils.arr import convert as _convert, calc_shape, fattr, \
        reshape_iter_pre, pad as _pad, fmap as _fmap, \
        calc_val_1_0_N as _calc_val_1_0_N   # noqa: F401
from mftsci.utils.misc import calc_val_1_0 as _calc_val_1_0
from mftsci.utils.str import defaults
from mftsci.utils.str.core import *     # noqa: F401, F403


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def concat(
            a: Optional[Union[np.ndarray, list, str,]] = None,
            axes: Optional[Union[np.ndarray, list, tuple,]] = None,
            join_str: Optional[str] = None,
            ) -> np.ndarray:

    join_str = defaults.join_str(join_str)

    a = _convert(a=a).astype(str)
    if np.size(a) == 0:
        if axes is not None:
            N_axes = np.size(np.unique(axes))
        else:
            N_axes = 0
        return np.full(np.zeros(np.max([0, len(np.shape(a))-N_axes]), dtype=int),
                       fill_value="")
    a, N_i, shape_m = reshape_iter_pre(a=a, axes=axes, col_iter=False)

    lens = _fmap(a=a, f=len)
    u_len = np.max(np.sum(lens, axis=-1)) + (N_i-1) * len(join_str)

    out_a = np.full(shape_m[0], "", dtype=f"<U{u_len}")

    for k in range(N_i):
        out_a += a[..., k]
        if k < N_i - 1:
            out_a += join_str

    return out_a


def concat_1(
            a: Optional[Union[np.ndarray, list, str,]] = None,
            axes: Optional[Union[np.ndarray, list, tuple,]] = None,
            join_str: Optional[str] = None,
            ) -> np.ndarray:

    join_str = defaults.join_str(join_str)

    a = _convert(a=a).astype(str)
    if np.size(a) == 0:
        if axes is not None:
            N_axes = np.size(np.unique(axes))
        else:
            N_axes = 0
        return np.full(np.zeros(np.max([0, len(np.shape(a))-N_axes]), dtype=int),
                       fill_value="")
    a, N_i, shape_m = reshape_iter_pre(a=a, axes=axes, col_iter=False)

    out_a = np.array([join_str.join(a[..., k]) for k in range(N_i)])

    return out_a


def pad(
        a_s: Optional[Union[np.ndarray, list, str,]] = None,
        mode: Optional[str] = None,
        constant_values: Optional[str] = None,
        *args, **kwargs,
        ) -> np.ndarray:

    a_s = defaults.null(a_s)

    mode = defaults.mode(mode)
    kwargs["mode"] = mode

    if mode == "constant":
        constant_values = defaults.null(constant_values)
        kwargs["constant_values"] = constant_values

    return _pad(a_s=a_s, *args, **kwargs)


def _calc_pre_scalar(
                        pre: Optional[str] = None,
                        ):

    return defaults.null(pre)


def _calc_suff_scalar(
                        suff: Optional[str] = None,
                        ):

    return defaults.null(suff)


def _check_pre_scalar(
                        s: Optional[str] = None,
                        pre: Optional[str] = None,
                        ):

    s = defaults.s(s=s)
    pre = _calc_pre_scalar(pre=pre)

    return s.startswith(pre)


def _check_suff_scalar(
                        s: Optional[str] = None,
                        suff: Optional[str] = None,
                        ):

    s = defaults.s(s=s)
    suff = _calc_suff_scalar(suff=suff)

    return s.endswith(suff)


def _capp_pre_scalar(
                        s: Optional[str] = None,
                        pre: Optional[str] = None,
                        app_pre: Optional[bool] = None,
                        app_null: Optional[bool] = None,
                        ):

    s = defaults.s(s=s)
    app_null = defaults.app_null(app_null)
    app_pre = defaults.app_pre(app_pre)
    if not app_null and s == "":
        app_pre = False

    if app_pre:
        pre = _calc_pre_scalar(pre=pre)
        return f"{pre}{s}"
    else:
        return s


def _capp_suff_scalar(
                        s: Optional[str] = None,
                        suff: Optional[str] = None,
                        app_suff: Optional[bool] = None,
                        app_null: Optional[bool] = None,
                        ) -> str:

    s = defaults.s(s=s)
    app_null = defaults.app_null(app_null)
    app_suff = defaults.app_suff(app_suff)
    if not app_null and s == "":
        app_suff = False

    if app_suff:
        suff = _calc_suff_scalar(suff=suff)
        return f"{s}{suff}"
    else:
        return s


def _crm_pre_scalar(
                    s: Optional[str] = None,
                    pre: Optional[str] = None,
                    rm_pre: Optional[bool] = None,
                    ):

    s = defaults.s(s=s)
    rm_pre = defaults.rm_pre(rm_pre)

    pre = _calc_pre_scalar(pre=pre)

    check = _check_pre_scalar(s=s, pre=pre)

    if check and rm_pre:
        return s[len(pre):]
    else:
        return s


def _crm_suff_scalar(
                        s: Optional[str] = None,
                        suff: Optional[str] = None,
                        rm_suff: Optional[bool] = None,
                        ) -> str:

    s = defaults.s(s)
    rm_suff = defaults.rm_suff(rm_suff)

    suff = _calc_suff_scalar(suff=suff)

    check = _check_suff_scalar(s=s, suff=suff)

    if check and rm_suff:
        return s[:-len(suff)]
    else:
        return s


calc_pre = vectorize_pd()(_calc_pre_scalar)
calc_suff = vectorize_pd()(_calc_suff_scalar)
check_pre = vectorize_pd()(_check_pre_scalar)
check_suff = vectorize_pd()(_check_suff_scalar)
capp_pre = vectorize_pd()(_capp_pre_scalar)
capp_suff = vectorize_pd()(_capp_suff_scalar)
crm_pre = vectorize_pd()(_crm_pre_scalar)
crm_suff = vectorize_pd()(_crm_suff_scalar)


def _calc_ps_scalar(
                    ps: Optional[Union[np.ndarray, list, str]] = None,
                    pre: Optional[str] = None,
                    suff: Optional[str] = None,
                    ) -> np.ndarray:

    if ps is None:
        pre = _calc_pre_scalar(pre=pre)
        suff = _calc_suff_scalar(suff=suff)
        ps = np.moveaxis(np.array([pre, suff]), source=0, destination=-1)

    return ps


def _check_ps_scalar(
                        s: Optional[str] = None,
                        ps: Optional[Union[np.ndarray, list, str]] = None,
                        pre: Optional[str] = None,
                        suff: Optional[str] = None,
                        ) -> np.ndarray:

    s = defaults.s(s)
    ps = _calc_ps_scalar(ps=ps, pre=pre, suff=suff)

    return np.moveaxis(np.array([s.startswith(ps[..., 0]), s.endswith(ps[..., 1])]),
                       source=0, destination=-1)


def _capp_ps_scalar(
                    s: Optional[str] = None,
                    ps: Optional[Union[np.ndarray, list, str]] = None,
                    pre: Optional[str] = None,
                    suff: Optional[str] = None,
                    app_null: Optional[bool] = None,
                    ) -> str:

    s = defaults.s(s)
    app_null = defaults.app_null(app_null)

    if not app_null and s == "":
        return s
    else:
        ps = _calc_ps_scalar(ps=ps, pre=pre, suff=suff)
        return f"{ps[..., 0]}{s}{ps[..., 1]}"


def _crm_ps_scalar(
                    s: Optional[str] = None,
                    ps: Optional[Union[np.ndarray, list, str]] = None,
                    pre: Optional[str] = None,
                    suff: Optional[str] = None,
                    require: Optional[bool] = None,
                    ) -> str:

    if require is None:
        require = np.repeat(False, 2)
    if not np.array_equal(np.shape(require), np.array([2])):
        require = _pad(require, master_shape=[2])[0]

    s = defaults.s(s)
    ps = _calc_ps_scalar(ps=ps, pre=pre, suff=suff)

    check = _check_ps_scalar(s=s, ps=ps)

    ps_i = np.array([0, len(s)], dtype=int) \
            + np.all(check[require]) * check \
            * np.array([len(ps[..., 0]), -len(ps[..., 1])], dtype=int)  # noqa: E127

    return s[ps_i[0]:ps_i[1]]


_calc_ps_vector = vectorize_pd()(_calc_ps_scalar)
_check_ps_vector = vectorize_pd()(_check_ps_scalar)
_capp_ps_vector = vectorize_pd()(_capp_ps_scalar)
_crm_ps_vector = vectorize_pd()(_crm_ps_scalar)


def calc_ps(
            ps: Optional[Union[np.ndarray, list, str]] = None,
            pre: Optional[Union[np.ndarray, list, str]] = None,
            suff: Optional[Union[np.ndarray, list, str]] = None,
            mode: Optional[Union[np.ndarray, list, str]] = None,
            constant_values: Optional[Union[np.ndarray, list, str]] = None,
            *args, **kwargs,
            ) -> np.ndarray:

    if ps is None:
        _pre = calc_pre(pre=pre)
        _suff = calc_suff(suff=suff)
    elif isinstance(ps, list):
        _pre = ps[0]
        _suff = ps[1]
    else:
        _pre = ps[..., 0]
        _suff = ps[..., 1]

    ps = np.moveaxis(pad([_pre, _suff], mode=mode, constant_values=constant_values,
                         *args, **kwargs), source=0, destination=-1)

    return ps


def check_ps(
                s: Optional[Union[np.ndarray, list, str]] = None,
                ps: Optional[Union[np.ndarray, list, str]] = None,
                pre: Optional[Union[np.ndarray, list, str]] = None,
                suff: Optional[Union[np.ndarray, list, str]] = None,
                mode: Optional[Union[np.ndarray, list, str]] = None,
                constant_values: Optional[Union[np.ndarray, list, str]] = None,
                *args, **kwargs,
                ) -> np.ndarray:

    s = defaults.s(s)

    ps = calc_ps(ps=ps, pre=pre, suff=suff, mode=mode, constant_values=constant_values,
                 *args, **kwargs)

    return np.moveaxis(np.array([
                                    check_pre(s=s, pre=ps[..., 0]),
                                    check_suff(s=s, suff=ps[..., 1])
                                    ]), source=0, destination=-1)


def capp_ps(
            s: Optional[Union[np.ndarray, list, str]] = None,
            ps: Optional[Union[np.ndarray, list, str]] = None,
            pre: Optional[Union[np.ndarray, list, str]] = None,
            suff: Optional[Union[np.ndarray, list, str]] = None,
            app_ps: Optional[Union[np.ndarray, list, bool]] = None,
            app_pre: Optional[bool] = None,
            app_suff: Optional[bool] = None,
            app_null: Optional[bool] = None,
            mode: Optional[Union[np.ndarray, list, str]] = None,
            constant_values: Optional[Union[np.ndarray, list, str]] = None,
            *args, **kwargs
            ) -> Union[np.ndarray, str,]:

    s = defaults.s(s)
    app_null = defaults.app_null(app_null)
    app_pre = defaults.app_pre(app_pre)
    app_suff = defaults.app_suff(app_suff)

    ps = calc_ps(ps=ps, pre=pre, suff=suff, mode=mode, constant_values=constant_values,
                 *args, **kwargs)

    ps_shape = calc_shape(ps)

    app_pre = _pad(a_s=app_pre, master_shape=ps_shape[:-1], mode="wrap")[0]
    app_suff = _pad(a_s=app_suff, master_shape=ps_shape[:-1], mode="wrap")[0]

    return capp_suff(s=capp_pre(s=s, pre=ps[..., 0], app_pre=app_pre),
                     suff=ps[..., 1], app_suff=app_suff)


def crm_ps(
            s: Optional[Union[np.ndarray, list, str]] = None,
            ps: Optional[Union[np.ndarray, list, str]] = None,
            pre: Optional[Union[np.ndarray, list, str]] = None,
            suff: Optional[Union[np.ndarray, list, str]] = None,
            rm_ps: Optional[Union[np.ndarray, list, bool]] = None,
            rm_pre: Optional[bool] = None,
            rm_suff: Optional[bool] = None,
            require_ps: Optional[Union[np.ndarray, list, bool]] = None,
            require_pre: Optional[bool] = None,
            require_suff: Optional[bool] = None,
            mode: Optional[Union[np.ndarray, list, str]] = None,
            constant_values: Optional[Union[np.ndarray, list, str]] = None,
            *args, **kwargs
            ) -> Union[np.ndarray, str,]:

    s = defaults.s(s)
    rm_pre = defaults.require_pre(rm_pre)
    rm_suff = defaults.require_suff(rm_suff)
    require_pre = defaults.require_pre(require_pre)
    require_suff = defaults.require_suff(require_suff)

    ps = calc_ps(ps=ps, pre=pre, suff=suff, mode=mode, constant_values=constant_values,
                 *args, **kwargs)
    _check = check_ps(s=s, ps=ps)

    ps_shape = calc_shape(ps)

    require_ps = np.moveaxis(np.array([
                        _pad(a_s=require_pre, master_shape=ps_shape[:-1],
                             mode="wrap")[0],
                        _pad(a_s=require_suff, master_shape=ps_shape[:-1],
                             mode="wrap")[0],
                    ]), source=0, destination=-1)

    pass_require = np.all(_check | ~require_ps, axis=-1)

    rm_ps = pass_require*np.moveaxis(np.array([
                        _pad(a_s=rm_pre, master_shape=ps_shape[:-1], mode="wrap")[0],
                        _pad(a_s=rm_suff, master_shape=ps_shape[:-1], mode="wrap")[0],
                    ]), source=0, destination=-1)

    return crm_suff(s=crm_pre(s=s, pre=ps[..., 0], rm_pre=rm_ps[..., 0]),
                    suff=ps[..., 1], rm_suff=rm_ps[..., 1])


def calc_val_1_0(
                    app_val_1: Optional[bool] = None,
                    app_val_1_default: Optional[bool] = None,
                    val_1: Optional[str] = None,
                    val_0: Optional[str] = None,
                    ) -> str:

    val_0 = defaults.null(val_0)
    val_1 = defaults.null(val_1)

    return _calc_val_1_0(**locals())


def calc_val_1_0_N(
                    app_val_1: Optional[bool] = None,
                    app_val_1_default: Optional[bool] = None,
                    val_1: Optional[str] = None,
                    val_0: Optional[str] = None,
                    N: Optional[int] = None,
                    ) -> Union[np.ndarray, str,]:

    val_0 = defaults.null(val_0)
    val_1 = defaults.null(val_1)

    return _calc_val_1_0_N(**locals())


def str_to_int(
                s: str,
                ) -> int:

    # Create a SHA-256 hash of the string
    hash_digest = hashlib.sha256(s.encode()).hexdigest()

    # Convert the hex digest to an integer
    i = int(hash_digest, 16)

    # Optionally reduce to 32-bit range
    if False:
        i = i % 2**32

    return i
