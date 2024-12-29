"""
Module for string utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union

# External packages
import numpy as np

# Internal packages
from mftsci.utils.misc import calc_val_1_0 as _calc_val_1_0
from mftsci.utils.arr import iterarr, calc_shape, fattr, reshape_iter_pre, \
        pad as _pad, default as _default, f as _f, \
        calc_val_1_0_N as _calc_val_1_0_N   # noqa: F401
from mftsci.utils.dict import filter_keys


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def concat(
            a: Optional[Union[np.ndarray, list, str,]] = None,
            axes: Optional[Union[np.ndarray, list, tuple,]] = None,
            ) -> np.ndarray:

    a = _default(a=a)
    if np.size(a) == 0:
        if axes is not None:
            N_axes = np.size(np.unique(axes))
        else:
            N_axes = 0
        return np.full(np.zeros(np.max([0, len(np.shape(a))-N_axes]), dtype=int),
                       fill_value="")
    a, N_i, shape_m = reshape_iter_pre(a=a, axes=axes, col_iter=False)

    lens = _f(a=a, f=len)
    u_len = np.max(np.sum(lens, axis=-1))

    out_a = np.full(shape_m[0], "", dtype=f"<U{u_len}")

    for k in range(N_i):
        out_a += a[..., k]

    return out_a


def str_concatenate(
                    str_list: Union[np.ndarray, list],
                    fill_str: Optional[str] = None,
                    fill_str_void: Optional[Union[np.ndarray, list, int, bool,]] = None,
                    ) -> str:

    S = len(str_list)

    if fill_str is None:
        fill_str = ""
    if fill_str_void is None:
        fill_str_void = []
    fill_str_void = iterarr(fill_str_void)
    fill_str_void[fill_str_void < 0] = fill_str_void[fill_str_void < 0] + S

    outstr = ""

    for s in range(S):
        outstr += str(str_list[s])
        if not np.isin(s, fill_str_void):
            outstr += fill_str

    return outstr


def pad(
        a_s: Optional[Union[np.ndarray, list, str,]] = None,
        mode: Optional[str] = None,
        constant_values: Optional[str] = None,
        *args, **kwargs,
        ) -> np.ndarray:

    a_s = default_null(s=a_s)

    if mode is None:
        mode = "constant"
    kwargs["mode"] = mode

    if mode == "constant":
        constant_values = default_null(s=constant_values)
        kwargs["constant_values"] = constant_values

    return _pad(a_s=a_s, *args, **kwargs)


def _calc_pre_scalar(
                        pre: Optional[str] = None,
                        ):

    return default_null(s=pre)


def _calc_suff_scalar(
                        suff: Optional[str] = None,
                        ):

    return default_null(s=suff)


def _check_pre_scalar(
                        s: Optional[str] = None,
                        pre: Optional[str] = None,
                        ):

    s = default(s=s)
    pre = _calc_pre_scalar(pre=pre)

    return s.startswith(pre)


def _check_suff_scalar(
                        s: Optional[str] = None,
                        suff: Optional[str] = None,
                        ):

    s = default(s=s)
    suff = _calc_suff_scalar(suff=suff)

    return s.endswith(suff)


def _capp_pre_scalar(
                        s: Optional[str] = None,
                        pre: Optional[str] = None,
                        app_pre: Optional[bool] = None,
                        app_null: Optional[bool] = None,
                        ):

    if app_pre is None:
        app_pre = True

    s = default(s=s)

    if app_null is None:
        app_null = True

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

    if app_suff is None:
        app_suff = True

    s = default(s=s)

    if app_null is None:
        app_null = True

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

    if rm_pre is None:
        rm_pre = True

    s = default(s=s)
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

    if rm_suff is None:
        rm_suff = True

    s = default(s=s)
    suff = _calc_suff_scalar(suff=suff)

    check = _check_suff_scalar(s=s, suff=suff)

    if check and rm_suff:
        return s[:-len(suff)]
    else:
        return s


calc_pre = np.vectorize(_calc_pre_scalar)
calc_suff = np.vectorize(_calc_suff_scalar)
check_pre = np.vectorize(_check_pre_scalar)
check_suff = np.vectorize(_check_suff_scalar)
capp_pre = np.vectorize(_capp_pre_scalar)
capp_suff = np.vectorize(_capp_suff_scalar)
crm_pre = np.vectorize(_crm_pre_scalar)
crm_suff = np.vectorize(_crm_suff_scalar)


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

    s = default(s=s)
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

    if app_null is None:
        app_null = True

    s = default(s=s)

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

    s = default(s=s)
    ps = _calc_ps_scalar(ps=ps, pre=pre, suff=suff)

    check = _check_ps_scalar(s=s, ps=ps)

    ps_i = np.array([0, len(s)], dtype=int) \
            + np.all(check[require]) * check \
            * np.array([len(ps[..., 0]), -len(ps[..., 1])], dtype=int)  # noqa: E127

    return s[ps_i[0]:ps_i[1]]


_calc_ps_vector = np.vectorize(_calc_ps_scalar)
_check_ps_vector = np.vectorize(_check_ps_scalar)
_capp_ps_vector = np.vectorize(_capp_ps_scalar)
_crm_ps_vector = np.vectorize(_crm_ps_scalar)


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

    s = default(s=s)

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

    s = default(s=s)
    if app_null is None:
        app_null = True
    if app_pre is None:
        app_pre = True
    if app_suff is None:
        app_suff = True

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

    if rm_pre is None:
        rm_pre = True
    if rm_suff is None:
        rm_suff = True

    if require_pre is None:
        require_pre = False
    if require_suff is None:
        require_suff = False

    s = default(s=s)

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

    val_0 = default_null(s=val_0)
    val_1 = default_null(s=val_1)

    return _calc_val_1_0(**locals())


def calc_val_1_0_N(
                    app_val_1: Optional[bool] = None,
                    app_val_1_default: Optional[bool] = None,
                    val_1: Optional[str] = None,
                    val_0: Optional[str] = None,
                    N: Optional[int] = None,
                    ) -> Union[np.ndarray, str,]:

    val_0 = default_null(s=val_0)
    val_1 = default_null(s=val_1)

    return _calc_val_1_0_N(**locals())


def calc_brac(
                brac_type: Optional[str] = None,
                add_lr: Optional[bool] = None,
                repeat: Optional[int] = None,
                app_brac: Optional[Union[np.ndarray, list, bool]] = None,
                phantom: Optional[bool] = None,
                null_phantom: Optional[bool] = None,
                plot: Optional[bool] = None,
                ) -> np.ndarray:

    if plot is None:
        plot = False

    if phantom is None:
        phantom = False

    if brac_type is None:
        brac_type = "parentheses"

    if add_lr is None:
        if plot:
            add_lr = True
        else:
            add_lr = False

    if repeat is None:
        repeat = 1

    if np.isin(brac_type, [""]):
        brac = ["", ""]
        add_lr = False
    elif np.isin(brac_type, ["parentheses", "()", "(", ")"]):
        brac = ["(", ")"]
    elif np.isin(brac_type, ["square", "[]", "[", "]"]):
        brac = ["[", "]"]
    elif np.isin(brac_type, ["angular", "<>", "<", ">"]):
        brac = ["<", ">"]
    elif np.isin(brac_type, ["curly", "{}", "{", "}"]):
        brac = [f"\\{{", f"\\}}"]   # noqa: F541
    else:
        brac = ["(", ")"]

    if phantom:
        add_lr = False
        if null_phantom is None:
            null_phantom = False

        if null_phantom:
            app_brac = False

        if plot:
            brac = np.repeat("\\:", 2)
        else:
            brac = capp_tex_fn(s=brac, tex_fn="phantom", app_tex_fn=True, app_null=None)

    if add_lr:
        lr = ["\\left", "\\right"]
    else:
        lr = ["", ""]

    return calc_val_1_0_N(app_val_1=app_brac, app_val_1_default=True,
                          val_1=[[concat(np.repeat(f"{lr[k]}{brac[k]}", repeat))
                                 for k in range(2)]], val_0="", N=2)


def calc_brac_ps(
                    brac_type: Optional[str] = None,
                    add_lr: Optional[bool] = None,
                    app_brac: Optional[Union[np.ndarray, list, bool]] = None,
                    repeat: Optional[int] = None,
                    ) -> np.ndarray:
    return calc_brac(brac_type=brac_type, add_lr=add_lr, app_brac=app_brac,
                     repeat=repeat)


def capp_brac(
                s: Optional[Union[np.ndarray, list, str]] = None,
                brac_type: Optional[str] = None,
                add_lr: Optional[bool] = None,
                app_brac: Optional[Union[np.ndarray, list, bool]] = None,
                repeat: Optional[int] = None,
                app_null: Optional[bool] = None,
                ) -> Union[np.ndarray, str,]:
    if app_null is None:
        app_null = False
    return capp_ps(s=s,
                   ps=calc_brac_ps(**filter_keys(locals(), keys_rm=["s", "app_null"])),
                   app_null=app_null)


def check_brac(
                s: Optional[Union[np.ndarray, list, str]] = None,
                brac_type: Optional[str] = None,
                add_lr: Optional[bool] = None,
                app_brac: Optional[Union[np.ndarray, list, bool]] = None,
                repeat: Optional[int] = None,
                ) -> np.ndarray:
    return check_ps(s=s, ps=calc_brac_ps(**filter_keys(locals(), keys_rm=["s"])))


def crm_brac(
                s: Optional[Union[np.ndarray, list, str]] = None,
                brac_type: Optional[str] = None,
                add_lr: Optional[bool] = None,
                app_brac: Optional[Union[np.ndarray, list, bool]] = None,
                repeat: Optional[int] = None,
                ) -> Union[np.ndarray, str,]:
    return crm_ps(s=s, ps=calc_brac_ps(**filter_keys(locals(), keys_rm=["s"])))


def calc_dlrs_ps(
                    app_dlrs: Optional[bool] = None,
                    ) -> np.ndarray:
    return calc_val_1_0_N(app_val_1=app_dlrs, app_val_1_default=True,
                          val_1="$", val_0="", N=2)


def capp_dlrs(
                s: Optional[Union[np.ndarray, list, str]] = None,
                app_dlrs: Optional[bool] = None,
                app_null: Optional[bool] = None,
                ) -> Union[np.ndarray, str,]:
    if app_null is None:
        app_null = False
    return capp_ps(s=s,
                   ps=calc_dlrs_ps(**filter_keys(locals(), keys_rm=["s", "app_null"])),
                   app_null=app_null)


def check_dlrs(
                s: Optional[Union[np.ndarray, list, str]] = None,
                app_dlrs: Optional[bool] = None,
                ) -> np.ndarray:
    return check_ps(s=s, ps=calc_dlrs_ps(**filter_keys(locals(), keys_rm=["s"])))


def crm_dlrs(
                s: Optional[Union[np.ndarray, list, str]] = None,
                app_dlrs: Optional[bool] = None,
                ) -> Union[np.ndarray, str,]:
    return crm_ps(s=s, ps=calc_dlrs_ps(**filter_keys(locals(), keys_rm=["s"])))


def calc_tex_fn_ps(
                    tex_fn: Optional[str] = None,
                    app_tex_fn: Optional[bool] = None,
                    ) -> np.ndarray:
    if tex_fn is None:
        tex_fn = ""
    if tex_fn == "":
        app_tex_fn = False
    return calc_val_1_0_N(app_val_1=app_tex_fn, app_val_1_default=True,
                          val_1=[f"\\{tex_fn}{{", f"}}"], val_0="", N=2)    # noqa: F541


def capp_tex_fn(
                s: Optional[Union[np.ndarray, list, str]] = None,
                tex_fn: Optional[str] = None,
                app_tex_fn: Optional[bool] = None,
                app_null: Optional[bool] = None,
                ) -> Union[np.ndarray, str,]:
    if app_null is None:
        app_null = False
    return capp_ps(s=s,
                   ps=calc_tex_fn_ps(**filter_keys(locals(),
                                                   keys_rm=["s", "app_null"])),
                   app_null=app_null)


def check_tex_fn(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_fn: Optional[str] = None,
                    app_tex_fn: Optional[bool] = None,
                    ) -> np.ndarray:
    return check_ps(s=s, ps=calc_tex_fn_ps(**filter_keys(locals(), keys_rm=["s"])))


def crm_tex_fn(
                s: Optional[Union[np.ndarray, list, str]] = None,
                tex_fn: Optional[str] = None,
                app_tex_fn: Optional[bool] = None,
                ) -> Union[np.ndarray, str,]:
    return crm_ps(s=s, ps=calc_tex_fn_ps(**filter_keys(locals(), keys_rm=["s"])))


def calc_tex_num_fn(
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    tex_num: Optional[str] = None,
                    ) -> str:

    if tex_num is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            tex_num = "num"
        else:
            tex_num = "num"

    return tex_num


def calc_tex_num_ps(
                    tex_num: Optional[str] = None,
                    app_tex_num: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> np.ndarray:

    if app_tex_num is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            app_tex_num = False
        else:
            app_tex_num = True

    tex_num = calc_tex_num_fn(pltf=pltf, use_tex=use_tex, tex_num=tex_num)

    return calc_tex_fn_ps(tex_fn=tex_num, app_tex_fn=app_tex_num)


def capp_tex_num(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_num: Optional[str] = None,
                    app_tex_num: Optional[bool] = None,
                    app_null: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> Union[np.ndarray, str,]:
    if app_null is None:
        app_null = False
    return capp_ps(s=s,
                   ps=calc_tex_num_ps(**filter_keys(locals(),
                                                    keys_rm=["s", "app_null"])),
                   app_null=app_null)


def check_tex_num(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_num: Optional[str] = None,
                    app_tex_num: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> np.ndarray:
    return check_ps(s=s, ps=calc_tex_num_ps(**filter_keys(locals(), keys_rm=["s"])))


def crm_tex_num(
                s: Optional[Union[np.ndarray, list, str]] = None,
                tex_num: Optional[str] = None,
                app_tex_num: Optional[bool] = None,
                pltf: Optional[bool] = None,
                use_tex: Optional[bool] = None,
                ) -> Union[np.ndarray, str,]:
    return crm_ps(s=s, ps=calc_tex_num_ps(**filter_keys(locals(), keys_rm=["s"])))


def capp_dlrs_tex_num(
                        s: Optional[Union[np.ndarray, list, str]] = None,
                        app_dlrs: Optional[bool] = None,
                        app_tex_num: Optional[bool] = None,
                        app_null: Optional[bool] = None,
                        pltf: Optional[bool] = None,
                        use_tex: Optional[bool] = None,
                        ) -> Union[np.ndarray, str,]:
    return capp_dlrs(
                        s=capp_tex_num(
                                        s=s, app_tex_num=app_tex_num, app_null=app_null,
                                        pltf=pltf, use_tex=use_tex,
                                        ),
                        app_dlrs=app_dlrs, app_null=app_null,
                        )


def calc_tex_vect_fn(
                        pltf: Optional[bool] = None,
                        use_tex: Optional[bool] = None,
                        tex_vect: Optional[str] = None,
                        ) -> str:

    if tex_vect is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            tex_vect = "mathbf"
        else:
            tex_vect = "vect"

    return tex_vect


def calc_tex_vect_ps(
                        tex_vect: Optional[str] = None,
                        app_tex_vect: Optional[bool] = None,
                        pltf: Optional[bool] = None,
                        use_tex: Optional[bool] = None,
                        ) -> np.ndarray:

    tex_vect = calc_tex_vect_fn(pltf=pltf, use_tex=use_tex, tex_vect=tex_vect)

    return calc_tex_fn_ps(tex_fn=tex_vect, app_tex_fn=app_tex_vect)


def capp_tex_vect(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_vect: Optional[str] = None,
                    app_tex_vect: Optional[bool] = None,
                    app_null: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> Union[np.ndarray, str,]:
    if app_null is None:
        app_null = False
    return capp_ps(s=s,
                   ps=calc_tex_vect_ps(**filter_keys(locals(),
                                                     keys_rm=["s", "app_null"])),
                   app_null=app_null)


def check_tex_vect(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_vect: Optional[str] = None,
                    app_tex_vect: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> np.ndarray:
    return check_ps(s=s, ps=calc_tex_vect_ps(**filter_keys(locals(), keys_rm=["s"])))


def crm_tex_vect(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_vect: Optional[str] = None,
                    app_tex_vect: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> Union[np.ndarray, str,]:
    return crm_ps(s=s, ps=calc_tex_vect_ps(**filter_keys(locals(), keys_rm=["s"])))


def calc_tex_hphantom_fn(
                            pltf: Optional[bool] = None,
                            use_tex: Optional[bool] = None,
                            tex_hphantom: Optional[str] = None,
                            ) -> str:

    if tex_hphantom is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            tex_hphantom = 'hphantom'
        else:
            tex_hphantom = 'hphantom'

    return tex_hphantom


def calc_tex_hphantom_ps(
                            tex_hphantom: Optional[str] = None,
                            app_tex_hphantom: Optional[bool] = None,
                            pltf: Optional[bool] = None,
                            use_tex: Optional[bool] = None,
                            ) -> np.ndarray:

    if app_tex_hphantom is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            app_tex_hphantom = False
        else:
            app_tex_hphantom = True

    tex_hphantom = calc_tex_hphantom_fn(pltf=pltf, use_tex=use_tex,
                                        tex_hphantom=tex_hphantom)

    return calc_tex_fn_ps(tex_fn=tex_hphantom, app_tex_fn=app_tex_hphantom)


def capp_tex_hphantom(
                        s: Optional[Union[np.ndarray, list, str]] = None,
                        tex_hphantom: Optional[str] = None,
                        app_tex_hphantom: Optional[bool] = None,
                        app_null: Optional[bool] = None,
                        pltf: Optional[bool] = None,
                        use_tex: Optional[bool] = None,
                        ) -> Union[np.ndarray, str,]:
    if app_null is None:
        app_null = False
    return capp_ps(s=s,
                   ps=calc_tex_hphantom_ps(**filter_keys(locals(),
                                                         keys_rm=["s", "app_null"])),
                   app_null=app_null)


def check_tex_hphantom(
                        s: Optional[Union[np.ndarray, list, str]] = None,
                        tex_hphantom: Optional[str] = None,
                        app_tex_hphantom: Optional[bool] = None,
                        pltf: Optional[bool] = None,
                        use_tex: Optional[bool] = None,
                        ) -> np.ndarray:
    return check_ps(s=s,
                    ps=calc_tex_hphantom_ps(**filter_keys(locals(), keys_rm=["s"])))


def crm_tex_hphantom(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_hphantom: Optional[str] = None,
                    app_tex_hphantom: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> Union[np.ndarray, str,]:
    return crm_ps(s=s, ps=calc_tex_hphantom_ps(**filter_keys(locals(), keys_rm=["s"])))


# Defaults
# -------------------------------------


def default_null(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    ) -> str:
    if s is None:
        s = ""
    return s


def default(
            s: Optional[Union[np.ndarray, list, str]] = None,
            ) -> str:

    return default_null(s=s)
