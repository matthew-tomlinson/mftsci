"""
Functions module for string prefix and suffix utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union


# External packages
import numpy as np


# Internal packages
from mftsci.utils.dict import filter_keys
from mftsci.utils.str import concat, pad, fattr, \
                            calc_ps, check_ps, capp_ps as _capp_ps, crm_ps, \
                            calc_val_1_0, calc_val_1_0_N    # noqa: F401
from mftsci.utils.ps import defaults   # noqa: F401


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


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

    app_null = defaults.app_null(app_null)

    return _capp_ps(**locals())


def calc_brac(
                brac_type: Optional[str] = None,
                add_lr: Optional[bool] = None,
                repeat: Optional[int] = None,
                app_brac: Optional[Union[np.ndarray, list, bool]] = None,
                phantom: Optional[bool] = None,
                null_phantom: Optional[bool] = None,
                pltf: Optional[bool] = None,
                ) -> np.ndarray:

    pltf = defaults.pltf(pltf)
    phantom = defaults.phantom(phantom)
    brac_type = defaults.brac_type(brac_type)
    repeat = defaults.repeat(repeat)

    add_lr = defaults.set_default(val=add_lr, default=(pltf and not phantom))

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
        null_phantom = defaults.null_phantom(null_phantom)

        if null_phantom:
            app_brac = False

        if pltf:
            brac = np.repeat("\\:", 2)
        else:
            brac = capp_tex_fn(s=brac, tex_fn="phantom", app_tex_fn=True, app_null=None)

    if add_lr:
        lr = ["\\left", "\\right"]
    else:
        lr = ["", ""]

    return calc_val_1_0_N(app_val_1=app_brac, app_val_1_default=True,
                          val_1=[concat(np.repeat(f"{lr[k]}{brac[k]}", repeat))
                                 for k in range(2)], val_0="", N=2)


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

    tex_fn = defaults.tex_fn(tex_fn)

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


def calc_tex_num_ps(
                    tex_num: Optional[str] = None,
                    app_tex_num: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> np.ndarray:

    if app_tex_num is None:
        pltf = defaults.pltf(pltf)
        use_tex = defaults.use_tex(use_tex)

        app_tex_num = not (pltf and not use_tex)

    tex_num = defaults.tex_num(tex_num)

    return calc_tex_fn_ps(tex_fn=tex_num, app_tex_fn=app_tex_num)


def capp_tex_num(
                    s: Optional[Union[np.ndarray, list, str]] = None,
                    tex_num: Optional[str] = None,
                    app_tex_num: Optional[bool] = None,
                    app_null: Optional[bool] = None,
                    pltf: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    ) -> Union[np.ndarray, str,]:

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
        pltf = defaults.pltf(pltf)
        use_tex = defaults.use_tex(use_tex)

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


def calc_tex_hphantom_ps(
                            tex_hphantom: Optional[str] = None,
                            app_tex_hphantom: Optional[bool] = None,
                            pltf: Optional[bool] = None,
                            use_tex: Optional[bool] = None,
                            ) -> np.ndarray:

    if app_tex_hphantom is None:
        pltf = defaults.pltf(pltf)
        use_tex = defaults.use_tex(use_tex)

        app_tex_hphantom = not (pltf and not use_tex)

    tex_hphantom = defaults.tex_hphantom(tex_hphantom)

    return calc_tex_fn_ps(tex_fn=tex_hphantom, app_tex_fn=app_tex_hphantom)


def capp_tex_hphantom(
                        s: Optional[Union[np.ndarray, list, str]] = None,
                        tex_hphantom: Optional[str] = None,
                        app_tex_hphantom: Optional[bool] = None,
                        app_null: Optional[bool] = None,
                        pltf: Optional[bool] = None,
                        use_tex: Optional[bool] = None,
                        ) -> Union[np.ndarray, str,]:

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
