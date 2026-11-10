"""
Functions module for character utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal

# External packages
import numpy as np

# Internal packages
from mftsci.utils import arr
from mftsci.utils import ps as _str_ps
from mftsci.utils.dict import keys, vals
from mftsci.utils.str import pad
from mftsci.utils.char import defaults


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def calc_ps_roman(
                    ps: Optional[Union[np.ndarray, list,]] = None,
                    pre: Optional[str] = None,
                    suff: Optional[str] = None,
                    app_mathrm: Optional[bool] = None,
                ) -> np.ndarray:

    app_mathrm = defaults.app_mathrm(app_mathrm)

    ps = _str_ps.calc_ps(ps=ps, pre=pre, suff=suff)

    ps_roman = _str_ps.calc_tex_fn_ps(tex_fn="mathrm", app_tex_fn=app_mathrm)

    ps[0] = f"{ps[0]}{ps_roman[0]}"
    ps[1] = f"{ps_roman[1]}{ps[1]}"

    return ps


def calc_ps_greek(
                    ps: Optional[Union[np.ndarray, list,]] = None,
                    pre: Optional[str] = None,
                    suff: Optional[str] = None,
                    app_bs: Optional[bool] = None,
                    app_up: Optional[bool] = None,
                    app_var: Optional[bool] = None,
                    ) -> np.ndarray:

    ps = _str_ps.calc_ps(ps=ps, pre=pre, suff=suff)

    pre_greek = "".join([
                    _str_ps.calc_val_1_0(
                                            app_val_1=app_bs, app_val_1_default=False,
                                            val_1="\\", val_0=None,
                                            ),
                    _str_ps.calc_val_1_0(
                                            app_val_1=app_up, app_val_1_default=False,
                                            val_1="up", val_0=None,
                                            ),
                    _str_ps.calc_val_1_0(
                                            app_val_1=app_var, app_val_1_default=False,
                                            val_1="var", val_0=None,
                                            ),
                    ])

    ps[0] = f"{ps[0]}{pre_greek}"

    return ps


def calc_ps_alphabet(
                        ps: Optional[Union[np.ndarray, list,]] = None,
                        pre: Optional[str] = None,
                        suff: Optional[str] = None,
                        alphabet: Optional[Literal["greek", "roman"]] = None,
                        app_mathrm: Optional[bool] = None,
                        app_bs: Optional[bool] = None,
                        app_up: Optional[bool] = None,
                        app_var: Optional[bool] = None,
                        ) -> np.ndarray:

    if alphabet == "greek":
        _ps = calc_ps_greek(ps=ps, pre=pre, suff=suff,
                            app_bs=app_bs, app_up=app_up, app_var=app_var)
    else:
        _ps = calc_ps_roman(ps=ps, pre=pre, suff=suff, app_mathrm=app_mathrm)

    return _ps


def get_alphabet_s(
                    s: Optional[Union[np.ndarray, list, str,]] = None,
                    ps: Optional[Union[np.ndarray, list,]] = None,
                    pre: Optional[str] = None,
                    suff: Optional[str] = None,
                    alphabet: Optional[Literal["greek", "roman"]] = None,
                    app_mathrm: Optional[bool] = None,
                    app_bs: Optional[bool] = None,
                    app_up: Optional[bool] = None,
                    app_var: Optional[bool] = None,
                    attr: Optional[str] = None,
                    attr_args: Optional[tuple] = None,
                    attr_kwargs: Optional[dict] = None,
                    pattr: Optional[str] = None,
                    pattr_args: Optional[tuple] = None,
                    pattr_kwargs: Optional[dict] = None,
                    ) -> Union[np.ndarray, str]:

    s = defaults.s(s)

    _s = _str_ps.fattr(s, attr=attr, args=attr_args, kwargs=attr_kwargs)

    _ps = calc_ps_alphabet(ps=ps, pre=pre, suff=suff, alphabet=alphabet,
                           app_mathrm=app_mathrm, app_bs=app_bs,
                           app_up=app_up, app_var=app_var)

    return _str_ps.fattr(
                        _str_ps.capp_ps(s=_s, ps=_ps),
                        attr=pattr,
                        args=pattr_args, kwargs=pattr_kwargs,
                        )


def get_arab_num(
                    ps: Optional[Union[np.ndarray, list,]] = None,
                    pre: Optional[str] = None,
                    suff: Optional[str] = None,
                    app_mathrm: Optional[bool] = None,
                    attr: Optional[str] = None,
                    attr_args: Optional[tuple] = None,
                    attr_kwargs: Optional[dict] = None,
                    pattr: Optional[str] = None,
                    pattr_args: Optional[tuple] = None,
                    pattr_kwargs: Optional[dict] = None,
                    ) -> np.ndarray:

    s = np.array(np.arange(10), dtype="<U1")

    return get_alphabet_s(alphabet="roman", **locals())


def get_roman_letters(
                        ps: Optional[Union[np.ndarray, list,]] = None,
                        pre: Optional[str] = None,
                        suff: Optional[str] = None,
                        app_mathrm: Optional[bool] = None,
                        attr: Optional[str] = None,
                        attr_args: Optional[tuple] = None,
                        attr_kwargs: Optional[dict] = None,
                        pattr: Optional[str] = None,
                        pattr_args: Optional[tuple] = None,
                        pattr_kwargs: Optional[dict] = None,
                        ) -> np.ndarray:

    s = np.array([
                    "a", "b", "c", "d", "e", "f",
                    "g", "h", "i", "j", "k", "l",
                    "m", "n", "o", "p", "q", "r",
                    "s", "t", "u", "v", "w", "x",
                    "y", "z",
                  ])

    return get_alphabet_s(alphabet="roman", **locals())


def calc_roman_letters_case_symb_dict(
                                        roman_app_mathrm: Optional[bool] = None,
                                        capital: Optional[bool] = None,
                                        tex_vect: Optional[str] = None,
                                        app_tex_vect: Optional[bool] = None,
                                        pltf: Optional[bool] = None,
                                        use_tex: Optional[bool] = None,
                                        ) -> dict:

    capital = defaults.capital(capital)
    app_tex_vect = defaults.app_tex_vect(app_tex_vect)
    pltf = defaults.pltf(pltf)
    use_tex = defaults.use_tex(use_tex)

    if roman_app_mathrm is None:
        if pltf and not use_tex:
            roman_app_mathrm = False
        else:
            roman_app_mathrm = app_tex_vect

    if capital:
        attr = "capitalize"
    else:
        attr = None

    _roman_letters = get_roman_letters(app_mathrm=False, attr=attr)
    _roman_letters_symb = _str_ps.capp_tex_vect(
                                    get_roman_letters(
                                                        app_mathrm=roman_app_mathrm,
                                                        attr=attr
                                                    ),
                                    tex_vect=tex_vect, app_tex_vect=app_tex_vect,
                                    pltf=pltf, use_tex=use_tex, app_null=False
                                    )

    _roman_letters_symb_dict = {_roman_letters[i]: _roman_letters_symb[i]
                                for i in range(_roman_letters.size)}

    return _roman_letters_symb_dict


def _get_greek_omicron(
                        ow_omicron: Optional[bool] = None,
                        app_omicron: Optional[bool] = None,
                        ) -> np.ndarray:

    ow_omicron = defaults.ow_omicron(ow_omicron)
    app_omicron = defaults.app_omicron(ow_omicron)

    if ow_omicron:
        omicron = ["omicron"]
    elif app_omicron:
        omicron = ["o", "omicron"]
    else:
        omicron = ["o"]

    return arr.iterarr(omicron)


def _get_greek_letters(
                            var: Optional[bool] = None,
                            blank: Optional[bool] = None,
                            blank_str: Optional[str] = None,
                            ow_omicron: Optional[bool] = None,
                            app_omicron: Optional[bool] = None,
                            ps: Optional[Union[np.ndarray, list,]] = None,
                            pre: Optional[str] = None,
                            suff: Optional[str] = None,
                            app_bs: Optional[bool] = None,
                            app_up: Optional[bool] = None,
                            app_var: Optional[bool] = None,
                            attr: Optional[str] = None,
                            attr_args: Optional[tuple] = None,
                            attr_kwargs: Optional[dict] = None,
                            pattr: Optional[str] = None,
                            pattr_args: Optional[tuple] = None,
                            pattr_kwargs: Optional[dict] = None,
                            ) -> np.ndarray:

    blank = defaults.blank(blank)
    blank_str = defaults.blank_str(blank_str)
    var = defaults.var(var)

    if app_var is None:
        if var:
            app_var = True

    if var:
        s = np.array(["epsilon", "theta", "pi", "rho", "sigma", "phi"])
    else:
        s = np.concatenate([
                            np.array(["alpha", "beta", "gamma", "delta", "epsilon",
                                      "zeta", "eta", "theta", "iota", "kappa", "lambda",
                                      "mu", "nu", "xi"]),
                            _get_greek_omicron(ow_omicron=ow_omicron,
                                               app_omicron=app_omicron),
                            np.array(["pi", "rho", "sigma", "tau", "upsilon", "phi",
                                      "chi", "psi", "omega"]),
                        ])
    if blank:
        s = np.repeat(blank_str, s.size)

    return get_alphabet_s(
                        s=s,
                        ps=ps, pre=pre, suff=suff,
                        alphabet="greek",
                        app_bs=app_bs, app_up=app_up, app_var=app_var,
                        attr=attr, attr_args=attr_args, attr_kwargs=attr_kwargs,
                        pattr=pattr, pattr_args=pattr_args, pattr_kwargs=pattr_kwargs
                        )


def get_greek_letters(
                        app_omicron: Optional[bool] = None,
                        ow_omicron: Optional[bool] = None,
                        ps: Optional[Union[np.ndarray, list,]] = None,
                        pre: Optional[str] = None,
                        suff: Optional[str] = None,
                        app_bs: Optional[bool] = None,
                        app_up: Optional[bool] = None,
                        app_var: Optional[bool] = None,
                        incl_var: Optional[bool] = None,
                        reorder_incl_var: Optional[bool] = None,
                        incl_var_app_var: Optional[bool] = None,
                        ow_app_var: Optional[bool] = None,
                        attr: Optional[str] = None,
                        attr_args: Optional[tuple] = None,
                        attr_kwargs: Optional[dict] = None,
                        pattr: Optional[str] = None,
                        pattr_args: Optional[tuple] = None,
                        pattr_kwargs: Optional[dict] = None,
                        blank_nvar: Optional[bool] = None,
                        blank_var: Optional[bool] = None,
                        blank_str: Optional[str] = None,
                        blank_str_nvar: Optional[bool] = None,
                        blank_str_var: Optional[bool] = None,
                      ) -> np.ndarray:

    reorder_incl_var = defaults.set_default(reorder_incl_var, default=False)
    incl_var = defaults.set_default(incl_var, default=reorder_incl_var)

    blank_str = defaults.blank_str(blank_str)
    blank_str_nvar = defaults.set_default(val=blank_str_nvar, default=blank_str)
    blank_str_var = defaults.set_default(val=blank_str_var, default=blank_str)

    greek_letters = _get_greek_letters(
                        var=False,
                        blank=blank_nvar, blank_str=blank_str_nvar,
                        ow_omicron=ow_omicron, app_omicron=app_omicron,
                        ps=ps, pre=pre, suff=suff,
                        app_bs=app_bs, app_up=app_up, app_var=False,
                        attr=attr, attr_args=attr_args, attr_kwargs=attr_kwargs,
                        pattr=pattr, pattr_args=pattr_args, pattr_kwargs=pattr_kwargs,
                        )

    if incl_var:
        incl_var_app_var = defaults.set_default(incl_var_app_var, False)

        greek_letters_var = _get_greek_letters(
                        var=False,
                        blank=blank_var, blank_str=blank_str_var,
                        ow_omicron=ow_omicron, app_omicron=app_omicron,
                        ps=ps, pre=pre, suff=suff,
                        app_bs=app_bs, app_up=app_up, app_var=True,
                        attr=attr, attr_args=attr_args, attr_kwargs=attr_kwargs,
                        pattr=pattr, pattr_args=pattr_args, pattr_kwargs=pattr_kwargs,
                        )

        greek_letters_incl = np.concatenate([greek_letters, greek_letters_var])

        if reorder_incl_var:
            _greek_letters = _get_greek_letters(
                                var=False,
                                ow_omicron=ow_omicron, app_omicron=app_omicron,
                                ps=ps, pre=pre, suff=suff,
                                app_bs=False, app_up=False, app_var=True,
                                attr=attr, attr_args=attr_args, attr_kwargs=attr_kwargs,
                                )

            _greek_letters_var = _get_greek_letters(
                                var=False,
                                ow_omicron=ow_omicron, app_omicron=app_omicron,
                                ps=ps, pre=pre, suff=suff,
                                app_bs=False, app_up=False, app_var=False,
                                attr=attr, attr_args=attr_args, attr_kwargs=attr_kwargs,
                                )
            order = np.argsort(np.concatenate([
                        np.arange(_greek_letters.size),
                        0.01 + np.array([
                            float(np.where(_greek_letters_var[i] == _greek_letters)[0])
                            for i in range(_greek_letters_var.size)
                            ])
                        ]))
        else:
            order = np.arange(greek_letters_incl.size)

        greek_letters = greek_letters_incl[order]

    return greek_letters


def get_greek_letters_var_bool(
                                app_omicron: Optional[bool] = None,
                                ow_omicron: Optional[bool] = None,
                                incl_var: Optional[bool] = None,
                                reorder_incl_var: Optional[bool] = None,
                                incl_var_app_var: Optional[bool] = None,
                                ) -> np.ndarray:

    incl_var = defaults.set_default(incl_var, default=False)
    reorder_incl_var = defaults.set_default(reorder_incl_var, default=True)
    incl_var_app_var = defaults.set_default(incl_var_app_var, default=False)

    _greek_letters = get_greek_letters(
                                        app_omicron=app_omicron, ow_omicron=ow_omicron,
                                        incl_var=incl_var,
                                        reorder_incl_var=reorder_incl_var,
                                        incl_var_app_var=incl_var_app_var,
                                        )
    _greek_letters_var = _get_greek_letters(var=True,
                                            app_var=bool(incl_var*incl_var_app_var))

    return np.isin(_greek_letters, _greek_letters_var)


def calc_greek_letters_roman_dict(
                                    roman_app_mathrm: Optional[bool] = None,
                                    tex_vect: Optional[str] = None,
                                    app_tex_vect: Optional[bool] = None,
                                    pltf: Optional[bool] = None,
                                    use_tex: Optional[bool] = None,
                                    ) -> dict:

    app_tex_vect = defaults.app_tex_vect(app_tex_vect=app_tex_vect)
    pltf = defaults.pltf(pltf=pltf)
    use_tex = defaults.use_tex(use_tex=use_tex)

    if roman_app_mathrm is None:
        if pltf and not use_tex:
            roman_app_mathrm = False
        else:
            roman_app_mathrm = app_tex_vect

    _greek_letters_roman_dict = {
                                    "Alpha": "A",
                                    "Beta": "B",
                                    "Epsilon": "E",
                                    "Zeta": "Z",
                                    "Eta": "H",
                                    "Iota": "I",
                                    "Kappa": "K",
                                    "Mu": "M",
                                    "Nu": "N",
                                    "Rho": "P",
                                    "Tau": "T",
                                    "Chi": "X",
                                }

    for l in _greek_letters_roman_dict:     # noqa: E741
        _greek_letters_roman_dict[l] = _str_ps.capp_tex_vect(
                                                _str_ps.capp_tex_fn(
                                                    _greek_letters_roman_dict[l],
                                                    tex_fn="mathrm",
                                                    app_tex_fn=roman_app_mathrm
                                                    ),
                                                tex_vect=tex_vect,
                                                app_tex_vect=app_tex_vect,
                                                pltf=pltf, use_tex=use_tex,
                                                app_null=False,
                                                )

    return _greek_letters_roman_dict


def calc_greek_letters_case_symb_dict(
                                        app_omicron: Optional[bool] = None,
                                        ow_omicron: Optional[bool] = None,
                                        safe_omicron: Optional[bool] = None,
                                        roman_sub: Optional[bool] = None,
                                        roman_app_mathrm: Optional[bool] = None,
                                        capital: Optional[bool] = None,
                                        var: Optional[bool] = None,
                                        blank_nvar: Optional[bool] = None,
                                        app_var_nvar: Optional[bool] = None,
                                        ow_app_var: Optional[bool] = None,
                                        tex_vect: Optional[str] = None,
                                        app_tex_vect: Optional[bool] = None,
                                        pltf: Optional[bool] = None,
                                        use_tex: Optional[bool] = None,
                                        ) -> dict:

    capital = defaults.capital(capital)
    ow_omicron = defaults.ow_omicron(ow_omicron)
    app_omicron = defaults.app_omicron(app_omicron)
    var = defaults.var(var)

    app_tex_vect = defaults.app_tex_vect(app_tex_vect=app_tex_vect)
    pltf = defaults.pltf(pltf=pltf)
    use_tex = defaults.use_tex(use_tex=use_tex)

    if capital:
        attr = "capitalize"
    else:
        attr = None

    if roman_app_mathrm is None:
        if pltf and not use_tex:
            roman_app_mathrm = False
        else:
            roman_app_mathrm = app_tex_vect

    if var and not capital:

        blank_nvar = defaults.set_default(blank_nvar, default=True)
        ow_app_var = defaults.set_default(ow_app_var, default=True)
        app_var_nvar = defaults.set_default(app_var_nvar, default=False)

        _greek_letters = get_greek_letters(
                                            app_omicron=app_omicron,
                                            ow_omicron=ow_omicron,
                                            app_bs=False,
                                            app_var=True,
                                            attr=attr,
                                            )
        _greek_letters_symb = _str_ps.capp_tex_vect(
                                    get_greek_letters(
                                                        app_omicron=app_omicron,
                                                        ow_omicron=ow_omicron,
                                                        app_bs=True,
                                                        app_var=app_var_nvar,
                                                        blank_nvar=blank_nvar,
                                                        blank_var=False,
                                                        ow_app_var=ow_app_var
                                                        ),
                                    tex_vect=tex_vect, app_tex_vect=app_tex_vect,
                                    pltf=pltf, use_tex=use_tex, app_null=False)
    else:
        _greek_letters = get_greek_letters(
                                            app_omicron=app_omicron,
                                            ow_omicron=ow_omicron,
                                            app_bs=False, attr=attr
                                            )
        _greek_letters_symb = _str_ps.capp_tex_vect(
                                get_greek_letters(
                                                    app_omicron=app_omicron,
                                                    ow_omicron=ow_omicron,
                                                    app_bs=True, attr=attr
                                                    ),
                                tex_vect=tex_vect,
                                app_tex_vect=app_tex_vect,
                                pltf=pltf, use_tex=use_tex,
                                app_null=False
                                )

    _greek_letters_symb_dict = {_greek_letters[i]: _greek_letters_symb[i]
                                for i in range(_greek_letters.size)}

    if capital:

        roman_sub = defaults.set_default(roman_sub, default=True)

        if roman_sub:
            _greek_letters_roman_dict = calc_greek_letters_roman_dict(
                                            roman_app_mathrm=roman_app_mathrm,
                                            tex_vect=tex_vect,
                                            app_tex_vect=app_tex_vect,
                                            pltf=pltf, use_tex=use_tex,
                                            )

            for l in _greek_letters_roman_dict:     # noqa: E741
                _greek_letters_symb_dict[l] = _greek_letters_roman_dict[l]

    if app_omicron or ow_omicron:

        safe_omicron = defaults.set_default(safe_omicron, default=True)

        if safe_omicron:
            o_keys = ["o", "O", "varo"]
            omicron_keys = [f"{o}micron" for o in o_keys]
            om_keys = arr.filter(omicron_keys, a_in=keys(_greek_letters_symb_dict))

            for om in om_keys:
                _greek_letters_symb_dict[om] = _greek_letters_symb_dict[om]\
                                                    .replace("micron", "")

    return _greek_letters_symb_dict


def calc_arab_num_symb_dict(
                            tex_vect: Optional[str] = None,
                            app_tex_vect: Optional[bool] = None,
                            pltf: Optional[bool] = None,
                            use_tex: Optional[bool] = None,
                            ) -> dict:

    app_tex_vect = defaults.app_tex_vect(app_tex_vect=app_tex_vect)
    pltf = defaults.pltf(pltf=pltf)
    use_tex = defaults.use_tex(use_tex=use_tex)

    _arab_num = get_arab_num()
    _arab_num_symb = _str_ps.capp_tex_vect(
                                        get_arab_num(),
                                        tex_vect=tex_vect, app_tex_vect=app_tex_vect,
                                        pltf=pltf, use_tex=use_tex, app_null=False
                                        )

    return {_arab_num[n]: _arab_num_symb[n] for n in range(np.size(_arab_num))}


def calc_roman_letters_symb_dict(
                                    roman_app_mathrm: Optional[bool] = None,
                                    tex_vect: Optional[str] = None,
                                    app_tex_vect: Optional[bool] = None,
                                    pltf: Optional[bool] = None,
                                    use_tex: Optional[bool] = None,
                                    ) -> dict:

    app_tex_vect = defaults.app_tex_vect(app_tex_vect=app_tex_vect)
    pltf = defaults.pltf(pltf=pltf)
    use_tex = defaults.use_tex(use_tex=use_tex)

    _locals = locals()

    return {
                **calc_roman_letters_case_symb_dict(capital=False, **_locals),
                **calc_roman_letters_case_symb_dict(capital=True, **_locals),
            }


def calc_greek_letters_symb_dict(
                                    app_omicron: Optional[bool] = None,
                                    ow_omicron: Optional[bool] = None,
                                    safe_omicron: Optional[bool] = None,
                                    roman_sub: Optional[bool] = None,
                                    roman_app_mathrm: Optional[bool] = None,
                                    blank_nvar: Optional[bool] = None,
                                    app_var_nvar: Optional[bool] = None,
                                    tex_vect: Optional[str] = None,
                                    app_tex_vect: Optional[bool] = None,
                                    pltf: Optional[bool] = None,
                                    use_tex: Optional[bool] = None,
                                    ) -> dict:

    blank_nvar = defaults.set_default(blank_nvar, default=False)

    _locals = locals()

    return {
                **calc_greek_letters_case_symb_dict(capital=False, var=False,
                                                    **_locals),
                **calc_greek_letters_case_symb_dict(capital=True, var=False,
                                                    **_locals),
                **calc_greek_letters_case_symb_dict(capital=False, var=True,
                                                    **_locals),
            }


def calc_arg_numletters_symb_dict(
                                    safe_omicron: Optional[bool] = None,
                                    roman_sub: Optional[bool] = None,
                                    roman_app_mathrm: Optional[bool] = None,
                                    blank_nvar: Optional[bool] = None,
                                    app_var_nvar: Optional[bool] = None,
                                    tex_vect: Optional[str] = None,
                                    app_tex_vect: Optional[bool] = None,
                                    pltf: Optional[bool] = None,
                                    use_tex: Optional[bool] = None,
                                    ) -> dict:

    return {
                **calc_arab_num_symb_dict(tex_vect=tex_vect, app_tex_vect=app_tex_vect,
                                          pltf=pltf, use_tex=use_tex),
                **calc_greek_letters_symb_dict(
                                                app_omicron=False, ow_omicron=True,
                                                safe_omicron=safe_omicron,
                                                roman_sub=roman_sub,
                                                roman_app_mathrm=roman_app_mathrm,
                                                blank_nvar=blank_nvar,
                                                app_var_nvar=app_var_nvar,
                                                tex_vect=tex_vect,
                                                app_tex_vect=app_tex_vect,
                                                pltf=pltf, use_tex=use_tex),
                **calc_roman_letters_symb_dict(
                                                roman_app_mathrm=roman_app_mathrm,
                                                tex_vect=tex_vect,
                                                app_tex_vect=app_tex_vect,
                                                pltf=pltf, use_tex=use_tex),
            }


def sb(
        l: Optional[str] = None,
        v: Optional[bool] = None,
        safe_omicron: Optional[bool] = None,
        roman_sub: Optional[bool] = None,
        roman_app_mathrm: Optional[bool] = None,
        blank_nvar: Optional[bool] = None,
        tex_vect: Optional[str] = None,
        pltf: Optional[bool] = None,
        use_tex: Optional[bool] = None,
        ) -> str:

    use_tex = defaults.use_tex(use_tex=use_tex)

    if l is None:
        s = defaults.null()
    else:
        s = calc_arg_numletters_symb_dict(
                                            safe_omicron=safe_omicron,
                                            roman_sub=roman_sub,
                                            roman_app_mathrm=roman_app_mathrm,
                                            blank_nvar=blank_nvar, tex_vect=tex_vect,
                                            app_tex_vect=v, pltf=pltf, use_tex=use_tex,
                                            )[l]

    return s


def sbv(
        l: Optional[str] = None,
        v: Optional[bool] = None,
        safe_omicron: Optional[bool] = None,
        roman_sub: Optional[bool] = None,
        roman_app_mathrm: Optional[bool] = None,
        blank_nvar: Optional[bool] = None,
        tex_vect: Optional[str] = None,
        pltf: Optional[bool] = None,
        use_tex: Optional[bool] = None,
        ) -> str:

    v = defaults.set_default(v, default=True)

    return sb(**locals())


def pltsb(
            l: Optional[str] = None,
            v: Optional[bool] = None,
            safe_omicron: Optional[bool] = None,
            roman_sub: Optional[bool] = None,
            roman_app_mathrm: Optional[bool] = None,
            blank_nvar: Optional[bool] = None,
            tex_vect: Optional[str] = None,
            pltf: Optional[bool] = None,
            use_tex: Optional[bool] = None,
            ):

    pltf = defaults.set_default(pltf, default=True)

    return sb(**locals())


def pltsbv(
            l: Optional[str] = None,
            v: Optional[bool] = None,
            safe_omicron: Optional[bool] = None,
            roman_sub: Optional[bool] = None,
            roman_app_mathrm: Optional[bool] = None,
            blank_nvar: Optional[bool] = None,
            tex_vect: Optional[str] = None,
            pltf: Optional[bool] = None,
            use_tex: Optional[bool] = None,
            ) -> str:

    v = defaults.set_default(v, default=True)

    return pltsb(**locals())


def calc_arab_num_array(
                        tex_vect: Optional[str] = None,
                        pltf: Optional[bool] = None,
                        use_tex: Optional[bool] = None,
                        ) -> np.ndarray:

    pltf = defaults.set_default(pltf, default=True)

    num_array = np.transpose(np.array([
            _str_ps.capp_dlrs(vals(d=calc_arab_num_symb_dict(
                                                    tex_vect=tex_vect,
                                                    app_tex_vect=bool(i % 2),
                                                    pltf=pltf,
                                                    use_tex=use_tex,
                                                    ), array=True)) for i in range(2)]))

    return num_array


def calc_roman_letters_array(
                                roman_app_mathrm: Optional[bool] = None,
                                tex_vect: Optional[str] = None,
                                pltf: Optional[bool] = None,
                                use_tex: Optional[bool] = None,
                                ) -> np.ndarray:

    pltf = defaults.set_default(pltf, default=True)

    letters_array = np.transpose(np.array([
            _str_ps.capp_dlrs(vals(d=calc_roman_letters_case_symb_dict(
                                                    roman_app_mathrm=roman_app_mathrm,
                                                    capital=bool(i//2),
                                                    tex_vect=tex_vect,
                                                    app_tex_vect=bool(i % 2),
                                                    pltf=pltf,
                                                    use_tex=use_tex,
                                                    ), array=True)) for i in range(4)]))

    return letters_array


def calc_greek_letters_array(
                                app_omicron: Optional[bool] = None,
                                ow_omicron: Optional[bool] = None,
                                roman_app_mathrm: Optional[bool] = None,
                                blank_nvar: Optional[bool] = None,
                                tex_vect: Optional[str] = None,
                                pltf: Optional[bool] = None,
                                use_tex: Optional[bool] = None,
                                ) -> np.ndarray:

    pltf = defaults.pltf(pltf)

    letters_array = np.transpose(np.array([
            _str_ps.capp_dlrs(
                            vals(
                                    d=calc_greek_letters_case_symb_dict(
                                                    app_omicron=app_omicron,
                                                    ow_omicron=ow_omicron,
                                                    roman_sub=None,
                                                    roman_app_mathrm=roman_app_mathrm,
                                                    capital=bool((i//2) == 1),
                                                    var=bool((i//2) == 2),
                                                    blank_nvar=blank_nvar,
                                                    tex_vect=tex_vect,
                                                    app_tex_vect=bool(i % 2),
                                                    pltf=pltf, use_tex=use_tex
                                                    ),
                                    array=True,
                                )
                            ) for i in range(6)]))

    return letters_array


def calc_arg_numletters_array(
                                app_omicron: Optional[bool] = None,
                                ow_omicron: Optional[bool] = None,
                                roman_app_mathrm: Optional[bool] = None,
                                blank_nvar: Optional[bool] = None,
                                tex_vect: Optional[str] = None,
                                pltf: Optional[bool] = None,
                                use_tex: Optional[bool] = None,
                                order_arg: Optional[Union[np.ndarray, list,]] = None,
                                ) -> np.ndarray:

    pltf = defaults.set_default(pltf, default=True)

    order_arg = defaults.order_arg(order_arg)

    _arg_array = {
                "arab": calc_arab_num_array(
                                            tex_vect=tex_vect,
                                            pltf=pltf, use_tex=use_tex,
                                            ),
                "roman": calc_roman_letters_array(
                                                    roman_app_mathrm=roman_app_mathrm,
                                                    tex_vect=tex_vect,
                                                    pltf=pltf, use_tex=use_tex,
                                                    ),
                "greek": calc_greek_letters_array(
                                                    app_omicron=app_omicron,
                                                    ow_omicron=ow_omicron,
                                                    roman_app_mathrm=roman_app_mathrm,
                                                    blank_nvar=blank_nvar,
                                                    tex_vect=tex_vect,
                                                    pltf=pltf, use_tex=use_tex,
                                                    )
                }

    return np.array(pad([_arg_array[o] for o in order_arg]))


def calc_greek_roman_tex_preamble_print_lines(
                                                app_omicron: Optional[bool] = None,
                                                def_Greek: Optional[bool] = None,
                                            ) -> np.ndarray:

    app_omicron = defaults.set_default(app_omicron, True)
    def_Greek = defaults.set_default(def_Greek, True)

    _greek_letters = get_greek_letters(app_omicron=app_omicron,
                                       ow_omicron=False,
                                       incl_var=True)
    _greek_letters_roman_dict = calc_greek_letters_roman_dict(roman_app_mathrm=True)

    tex_preamble_pack = [
                            r"\usepackage{bm} % bold math",
                            r"\usepackage{upgreek} % non-italic greek",
                            "",
                            ]
    tex_preamble_vect = [
                            r"\def\vect#1{{",
                            r"\bm{#1}}}",
                            "",
                            ]

    tex_preamble_vect_greek = [f"\\let\\{l}\\up{l}"
                               for l in arr.filter(_greek_letters,  # noqa: E741
                                                   a_rm=["o", "omicron"])]

    if app_omicron:
        tex_preamble_omicron = np.concatenate([[
                                f"\\def\\{o}micron{{{{\\{o}}}}}" for o in ["o", "O"]
                            ], [""]])
    else:
        tex_preamble_omicron = []

    if def_Greek:
        tex_preamble_Greek = np.concatenate([[
                                f"\\def\\{l}{{{{{_greek_letters_roman_dict[l]}}}}}"
                                for l in _greek_letters_roman_dict  # noqa: E741
                            ], [""]])
    else:
        tex_preamble_Greek = []

    tex_preamble = np.concatenate([
                                    tex_preamble_pack,
                                    tex_preamble_omicron,
                                    tex_preamble_Greek,
                                    arr.iterarr(tex_preamble_vect[0]),
                                    tex_preamble_vect_greek,
                                    arr.iterarr(tex_preamble_vect[1:]),
                                    ])

    return tex_preamble


def calc_greek_roman_tex_preamble(
                                    app_omicron: Optional[bool] = None,
                                    def_Greek: Optional[bool] = None,
                                    ) -> str:

    return "\n".join([
                        *calc_greek_roman_tex_preamble_print_lines(
                            app_omicron=app_omicron,
                            def_Greek=def_Greek),
                        "",
                        ])
