"""
Functions module for linalg.polynom module.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union
import itertools

# External packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as _PolynomialFeatures

# Local
from mftsci.utils.arr import N_index
from mftsci.linalg.polynom import defaults   # noqa: F401


# =====================================
# CLASSES
# =====================================

class PolynomialFeatures(_PolynomialFeatures):
    """
    Generate polynomial and interaction features with extended constraints.

    Parameters
    ----------

    mono_degree_mm : np.ndarray or None
        Min and max allowed degree of each feature

    max_degree : int or None, default=None
        Maximum allowed total degree.

    min_degree : int or None, default=None
        Minimum total degree for generated polynomials. If 0, includes bias term.

    max_mono_degree : int or None, default=None
        Maximum allowed degree for any single feature in a term.

    min_mono_degree : int or None, default=None
        Minimum allowed degree for any single feature in a term.

    max_nonzero : int or None, default=None
        Maximum number of non-zero exponents in a term.

    min_nonzero : int or None, default=None
        Maximum number of non-zero exponents in a term.

    force_bias : bool or None, default=None
        Force bias term

    Attributes
    ----------
    powers_ : ndarray of shape (n_output_features, n_input_features)
        The exponents for each term in the output features.
    """

    @staticmethod
    def degree_kwargs_props():

        props = {
                    **dict(zip(_degree_kw, "_" + _degree_kw)),
                    **dict(zip(_degree_pkw, _degree_pkw)),
                    **{
                        "powers": "powers",
                        "mono_degree_mm": "mono_degree_mm",
                        "n_features": "n_features_in_",
                        "input_features": "feature_names_in_",
                        },
                }

        return props

    @staticmethod
    def degree_kwargs_ow():

        props = {
                    **dict(zip(_degree_kw[:2], "_" + _degree_kw[:2])),
                    **{
                        "n_features": "n_features_in_",
                        "input_features": "feature_names_in_",
                        },
                }

        return props

    def _html_repr(self):
        return repr(self)

    def __init__(
            self,
            max_degree: Optional[int] = None,
            min_degree: Optional[int] = None,
            max_nonzero: Optional[int] = None,
            min_nonzero: Optional[int] = None,
            max_mono_degree: Optional[int] = None,
            min_mono_degree: Optional[int] = None,
            include_bias: Optional[bool] = None,
            interaction_only: Optional[bool] = None,
            powers: Optional[Union[
                        np.ndarray, pd.DataFrame, pd.Series, pd.Index,
                        ]] = None,
            mono_degree_mm: Optional[Union[
                                np.ndarray, pd.DataFrame, pd.Series,
                                ]] = None,
            input_features: Optional[Union[
                pd.Index, pd.Series, np.ndarray, list, object,
                ]] = None,
            n_features: Optional[int] = None,
            ):

        degree_kwargs = calc_powers(
                            powers=powers,
                            max_degree=max_degree,
                            min_degree=min_degree,
                            max_nonzero=max_nonzero,
                            min_nonzero=min_nonzero,
                            max_mono_degree=max_mono_degree,
                            min_mono_degree=min_mono_degree,
                            include_bias=include_bias,
                            interaction_only=interaction_only,
                            mono_degree_mm=mono_degree_mm,
                            n_features=n_features,
                            input_features=input_features,
                            as_kwargs=True,
                            app_sort=True,
                            as_index=False,
                            as_pd=False,
                            )

        super().__init__(
                    degree=degree_kwargs["max_degree"],
                    **{k: degree_kwargs[k] for k in _degree_pkw}
                    )

        for k in _degree_kw:
            setattr(self, k, eval(k))

        for k in _degree_ikw:
            setattr(self, k, degree_kwargs.get(k, None))

        props = self.degree_kwargs_ow()
        for k in props:
            setattr(self, props[k], degree_kwargs.get(k, None))
        if self.feature_names_in_ is None:
            del self.feature_names_in_

        self._n_out_full = self._num_combinations(
                                    n_features=self.n_features_in_,
                                    min_degree=self._min_degree,
                                    max_degree=self._max_degree,
                                    interaction_only=self.interaction_only,
                                    include_bias=self.include_bias
                                    )
        self.n_output_features_ = self._n_out_full
        self.n_toutput_features_ = self.powers.shape[0]

    def fit(
            self,
            X,
            y=None,
            max_degree: Optional[int] = None,
            min_degree: Optional[int] = None,
            max_nonzero: Optional[int] = None,
            min_nonzero: Optional[int] = None,
            max_mono_degree: Optional[int] = None,
            min_mono_degree: Optional[int] = None,
            include_bias: Optional[bool] = None,
            interaction_only: Optional[bool] = None,
            powers: Optional[Union[
                        np.ndarray, pd.DataFrame, pd.Series, pd.Index,
                        ]] = None,
            mono_degree_mm: Optional[Union[
                                np.ndarray, pd.DataFrame, pd.Series,
                                ]] = None,
            ):

        super().fit(
                    X=X,
                    y=y,
                    )

        n_features = self.n_features_in_
        input_features = getattr(self, "feature_names_in_")
        match_i = self.powers.shape[1] == n_features
        if match_i:
            match_features = np.all(input_features == self.input_features)
            if powers is None:
                if match_features:
                    powers = self.powers
                else:
                    powers = self.powers.to_frame().values
            if mono_degree_mm is None:
                mono_degree_mm = self.mono_degree_mm

        kwargs = {}
        for k in [*_degree_kw, *_degree_pkw]:
            if eval(k) is not None:
                setattr(self, k, eval(k))
            kwargs[k] = getattr(self, k, None)

        for k in _degree_ikw:
            kwargs[k] = eval(k)

        degree_kwargs = calc_powers(
                            **kwargs,
                            as_kwargs=True,
                            app_sort=True,
                            as_index=False,
                            as_pd=False,
                            )

        for k in [*_degree_ikw, *_degree_pkw]:
            setattr(self, k, degree_kwargs.get(k, None))

        props = self.degree_kwargs_ow()
        for k in props:
            setattr(self, props[k], degree_kwargs.get(k, None))
        if self.feature_names_in_ is None:
            del self.feature_names_in_

        self._n_out_full = self._num_combinations(
                                    n_features=self.n_features_in_,
                                    min_degree=self._min_degree,
                                    max_degree=self._max_degree,
                                    interaction_only=self.interaction_only,
                                    include_bias=self.include_bias
                                    )
        self.n_output_features_ = self._n_out_full
        self.n_toutput_features_ = self.powers.shape[0]

    @property
    def powers_index_(self):

        return pd.MultiIndex.from_arrays(self.powers.T, names=self.input_features)

    @property
    def powers_findex_(self):

        return pd.MultiIndex.from_arrays(self.powers_.T, names=self.input_features)

    @property
    def _tcombinations(self):
        return map(
                lambda row: tuple([i for i in range(len(row)) for k in range(row[i])]),
                self.powers,
                )

    def ftransform(
            self,
            X,
            as_pd: Optional[bool] = None,
            ):

        as_pd = defaults.set_default(as_pd, self.input_features is not None)
        X_t = pd.DataFrame(
                self.transform(X), columns=self.powers_findex_
                )[self.powers_index_]
        if as_pd:
            return X_t
        else:
            return X_t.values

    def stransform(
            self,
            X,
            as_pd: Optional[bool] = None,
            ):

        as_pd = defaults.set_default(as_pd, self.input_features is not None)
        return stransform(
                X=X, powers=self.powers,
                input_features=self.input_features, as_pd=as_pd,
                )

    def get_feature_names_tout(
                self,
                input_features=None,
                as_pd: Optional[bool] = None,
                ):

        as_pd = defaults.set_default(as_pd, self.input_features is not None)
        feature_names_out = pd.Series(
                                super().get_feature_names_out(input_features),
                                index=self.powers_findex_
                                ).loc[self.powers_index_]

        if as_pd:
            return feature_names_out
        else:
            return feature_names_out.values


# =====================================
# FUNCTIONS
# =====================================


def format_degree_df(
        df: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        input_features: Optional[Union[
                        pd.Index, pd.Series, np.ndarray, list, object,
                        ]] = None,
        n_features: Optional[int] = None,
        ):

    if df is not None:
        if hasattr(df, "values"):
            if isinstance(df, pd.Index):
                df = df.to_frame()
            elif isinstance(df, pd.Series):
                df = df.to_frame().T
            if input_features is None:
                input_features = df.columns
            else:
                df = df[input_features]
            df = df.values
        else:
            df = np.atleast_2d(df)
            if n_features is not None:
                df = df[:, :n_features]
        n_features = df.shape[1]

    return df, input_features, n_features


def calc_degree_kwargs(
        max_degree: Optional[int] = None,
        min_degree: Optional[int] = None,
        max_nonzero: Optional[int] = None,
        min_nonzero: Optional[int] = None,
        max_mono_degree: Optional[int] = None,
        min_mono_degree: Optional[int] = None,
        include_bias: Optional[bool] = None,
        interaction_only: Optional[bool] = None,
        powers: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        mono_degree_mm: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        input_features: Optional[Union[
                        pd.Index, pd.Series, np.ndarray, list, object,
                        ]] = None,
        n_features: Optional[int] = None,
        export_ikw: Optional[bool] = None,
        ):

    mm = ["min", "max"]
    mm_ineq = ["less", "greater"]

    # has_ikw = {k: eval(f"{k} is not None") for k in _degree_ikw}
    has_powers = powers is not None
    has_mono_degree_mm = mono_degree_mm is not None
    has_input_features = input_features is not None
    has_n_features = n_features is not None

    has_ikw = has_powers | has_mono_degree_mm | has_input_features | has_n_features

    export_ikw = defaults.set_default(
                                export_ikw, has_ikw,
                                )

    powers, input_features, n_features = format_degree_df(
                                            powers, input_features, n_features
                                            )
    mono_degree_mm, input_features, n_features = format_degree_df(
                                            mono_degree_mm, input_features, n_features
                                            )
    has_input_features = input_features is not None

    if has_powers:
        pow_mono_degree_mm = np.array([
                                getattr(powers, m)(axis=0)
                                for m in mm
                                ])
        if has_mono_degree_mm:
            for k in range(2):
                mono_degree_mm[k, :] = np.where(
                                    getattr(np, mm_ineq[k])(
                                            pow_mono_degree_mm[k, :],
                                            mono_degree_mm[k, :]
                                            ),
                                    mono_degree_mm[k, :],
                                    pow_mono_degree_mm[k, :]
                                    )
        else:
            mono_degree_mm = pow_mono_degree_mm
            has_mono_degree_mm = True

    if not has_mono_degree_mm:
        if not has_input_features:
            if not has_n_features:
                n_features = 1
        else:
            input_features = np.atleast_1d(input_features).reshape(-1)
            n_features = input_features.shape[0]

    min_degree = defaults.min_degree(min_degree)
    if min_mono_degree is None:
        min_mono_degree = defaults.min_degree(min_mono_degree)
        if mono_degree_mm is not None:
            if mono_degree_mm.shape[0] > 1:
                min_mono_degree = np.min(mono_degree_mm[0])

    if max_mono_degree is None:
        if mono_degree_mm is None:
            if max_degree is not None:
                max_mono_degree = max_degree
            else:
                max_mono_degree = defaults.max_degree(max_mono_degree)
        else:
            max_mono_degree = np.max(mono_degree_mm)

    if max_degree is None:
        max_degree = n_features * max_mono_degree
    else:
        max_mono_degree = np.min([max_degree, max_mono_degree])

    mm_rng = [min_mono_degree, max_mono_degree]
    if export_ikw:
        if mono_degree_mm is None:
            mono_degree_mm = np.repeat(mm_rng, n_features).reshape(2, n_features)
        else:
            if mono_degree_mm.shape[0] == 1:
                mono_degree_mm = np.array([
                                np.full(n_features, min_mono_degree),
                                mono_degree_mm[0],
                                ])

            for k in range(2):
                mono_degree_mm[k, :] = np.where(
                                    getattr(np, mm_ineq[k])(
                                        mono_degree_mm[k, :],
                                        mm_rng[k]
                                        ),
                                    mm_rng[k],
                                    mono_degree_mm[k, :]
                                    )

    interaction_only = defaults.set_default(interaction_only, False)
    if interaction_only:
        if min_nonzero is None:
            min_nonzero = 2
    if min_nonzero is not None:
        if min_nonzero > 1:
            interaction_only = True

    include_bias = defaults.set_default(include_bias, True)
    if min_degree <= 0:
        include_bias = True

    if export_ikw:
        kw = [
                *_degree_kw,
                *_degree_pkw,
                *_degree_ikw,
                ]
    else:
        kw = [
                *_degree_kw,
                *_degree_pkw,
                ]
    kwargs = {}
    for k in kw:
        kwargs[k] = eval(k)

    return kwargs


def check_powers(
                powers: Union[pd.DataFrame, list, tuple,],
                max_degree: Optional[int] = None,
                min_degree: Optional[int] = None,
                max_nonzero: Optional[int] = None,
                min_nonzero: Optional[int] = None,
                max_mono_degree: Optional[int] = None,
                min_mono_degree: Optional[int] = None,
                include_bias: Optional[bool] = None,
                ):

    powers = np.atleast_1d(powers)
    total_degree = powers.sum(axis=-1)
    nonzero = (powers != 0).sum(axis=-1)
    check = np.ones(total_degree.shape, dtype=bool)

    if max_degree is not None:
        check &= (total_degree <= max_degree)
    if min_degree is not None:
        check &= (total_degree >= min_degree)
    if max_nonzero is not None:
        check &= (nonzero <= max_nonzero)
    if min_nonzero is not None:
        check &= (nonzero >= min_nonzero)
    if max_mono_degree is not None:
        check &= (powers <= max_degree).all(axis=-1)
    if min_mono_degree is not None:
        check &= (powers >= min_degree).all(axis=-1)
    if include_bias:
        check[(powers == 0).all(axis=-1)] = True

    return check


def calc_powers(
            powers: Optional[np.ndarray] = None,
            max_degree: Optional[int] = None,
            min_degree: Optional[int] = None,
            max_nonzero: Optional[int] = None,
            min_nonzero: Optional[int] = None,
            max_mono_degree: Optional[int] = None,
            min_mono_degree: Optional[int] = None,
            include_bias: Optional[bool] = None,
            interaction_only: Optional[bool] = None,
            mono_degree_mm: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            input_features: Optional[Union[
                pd.Index, pd.Series, np.ndarray, list, object,
                ]] = None,
            n_features: Optional[int] = None,
            app_sort: Optional[bool] = None,
            as_kwargs: Optional[bool] = None,
            as_index: Optional[bool] = None,
            as_pd: Optional[bool] = None,
            ) -> np.ndarray:

    was_index = isinstance(powers, pd.Index)
    was_pd = hasattr(powers, "values") and not was_index
    as_index = defaults.set_default(as_index, was_index)
    as_pd = defaults.set_default(as_pd, was_pd)
    as_kwargs = defaults.set_default(as_kwargs, False)

    kwargs = calc_degree_kwargs(
                                max_degree=max_degree,
                                min_degree=min_degree,
                                max_nonzero=max_nonzero,
                                min_nonzero=min_nonzero,
                                max_mono_degree=max_mono_degree,
                                min_mono_degree=min_mono_degree,
                                include_bias=include_bias,
                                interaction_only=interaction_only,
                                powers=powers,
                                mono_degree_mm=mono_degree_mm,
                                n_features=n_features,
                                input_features=input_features,
                                export_ikw=True,
                                )
    ckwargs = {k: kwargs[k] for k in [*_degree_kw, "include_bias"]}

    if kwargs["powers"] is None:
        ckwargs["max_mono_degree"] = None
        ckwargs["min_mono_degree"] = None

        mono_degree_mm = kwargs["mono_degree_mm"].copy()
        mono_degree_mm[1, :] += 1

        kwargs["powers"] = np.array([
                    mono[::-1]
                    for mono in itertools.product(
                                    *[
                                        range(*mono_degree_mm[:, k])
                                        for k in np.arange(
                                            mono_degree_mm.shape[1]
                                            )[::-1]
                                        ]
                                    )
                    if check_powers(powers=mono, **ckwargs)
                    ])
    else:
        if kwargs["powers"].shape[0] > 0:
            kwargs["powers"] = kwargs["powers"][
                                    check_powers(powers=kwargs["powers"], **ckwargs)
                                    ]

    kwargs["powers"] = sort_powers(kwargs["powers"], app_sort=app_sort)

    if as_index:
        kwargs["powers"] = pd.MultiIndex.from_arrays(
                                kwargs["powers"].T,
                                names=kwargs["input_features"],
                                )
    elif as_pd:
        kwargs["powers"] = pd.DataFrame(
                            kwargs["powers"],
                            columns=kwargs["input_features"],
                            )

    if as_kwargs:
        return kwargs
    else:
        return kwargs["powers"]


def sort_powers(
            powers: Optional[np.ndarray] = None,
            app_sort: Optional[bool] = None,
            ) -> np.ndarray:

    app_sort = defaults.set_default(app_sort, True)
    if app_sort:
        if powers.ndim == 2 and powers.shape[0] > 1:
            sort_N_shape = 1 + np.max(powers[:, ::-1], axis=0)
            sort_N = N_index(shape=sort_N_shape, index=powers[:, ::-1])
            sort_N += np.sum(powers, axis=1) * np.prod(sort_N_shape)
            powers = powers[np.argsort(sort_N)]

    return powers


def stransform(
        X: Union[np.ndarray, pd.DataFrame],
        powers: Optional[np.ndarray] = None,
        max_degree: Optional[int] = None,
        min_degree: Optional[int] = None,
        max_nonzero: Optional[int] = None,
        min_nonzero: Optional[int] = None,
        max_mono_degree: Optional[int] = None,
        min_mono_degree: Optional[int] = None,
        include_bias: Optional[bool] = None,
        interaction_only: Optional[bool] = None,
        mono_degree_mm: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        input_features: Optional[Union[
            pd.Index, pd.Series, np.ndarray, list, object,
            ]] = None,
        n_features: Optional[int] = None,
        app_sort: Optional[bool] = None,
        as_pd: Optional[bool] = None,
        ):

    was_pd = hasattr(X, "values")
    as_pd = defaults.set_default(as_pd, was_pd)
    if was_pd:
        if isinstance(X, pd.Series):
            X = X.to_frame()
        input_features = X.columns
        index = X.index
        X = X.values
    else:
        index = None
        X = np.atleast_1d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    n_features = X.shape[1]

    powers = calc_powers(
                        powers=powers,
                        max_degree=max_degree,
                        min_degree=min_degree,
                        max_nonzero=max_nonzero,
                        min_nonzero=min_nonzero,
                        max_mono_degree=max_mono_degree,
                        min_mono_degree=min_mono_degree,
                        include_bias=include_bias,
                        interaction_only=interaction_only,
                        mono_degree_mm=mono_degree_mm,
                        n_features=n_features,
                        input_features=input_features,
                        app_sort=app_sort,
                        as_kwargs=False,
                        as_index=True,
                        )

    XP = np.array([
                np.prod(X**i, axis=1)
                for i in powers.to_frame().values
                ]).T

    if as_pd:
        XP = pd.DataFrame(XP, columns=powers, index=index)

    return XP


_degree_kw = np.array([
            "max_degree",
            "min_degree",
            "max_nonzero",
            "min_nonzero",
            "max_mono_degree",
            "min_mono_degree",
            ])

_degree_pkw = np.array([
            "include_bias",
            "interaction_only",
            ])

_degree_ikw = np.array([
        "powers",
        "mono_degree_mm",
        "input_features",
        "n_features",
        ])
