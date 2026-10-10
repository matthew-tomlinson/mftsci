"""
Functions module for regr utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import numpy as np
import pandas as pd
import scipy


# Internal packages
from mftsci.utils.regr import defaults


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================

def clean_X(
        X: Union[np.ndarray, pd.DataFrame, pd.Series],
        as_pd: Optional[bool] = None,
        ) -> Union[np.ndarray, pd.Series]:

    if isinstance(X, pd.Series):
        X = X.to_frame()
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            s2 = 1 if X.ndim == 1 else int(np.prod(X.shape[1:]))
            X.reshape(X.shape[0], s2)

    was_pd = hasattr(X, "values")
    if as_pd is None:
        as_pd = was_pd

    if as_pd and not was_pd:
        X = pd.DataFrame(X)
    elif not as_pd and was_pd:
        X = X.values

    return X


def clean_y(
        y: Union[np.ndarray, pd.Series],
        as_pd: Optional[bool] = None,
        ) -> Union[np.ndarray, pd.Series]:

    was_pd = hasattr(y, "values")
    if as_pd is None:
        as_pd = was_pd

    if as_pd and not was_pd:
        y = pd.Series(y)
    elif not as_pd and was_pd:
        y = y.values

    return y


def clean_W(
        W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
        as_pd: Optional[bool] = None,
        as_matrix: Optional[bool] = None,
        ) -> Union[np.ndarray, pd.DataFrame, pd.Series,]:

    if W is not None:
        was_pd = hasattr(W, "values")
        if as_pd is None:
            as_pd = was_pd
        was_matrix = W.ndim == 2
        if as_matrix is None:
            as_matrix = was_matrix

        if was_matrix != as_matrix:
            if as_pd:
                if was_pd:
                    index = W.index
                else:
                    index = None
            W = np.diag(W)
            if as_pd:
                if as_matrix:
                    W = pd.DataFrame(W, index=index, columns=index)
                else:
                    W = pd.Series(W, index=index)
        elif was_pd and not as_pd:
            W = W.values
        elif not was_pd and as_pd:
            if W.ndim == 2:
                W = pd.DataFrame(W)
            else:
                W = pd.Series(W)

    return W


def capp_bias(
            X: Union[np.ndarray, pd.DataFrame, pd.Series],
            app_bias: Optional[bool] = None,
            check_bias: Optional[bool] = None,
            bias_name: Optional[str] = None,
            as_pd: Optional[bool] = None,
            ):

    X = clean_X(X=X, as_pd=as_pd)
    is_pd = hasattr(X, "values")

    app_bias = defaults.app_bias(app_bias)
    check_bias = defaults.check_bias(check_bias)
    bias_name = defaults.bias_name(bias_name)

    if app_bias and check_bias:
        if is_pd:
            app_bias = not (X.iloc[:, 0] == 1).all()
        else:
            app_bias = not np.all(X[:, 0] == 1)

    if app_bias:
        if is_pd:
            X = pd.concat([
                    pd.Series(1.0, index=X.index, name=bias_name),
                    X
                    ], axis=1)
        else:
            X = np.concat([
                            np.ones((X.shape[0], 1), dtype=float),
                            X,
                            ], axis=1)

    return X


def prep_y_X_W(
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
        as_pd: Optional[bool] = None,
        save_ic: Optional[bool] = None,
        as_matrix: Optional[bool] = None,
        app_bias: Optional[bool] = None,
        check_bias: Optional[bool] = None,
        bias_name: Optional[str] = None,
        valid_only: Optional[bool] = None,
        ret_all: Optional[bool] = None,
        ) -> tuple:

    ret_all = defaults.set_default(ret_all, True)

    y_is_pd = hasattr(y, "values")
    if X is None:
        if y_is_pd:
            X = y.apply(lambda v: 1.0).to_frame()
        else:
            X = np.ones((y.shape[0], 1), dtype=float)
    X_is_pd = hasattr(X, "values")

    W_is_pd = hasattr(W, "values")
    if as_pd is None:
        as_pd = X_is_pd or y_is_pd

    if not ret_all:
        save_ic = False
    else:
        if save_ic is None:
            save_ic = True
        if as_pd:
            save_ic = True

    columns = None
    index = None
    if y_is_pd:
        index = y.index
    elif X_is_pd:
        index = X.index
    elif W_is_pd:
        index = W.index

    if index is not None:
        if y_is_pd:
            y = y.loc[index]
        if X_is_pd:
            X = X.loc[index]
        if W_is_pd:
            if W.ndim == 2:
                W = W.loc[index, index]
            else:
                W = W.loc[index]
    y = clean_y(y=y, as_pd=as_pd)
    valid = ~(np.isinf(y) | np.isnan(y))
    W = clean_W(W=W, as_pd=as_pd, as_matrix=as_matrix)
    X = capp_bias(
                X=X,
                app_bias=app_bias,
                check_bias=check_bias,
                bias_name=bias_name,
                as_pd=True,
                )

    if save_ic:
        columns = getattr(X, "columns", None)
    else:
        index = None
    if not as_pd:
        X = X.values

    if defaults.set_default(valid_only, False):
        y = y[valid]
        X = X[valid]
        if W is not None:
            if W.ndim == 2:
                if hasattr(W, "values"):
                    W = W.loc[valid, valid]
                else:
                    W = W[valid, valid]
            else:
                W = W[valid]

        if ret_all:
            valid = valid[valid]
            index = index[valid]

    if defaults.set_default(ret_all, True):
        return y, X, W, valid, index, columns
    else:
        return y, X, W


def calc_df(
        N_data: int,
        N_params: int,
        ) -> int:
    """
    Calculate degrees of freedom.
    """

    return N_data - N_params


def calc_N_data_params(
            N_data: Optional[int] = None,
            N_params: Optional[int] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y: Optional[Union[np.ndarray, pd.Series]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_resid: Optional[Union[np.ndarray, pd.Series]] = None,
            beta: Optional[Union[np.ndarray, pd.Series]] = None,
            ):
    """
    Calculate sample size N_data and parameter dimensions N_params

    """
    if X is not None:
        N_data,  N_params = X.shape[:2]
    else:
        if beta is not None:
            N_params = beta.shape[0]

        if y is not None:
            N_data = y.shape[0]
        elif y_resid is not None:
            N_data = y_resid.shape[0]
        elif W is not None:
            N_data = W.shape[0]

    return N_data, N_params


def f_X_i_max(
        X: Union[pd.DataFrame, pd.Series, np.ndarray,],
        func: Callable = None,
        i_max: Optional[int] = None,
        ):

    if i_max is None:
        N_batch = 1
    else:
        N_rows = len(X)
        N_batch = 1 + int((N_rows - 1) // i_max)

    if N_batch == 1:
        y = func(X)
    else:
        was_pd = hasattr(X, "values")
        if was_pd:
            y = pd.concat(
                    [func(X.iloc[k*i_max:(k+1)*i_max]) for k in range(N_batch)],
                    axis=0, ignore_index=True,
                    )
            y.index = X.index
        else:
            y = np.concat(
                    [func(X[k*i_max:(k+1)*i_max]) for k in range(N_batch)],
                    axis=0,
                    )

    return y


def calc_D_dist_bounds(
        D_dist=None,
        D_bounds=None,
        input_features=None,
        safe_alpha=None,
        ):

    safe_alpha = defaults.safe_alpha(safe_alpha) / 2

    if isinstance(D_dist, dict):
        D_dist = pd.Series(D_dist)

    if input_features is None:
        if D_dist is not None:
            input_features = D_dist.index
        elif D_bounds is not None:
            input_features = getattr(D_bounds, "columns", None)
            if input_features is None:
                input_features = np.arange(D_bounds.shape[1])

    n_features = len(input_features)

    if D_dist is not None:
        D_dist = D_dist.loc[input_features]

    if D_bounds is not None:
        if hasattr(D_bounds, "columns"):
            D_bounds = D_bounds[input_features]
        else:
            D_bounds = D_bounds[:, :n_features]
            D_bounds = pd.DataFrame(D_bounds, input_features[:D_bounds.shape[1]])
    else:
        if D_dist is None:
            D_bounds = pd.DataFrame(
                            np.repeat([-1, 1], n_features).reshape(2, n_features),
                            columns=input_features,
                            )
        else:
            D_bounds = pd.DataFrame()

    if D_dist is None:
        D_dist = D_bounds.apply(
                        lambda v:
                        scipy.stats.uniform(v[0], v[1] - v[0]),
                        axis=0)

    D_bounds_dist_0 = D_dist.apply(lambda v: pd.Series(v.ppf([0, 1]))).T
    D_bounds_dist_safe = D_dist.apply(
                            lambda v:
                            pd.Series(v.ppf([safe_alpha, 1 - safe_alpha]))
                            ).T
    D_bounds_dist = pd.DataFrame(
                        np.where(np.isinf(D_bounds), D_bounds_dist_safe, D_bounds),
                        index=D_bounds_dist_0.index, columns=D_bounds_dist_0.columns,
                        )

    D_bounds = D_bounds.combine_first(D_bounds_dist).loc[
                                D_bounds_dist.index, D_bounds_dist.columns,
                                ]

    mm_fun = [np.less, np.greater]
    for k, fun in enumerate(mm_fun):
        D_bounds.loc[k] = np.where(
                            fun(D_bounds.loc[k],  D_bounds_dist_0.loc[k]),
                            D_bounds_dist.loc[k],
                            D_bounds.loc[k],
                            )

    return D_dist, D_bounds


def calc_X_samp(
        D_dist: Optional[Union[pd.Series, np.ndarray, list,]] = None,
        D_bounds: Optional[Union[pd.DataFrame, np.ndarray,]] = None,
        input_features: Optional[Union[pd.Series, np.ndarray, list,]] = None,
        safe_alpha: Optional[float] = None,
        N_samp: Optional[int] = None,
        N_cline: Optional[int] = None,
        ):

    N_samp = defaults.N_samp(N_samp)
    N_cline = defaults.N_cline(N_cline)
    if N_cline < 0:
        N_cline = 21

    D_dist, D_bounds = calc_D_dist_bounds(
                            D_dist=D_dist,
                            D_bounds=D_bounds,
                            input_features=input_features,
                            safe_alpha=safe_alpha,
                            )

    if N_cline == 0:
        X_samp = D_dist.apply(lambda v: pd.Series(v.rvs(N_samp))).T
        X_samp.index = X_samp.index.set_names(["X_i"])
    else:
        X_samp = pd.DataFrame()
        X_j_index = pd.MultiIndex.from_product([
                            [""], np.arange(N_samp), np.arange(N_cline)
                            ], names=["c", "rc_i", "c_i"]
                            )

        for j, c in enumerate(D_dist.index):
            rc = D_bounds.columns.drop(c)
            c_coords = pd.Series(
                            np.linspace(*D_bounds[c].values, N_cline),
                            name=c,
                            ).to_frame()
            rc_coords = D_dist.loc[rc].apply(lambda v: pd.Series(v.rvs(N_samp))).T
            _X_j = rc_coords.merge(c_coords, how="cross")[D_bounds.columns]
            _X_j.index = X_j_index
            _X_j = _X_j.rename(index={"": c}, level=0)

            X_samp = pd.concat([X_samp, _X_j], axis=0)

    return X_samp


def calc_y_samp(
        X_samp: pd.DataFrame,
        f_y: Optional[Union[pd.Series, np.ndarray, list, Callable,]] = None,
        i_max: Optional[int] = None,
        stack_H: Optional[bool] = None,
        ):

    stack_H = defaults.stack_H(stack_H)

    if f_y is None:
        def f_y(X): return np.sum(X, axis=1)

    if np.isscalar(f_y):
        f_y = [f_y]

    if not isinstance(f_y, pd.Series):
        f_y = pd.Series(f_y)

    y_samp = pd.concat(
                [f_X_i_max(X=X_samp, func=v, i_max=i_max) for v in f_y.values],
                axis=1, keys=f_y.index,
                )

    index = None
    if hasattr(X_samp, "values"):
        index = X_samp.index
        if len(index) == len(y_samp):
            y_samp.index = index

    N_stack = y_samp.columns.nlevels - f_y.index.nlevels

    if N_stack > 0 and stack_H:
        H_i = np.arange(N_stack)
        stack_i = [int(i) for i in (f_y.index.nlevels + H_i)]
        index_names = y_samp.index.names
        stack_names = "H_i_" + H_i.astype(str)

        y_samp = y_samp.stack(stack_i)
        y_samp.index = y_samp.index.set_names([*index_names, *stack_names])

    return y_samp


def merge_X_y(
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        app_merge: Optional[bool] = None,
        expand_X: Optional[bool] = None,
        match_head: Optional[bool] = None,
        ) -> Union[pd.DataFrame, tuple]:

    app_merge = defaults.app_merge(app_merge)
    expand_X = defaults.expand_X(expand_X)
    match_head = defaults.match_head(match_head)

    if app_merge:
        expand_X = True
        match_head = True

    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()

    if expand_X:
        if y.index.nlevels > X.index.nlevels:
            lvl = list(X.index.names)
            if len(lvl) == 1:
                lvl = lvl[0]

            X = X.reindex(y.index, level=lvl)

    if match_head:
        head_diff = y.columns.nlevels - X.columns.nlevels
        if head_diff < 0:
            y = pd.concat([y], axis=1, keys=[tuple(np.repeat("y", -head_diff))])
        elif head_diff > 0:
            X = pd.concat([X], axis=1, keys=[tuple(np.repeat("X", head_diff))])

    if app_merge:
        X_y = pd.concat(
                    [X, y],
                    axis=1, keys=["X", "y"],
                    )

        return X_y
    else:
        return X, y


def calc_X_y_samp(
        D_dist: Optional[Union[pd.Series, np.ndarray, list,]] = None,
        D_bounds: Optional[Union[pd.DataFrame, np.ndarray,]] = None,
        input_features: Optional[Union[pd.Series, np.ndarray, list,]] = None,
        safe_alpha: Optional[float] = None,
        N_samp: Optional[int] = None,
        N_cline: Optional[int] = None,
        f_y: Optional[Union[pd.Series, np.ndarray, list, Callable,]] = None,
        i_max: Optional[int] = None,
        stack_H: Optional[bool] = None,
        app_merge: Optional[bool] = None,
        expand_X: Optional[bool] = None,
        match_head: Optional[bool] = None,
        ) -> Union[pd.DataFrame, tuple]:

    X_samp = calc_X_samp(
                D_dist=D_dist,
                D_bounds=D_bounds,
                input_features=input_features,
                safe_alpha=safe_alpha,
                N_samp=N_samp,
                N_cline=N_cline,
                )

    y_samp = calc_y_samp(
                X_samp=X_samp,
                f_y=f_y,
                i_max=i_max,
                stack_H=stack_H,
                )

    return merge_X_y(
            X=X_samp, y=y_samp,
            app_merge=app_merge,
            expand_X=expand_X,
            match_head=match_head,
            )


def calc_X_y_minmax(
        X,
        y,
        kind=None,
        **kwargs,
        ):

    kind = str(defaults.set_default(kind, "both"))[:3].lower()
    if kind[:1] == "b":
        mm = ["min", "max"]
        return pd.concat(
                    [calc_X_y_minmax(X=X, y=y, kind=m, **kwargs) for m in mm],
                    axis=0, keys=mm,
                    )
    else:
        idxm = getattr(y, f"idx{kind}")(axis=0)
        X_y_m = merge_X_y(X=X, y=y).loc[idxm].reset_index().set_axis(idxm.index, axis=0)

        return X_y_m


def calc_X_y_samp_alt(
        D_dist: Optional[Union[pd.Series, np.ndarray, list,]] = None,
        D_bounds: Optional[Union[pd.DataFrame, np.ndarray,]] = None,
        input_features: Optional[Union[pd.Series, np.ndarray, list,]] = None,
        safe_alpha: Optional[float] = None,
        N_samp: Optional[int] = None,
        N_cline: Optional[int] = None,
        f_y: Optional[Union[pd.Series, np.ndarray, list, Callable,]] = None,
        i_max: Optional[int] = None,
        app_merge: Optional[bool] = None,
        ) -> Union[pd.DataFrame, tuple]:

    if f_y is None:
        def f_y(X): return np.sum(X, axis=1)

    if np.isscalar(f_y):
        f_y = [f_y]

    if not isinstance(f_y, pd.Series):
        f_y = pd.Series(f_y)

    N_samp = defaults.N_samp(N_samp)
    N_cline = defaults.N_cline(N_cline)
    if N_cline < 0:
        N_cline = 21

    D_dist, D_bounds = calc_D_dist_bounds(
                            D_dist=D_dist,
                            D_bounds=D_bounds,
                            input_features=input_features,
                            safe_alpha=safe_alpha,
                            )

    if N_cline == 0:
        X_samp = D_dist.apply(lambda v: pd.Series(v.rvs(N_samp))).T
        y_samp = calc_y_samp(X_samp=X_samp, f_y=f_y, i_max=i_max)
    else:
        X_samp = pd.DataFrame()
        y_samp = pd.DataFrame()
        X_j_index = pd.MultiIndex.from_product([
                            [""], np.arange(N_samp), np.arange(N_cline)
                            ], names=["c", "rc_i", "c_i"]
                            )

        for j, c in enumerate(D_dist.index):
            rc = D_bounds.columns.drop(c)
            c_coords = pd.Series(
                            np.linspace(*D_bounds[c].values, N_cline),
                            name=c,
                            ).to_frame()
            rc_coords = D_dist.loc[rc].apply(lambda v: pd.Series(v.rvs(N_samp))).T
            _X_j = rc_coords.merge(c_coords, how="cross")[D_bounds.columns]
            _X_j.index = X_j_index
            _X_j = _X_j.rename(index={"": c}, level=0)
            _y_j = calc_y_samp(X_samp=_X_j, f_y=f_y, i_max=i_max)

            X_samp = pd.concat([X_samp, _X_j], axis=0)
            y_samp = pd.concat([y_samp, _y_j], axis=0)

    return merge_X_y(X=X_samp, y=y_samp, app_merge=app_merge)
