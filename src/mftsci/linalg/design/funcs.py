"""
Functions module for linalg.design.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import numpy as np
import pandas as pd
from mftsci.utils.meta import require_optional
from mftsci.linalg.regr import calc_hlog_D_opt, calc_D_opt, \
                                calc_D_opt_slow  # noqa: F401

# Local
from mftsci.linalg.design import defaults

# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def calc_mirror(
        mirror: Optional[Union[np.ndarray, list, int,]] = None,
        N_mirror: Optional[int] = None,
        ):

    if mirror is None:
        if N_mirror is None:
            N_mirror = 1
        mirror = (-1)**np.arange(N_mirror)
    else:
        mirror = np.atleast_1d(mirror).reshape(-1)

    return mirror


def mirror_X(
        X: Union[np.ndarray, pd.DataFrame, pd.Series,],
        mirror: Optional[Union[np.ndarray, list, int,]] = None,
        N_mirror: Optional[int] = None,
        as_pd: Optional[bool] = None,
        reindex: Optional[bool] = None,
        ):

    mirror = calc_mirror(mirror=mirror, N_mirror=N_mirror)
    N_mirror = mirror.size
    X_shape = X.shape
    X_ndim = X.ndim

    was_pd = hasattr(X, "values")
    if as_pd is None:
        as_pd = was_pd

    index = None
    columns = getattr(X, "columns", None)
    if was_pd:
        if reindex is None:
            reindex = True
        if not reindex:
            index = getattr(X, "index", None)
        X = X.values

    # X_m = np.outer(X.T, mirror).reshape((X.shape[1], X.shape[0]*mirror.size)).T
    X_m = np.concat(
            np.expand_dims(X, 0) * mirror.reshape(-1, *np.ones(X_ndim, dtype=int)),
            axis=0,
            )[
                np.concat(np.add.outer(
                            np.arange(X_shape[0]),
                            X_shape[0]*np.arange(N_mirror)
                            ))]

    if as_pd:
        if X_ndim == 1:
            X_m = pd.Series(X_m, index=index)
        elif X_ndim == 2:
            X_m = pd.DataFrame(X_m, index=index, columns=columns)

    return X_m


def calc_conference_matrix(
                    n: int,
                    as_array: Optional[bool] = None,
                    ):

    oapackage = require_optional(module="oapackage", install="mftsci[linalg_design]")
    conference_class = oapackage.conference_t(n, n, 0)
    conference_designs = [[conference_class.create_root()]]

    if n > 2:
        for ii, ncols in enumerate(range(3, n+1)):
            arrays = oapackage.extend_conference(
                conference_designs[ii],
                conference_class,
                verbose=0)
            conference_designs.append(arrays)

    design = conference_designs[-1][0]

    if as_array is None:
        as_array = False

    if as_array:
        return np.array(design)
    else:
        return design


def calc_dsd(
            n: int,
            n_fake: Optional[int] = None,
            n_centre: Optional[int] = None,
            ):

    n_fake = defaults.set_default(n_fake, 0)
    n_centre = defaults.set_default(n_centre, 1)

    n_eff = n + n_fake
    if (n_eff % 2) == 1:
        n_eff += 1

    C = calc_conference_matrix(n=n_eff, as_array=True)[:, :n]

    D = np.concat([
                np.zeros((n_centre, n), dtype=int),
                mirror_X(C, N_mirror=2)
                ], axis=0)

    return D


def D_cont_coord_exchange(
                D: Optional[np.ndarray] = None,
                N_rows: Optional[int] = None,
                N_cols: Optional[int] = None,
                f_model: Optional[Callable] = None,
                mirror: Optional[Union[np.ndarray, list, int,]] = None,
                N_mirror: Optional[int] = None,
                D_prior: Optional[np.ndarray] = None,
                N_init: Optional[int] = None,
                N_iter: Optional[int] = None,
                ):

    if f_model is None:
        def f_model(x): return x

    if D is not None:
        N_rows, N_cols = np.shape(D)
    else:
        if D_prior is not None:
            N_cols = np.shape(D_prior)[1]
        D = np.full((N_rows, N_cols), np.nan)

    if D_prior is None:
        D_prior = np.zeros((0, N_cols), dtype=float)
    N_prior = np.shape(D_prior)[0]

    mirror = calc_mirror(mirror=mirror, N_mirror=N_mirror)
    N_mirror = mirror.size

    X = f_model(
            np.concat([
                D_prior[:, :N_cols],
                mirror_X(np.zeros((N_rows, N_cols), dtype=float), mirror=mirror),
                ], axis=0)
            )

    D_free = np.isnan(D)
    D_coords = [*zip(*np.where(D_free))]

    if N_init is None:
        N_init = 100

    if N_iter is None:
        N_iter = 100

    c_vals = np.array([-1, 1], dtype=float)
    c_hlog_D_opt = np.repeat(np.nan, c_vals.size)

    best_hlog_D_opt = -np.inf
    record_iter = [(-1,  best_hlog_D_opt)]
    D_best = None

    record = []
    for i in range(N_init):

        D[D_free] = 2*(np.random.rand(len(D_coords)) - 0.5)

        X[N_prior:] = f_model(mirror_X(D, mirror=mirror))
        hlog_D_opt = np.sum(np.log(np.linalg.svd(X, compute_uv=False)))
        record.append([(-1, hlog_D_opt)])

        for j in range(N_iter):
            np.random.shuffle(D_coords)
            for c in D_coords:
                for k, v in enumerate(c_vals):
                    # D[*c] = v   # Invalid syntax in python/3.10: array[*tuple/list]
                    D[c[0], c[1]] = v
                    if False:
                        X[
                            N_prior+N_mirror*c[0]:N_prior+N_mirror*(1+c[0])
                            ] = f_model(mirror_X(D[[c[0]]], mirror=mirror))
                    else:
                        X[N_prior:] = f_model(mirror_X(D, mirror=mirror))
                    c_hlog_D_opt[k] = calc_hlog_D_opt(X=X)

                amax = np.argmax(c_hlog_D_opt)
                # D[*c] = c_vals[amax]    # Invalid syntax in python/3.10
                D[c[0], c[1]] = c_vals[amax]
            if c_hlog_D_opt[amax] > hlog_D_opt:
                hlog_D_opt = c_hlog_D_opt[amax]
                record[-1].append((j, hlog_D_opt))

        if hlog_D_opt > best_hlog_D_opt:
            D_best = D.copy()
            best_hlog_D_opt = hlog_D_opt
            record_iter.append((i,  best_hlog_D_opt))

    D_out = np.concat([
                D_prior[:, :N_cols],
                mirror_X(D_best, mirror=mirror),
                ], axis=0)

    return D_out, record_iter, record
