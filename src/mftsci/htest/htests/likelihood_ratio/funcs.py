"""
Functions module for Likelihood--Ratio test.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from __future__ import annotations
from typing import Optional, Union, Callable, Tuple

# External packages
import numpy as np  # temp
import scipy  # temp
import pandas as pd     # temp

# Internal packages
from mftsci.htest.htests.core import *   # noqa: F401, F403
from mftsci.htest.htests.core import TestMeta, TestWrapper, TestRes
from mftsci import utils    # temp
from mftsci.htest.htests.likelihood_ratio import defaults


# =====================================
# CLASSES
# =====================================


class LikelihoodRatioMeta(TestMeta):

    def __init__(
                    self,
                    ):

        TestMeta.__init__(self,
                          test_long_name="Likelihood--Ratiov",
                          test_short_name="LR")


class LikelihoodRatio(TestWrapper, LikelihoodRatioMeta):

    def __init__(
                    self,
                    log_L: Union[np.ndarray, list, float,],
                    N_params: Union[np.ndarray, list, int,] = None,
                    names: Union[np.ndarray, list, str, int,] = None,
                    ascending: Optional[bool] = None,
                ):

        self.set_res(log_L=log_L, N_params=N_params,
                     names=names, ascending=ascending)

    def set_res(
                    self,
                    log_L: Union[np.ndarray, list, float,],
                    N_params: Union[np.ndarray, list, int,] = None,
                    names: Union[np.ndarray, list, str, int,] = None,
                    ascending: Optional[bool] = None,
                    ):

        self.res = LikelihoodRatioResult(log_L=log_L, N_params=N_params,
                                         names=names, ascending=ascending)


class LikelihoodRatioResult(TestRes, LikelihoodRatioMeta):

    def __init__(
                    self,
                    log_L: Union[np.ndarray, list, float,],
                    N_params: Union[np.ndarray, list, int,] = None,
                    names: Union[np.ndarray, list, str, int,] = None,
                    ascending: Optional[bool] = None,
                    ):

        LikelihoodRatioMeta.__init__(self)
        self.run(log_L=log_L, N_params=N_params, names=names, ascending=ascending)

    def __defaultkeys__(
                        self,
                        ) -> list:

        return ["stat", "df", "p"]

    def calc_res_index(
                        self,
                        ) -> pd.core.indexes.multi.MultiIndex:

        return calc_index(names=self.names, N_mods=self.N_mods)

    def calc_iloc_idx(
                        self,
                        ) -> np.ndarray:

        return calc_iloc_idx(N_mods=self.N_mods, ascending=self.ascending)

    def calc_metafunc_index(
                                self,
                                *args, **kwargs,
                                ) -> Callable:

        return lambda x: x[self.calc_iloc_idx()]

    def calc_metafunc_value(
                                self,
                                *args, **kwargs,
                                ) -> Callable:

        return lambda x: np.hstack(x)[self.calc_iloc_idx()]

    def run(
            self,
            log_L: Union[np.ndarray, list, float,],
            N_params: Union[np.ndarray, list, int,],
            names: Union[np.ndarray, list, str, int,] = None,
            ascending: Optional[bool] = None,
            ):

        self.log_L, self.N_params, self.names, self.ascending = sort_LR_args(
                                                                    log_L,
                                                                    N_params,
                                                                    names=names,
                                                                    ascending=ascending,
                                                                    )
        self.N_mods = np.size(self.log_L)

        self.stat = calc_stat(log_L=self.log_L)
        self.df = calc_df(N_params=self.N_params)
        self.p = calc_p(stat=self.stat, df=self.df)


# =====================================
# FUNCTIONS
# =====================================


# -------------------------------------
# DECORATORS
# -------------------------------------


def sort_LR_args(
                    log_L,
                    N_params,
                    names=None,
                    ascending=None,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    ascending = defaults.ascending(ascending=ascending)

    log_L = utils.arr.iterarr(log_L)
    N_params = utils.arr.iterarr(N_params)

    if names is None:
        names = np.arange(log_L.size)

    sort_args = np.argsort(N_params)
    if not ascending:
        sort_args = np.flip(sort_args)

    log_L = log_L[sort_args]
    N_params = N_params[sort_args]
    names = names[sort_args]

    return log_L, N_params, names, ascending


def calc_stat(
                log_L,
                ) -> np.ndarray:

    return 2*np.subtract.outer(log_L, log_L)


def calc_df(
            N_params,
            ) -> np.ndarray:

    return np.subtract.outer(N_params, N_params)


def calc_p(
            stat,
            df,
            ) -> np.ndarray:

    return scipy.stats.chi2(df=df).sf(stat)


def calc_level_names() -> np.ndarray:

    return "Theta_" + np.array(np.arange(2), dtype=str)


def calc_arr_index_vals(
                            names=None,
                            N_mods=None,
                            ) -> dict:

    level_names = calc_level_names()

    if names is None:
        N_mods = defaults.N_mods(N_mods)
        names = np.arange(N_mods)
    else:
        names = utils.arr.iterarr(names)
        N_mods = np.size(names)

    return {ln: names for ln in level_names}


def calc_index(
                names=None,
                N_mods=None,
                ) -> pd.core.indexes.multi.MultiIndex:

    arr_index_vals = calc_arr_index_vals(names=names, N_mods=N_mods)
    N_mods = len(list(arr_index_vals.values())[0])

    return utils.pd.calc_arr_index(a_shape=np.repeat(N_mods, 2),
                                   arr_index_vals=arr_index_vals)


def calc_iloc_idx(
                    N_mods: int,
                    ascending: Optional[bool] = None,
                    ) -> np.ndarray:

    ascending = defaults.ascending(ascending=ascending)

    if ascending:
        f_indices = np.tril_indices
        k = -1
    else:
        f_indices = np.triu_indices
        k = 1

    return np.array([N_mods, 1]) @ np.array(f_indices(n=N_mods, k=k))
