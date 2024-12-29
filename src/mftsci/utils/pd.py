"""
Module for pandas utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union
import datetime

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils import arr
from mftsci.utils.dict import keys, vals

# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def index_df_bool(
                df: pd.core.frame.DataFrame,
                cv: Optional[dict] = None,
                ) -> pd.core.frame.DataFrame:

    if cv is None:
        cv = {}

    return pd.Series(
                    np.all(
                        np.array([
                                    np.isin(df[col].values, arr.iterarr(cv[col]))
                                    for col in cv
                                    ]),
                        axis=0), index=df.index)


def index_df(
                df: pd.core.frame.DataFrame,
                cv: Optional[dict] = None,
                ) -> pd.core.frame.DataFrame:

    return df[index_df_bool(df=df, cv=cv)]


def calc_level_names(
                        N_levels: Optional[int] = None,
                        level_names: Optional[Union[np.ndarray, list, str,]] = None,
                        ) -> np.ndarray:

    level_names = arr.iterarr(level_names)
    len_level_names = level_names.size
    if N_levels is None:
        N_levels = len_level_names

    default_level_names = "level_" + np.array(np.arange(N_levels), dtype=str)
    null_level_names = level_names is None
    if np.all(null_level_names):
        level_names = default_level_names
    else:
        level_names[null_level_names] = default_level_names[:level_names.size
                                                            ][null_level_names]
        eff_size = np.min([level_names.size, N_levels])
        level_names = np.concat([level_names[:eff_size],
                                 default_level_names[eff_size:N_levels]])

    return level_names


def calc_index_vals(
                    N_vals: Optional[int] = None,
                    index_vals: Optional[Union[np.ndarray, list, int,]] = None,
                    ):

    index_vals = arr.iterarr(index_vals)
    if N_vals is None:
        N_vals = index_vals.size

    default_index_vals = np.arange(N_vals)
    null_index_vals = index_vals == None    # noqa: E711
    if np.all(null_index_vals):
        index_vals = default_index_vals
    else:
        index_vals[null_index_vals] = default_index_vals[:index_vals.size
                                                         ][null_index_vals]
        eff_size = np.min([index_vals.size, N_vals])
        index_vals = np.concat([index_vals[:eff_size],
                                default_index_vals[eff_size:N_vals]])

    return index_vals


def calc_arr_index_vals(
                        a: Optional[Union[
                                            np.ndarray,
                                            list,
                                            float,
                                            int,
                                            bool,
                                            str,
                                            ]] = None,
                        a_shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                        arr_index_vals: Optional[Union[
                                                        dict,
                                                        np.ndarray,
                                                        list,
                                                        object,
                                                        ]] = None,
                        level_names: Optional[Union[np.ndarray, list, str,]] = None,
                        overwrite_level_names: Optional[bool] = None,
                        ) -> dict:

    if overwrite_level_names is None:
        if level_names is None:
            overwrite_level_names = False
        else:
            overwrite_level_names = False

    if arr_index_vals is None:
        arr_index_vals = {level: None for level in arr.iterarr(level_names)}
    elif np.isscalar(arr_index_vals):
        arr_index_vals = [[arr_index_vals]]
    elif isinstance(arr_index_vals, np.ndarray):
        if len(arr_index_vals.shape) == 1:
            arr_index_vals = [arr_index_vals]

    if a_shape is None:
        if a is not None:
            a_shape = arr.calc_shape(a=a)
        else:
            a_shape = [None for k in len(arr_index_vals)]
    N_levels = len(a_shape)
    if N_levels == 0:
        a_shape = [1]
        N_levels = 1

    if isinstance(arr_index_vals, dict):
        if level_names is None or not overwrite_level_names:
            level_names = keys(arr_index_vals)
        arr_index_vals = vals(arr_index_vals, array=False)

    level_names = calc_level_names(N_levels=N_levels, level_names=level_names)
    len_arr_index_vals = len(arr_index_vals)
    if len_arr_index_vals < N_levels:
        arr_index_vals = [*arr_index_vals,
                          *np.repeat(None, N_levels-len_arr_index_vals)]

    return {level_names[l]: calc_index_vals(
                                            N_vals=a_shape[l],
                                            index_vals=arr_index_vals[l],
                                            )
            for l in range(N_levels)}   # noqa: E741


def calc_arr_index(
                    a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                    a_shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                    arr_index_vals: Optional[Union[
                                                    dict,
                                                    np.ndarray,
                                                    list,
                                                    object,
                                                    ]] = None,
                    level_names: Optional[Union[np.ndarray, list, str,]] = None,
                    overwrite_level_names: Optional[bool] = None,
                    ) -> Union[
                                pd.core.indexes.base.Index,
                                pd.core.indexes.multi.MultiIndex,
                                ]:

    arr_index_vals = calc_arr_index_vals(a=a, a_shape=a_shape,
                                         arr_index_vals=arr_index_vals,
                                         level_names=level_names,
                                         overwrite_level_names=overwrite_level_names)

    level_names = keys(arr_index_vals)

    return pd.MultiIndex.from_product([arr_index_vals[n] for n in level_names],
                                      names=level_names)


def arr_to_srs(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                a_shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                arr_index_vals: Optional[Union[dict, np.ndarray, list, object,]] = None,
                level_names: Optional[Union[np.ndarray, list, str,]] = None,
                overwrite_level_names: Optional[bool] = None,
                name: Optional[str] = None,
                ) -> pd.core.series.Series:

    index = calc_arr_index(
                            a=a, a_shape=a_shape,
                            arr_index_vals=arr_index_vals, level_names=level_names,
                            overwrite_level_names=overwrite_level_names
                            )

    return pd.Series(arr.iterarr(a), index=index, name=name)


def df_to_srs(
                df: Optional[pd.core.frame.DataFrame] = None,
                name: Optional[str] = None,
                ) -> pd.core.series.Series:

    if df is None:
        df = pd.DataFrame()

    names = arr.iterarr(df.columns.names)
    for l in range(names.size):     # noqa: E741
        df = df.stack(future_stack=True)

    df.name = name

    return df


def index_unique(
                    index: Optional[Union[
                                            pd.core.indexes.base.Index,
                                            pd.core.indexes.multi.MultiIndex,
                                            ]] = None,
                    appear_order: Optional[bool] = None,
                    ) -> dict:

    if appear_order is None:
        appear_order = True

    if index is None:
        index = pd.Series().index

    level_names = arr.iterarr(index.names)
    N_levels = level_names.size

    return {level_names[i]: arr.unique(index.get_level_values(i),
                                       appear_order=appear_order)
            for i in range(N_levels)}


def srs_to_arr(
                srs: Optional[pd.core.series.Series] = None,
                ) -> np.ndarray:

    if srs is None:
        srs = pd.Series()

    arr_index_vals = index_unique(index=srs.index, appear_order=False)
    a_shape = np.array([arr_index_vals[l].size for l in arr_index_vals])    # noqa: E741

    return np.reshape(srs.values, a_shape)


def df_to_arr(
                df: Optional[pd.core.frame.DataFrame] = None,
                ) -> np.ndarray:

    return srs_to_arr(srs=df_to_srs(df=df))


def srs_to_df_1(
                srs: Optional[pd.core.series.Series] = None,
                col_levels: Optional[Union[np.ndarray, list, int, ]] = None,
                ) -> pd.core.frame.DataFrame:

    if srs is None:
        srs = pd.Series()

    if col_levels is None:
        col_levels = []
    col_levels = arr.iterarr(col_levels)

    name = srs.name
    if name is None:
        name = 0

    level_names = arr.iterarr(srs.index.names)
    if col_levels.size > 0:
        col_names = level_names[col_levels]
    else:
        col_names = arr.iterarr([])
    index_names = arr.filter(level_names, a_rm=col_names)

    if col_levels.size == 0:
        df = srs.to_frame()
    else:
        df = srs.reset_index()
        if index_names.size > 0:
            df = df.set_index(keys=[*index_names])
        df = df.pivot(columns=[*col_names])
        if col_names.size > 0:
            df.columns = df.columns.droplevel(0)

    return df


def arr_to_df(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                a_shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                arr_index_vals: Optional[Union[dict, np.ndarray, list, object,]] = None,
                level_names: Optional[Union[np.ndarray, list, str,]] = None,
                overwrite_level_names: Optional[bool] = None,
                name: Optional[str] = None,
                level: Optional[Union[np.ndarray, list, int,]] = None,
                ) -> pd.core.frame.DataFrame:

    srs = arr_to_srs(
                        a=a, a_shape=a_shape,
                        arr_index_vals=arr_index_vals, level_names=level_names,
                        overwrite_level_names=overwrite_level_names,
                        name=name,
                        )

    return srs.unstack(level=level)


def dt_str(
            dt: datetime.datetime,
            ) -> str:

    return str(dt.strftime("%Y-%m-%d_%H%M%S"))
