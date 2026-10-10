"""
Functions module for pandas utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal, Callable
import datetime
from itertools import product, cycle
from operator import mul
from functools import reduce

# External packages
import numpy as np
import pandas as pd
import pint

# Internal packages
from .. import arr
from ..meta import (
                    set_default_kwargs,
                    wraps_ovar, merge_args_kwargs,
                    return_x,
                    )
from ..dict import keys, vals
from ..pd import defaults
from ..str.core import num_convert as _num_convert


# =====================================
# FUNCTIONS
# =====================================


def via_numpy(
                ret_transposed=None,
                *vargs, **vkwargs,
                ) -> Callable:

    if ret_transposed is None:
        ret_transposed = True

    def decorator(func):

        @wraps_ovar(func)
        def wrapper(
                    *args, **kwargs,
                    ) -> object:

            kwargs = merge_args_kwargs(_func=func, args=args, kwargs=kwargs)
            kw_0 = next(iter(kwargs.keys()), None)
            v_0 = kwargs.get(kw_0, None)

            index = None
            columns = None
            is_srs = isinstance(v_0, pd.Series)
            is_df = isinstance(v_0, pd.DataFrame)

            if is_srs or is_df:
                index = v_0.index
                if is_df:
                    columns = v_0.columns
                kwargs[kw_0] = v_0.values

            ret = func(**kwargs)
            if is_srs:
                ret = pd.Series(ret, index=index)
            elif is_df:
                if ret_transposed:
                    ret = pd.DataFrame(ret, index=columns, columns=index)
                else:
                    ret = pd.DataFrame(ret, index=index, columns=columns)

            return ret

        return wrapper

    return decorator


inv = via_numpy(ret_transposed=True)(np.linalg.inv)


def get(
        df=None,
        idx=None,
        null_kwargs=None,
        ):

    if df is None:
        df = pd.Series()

    if idx is None:
        return df
    else:
        if np.isscalar(idx):
            idx = [idx]
        if isinstance(idx, list):
            idx = np.array(idx)
        isin = np.isin(idx, df.index)

        if not isin.all():
            null_kwargs = set_default_kwargs(null_kwargs)

            if isinstance(df, pd.Series):
                df_null = pd.Series(
                            index=idx[~isin],
                            name=df.name,
                            **null_kwargs
                            )
            else:
                df_null = pd.DataFrame(
                            index=idx[~isin],
                            columns=df.columns,
                            **null_kwargs
                            )

            df = pd.concat([
                    df.loc[idx[isin]],
                    df_null.rename_axis(df.index.names, axis=0),
                    ],
                    axis=0)

        return df.loc[idx]


def cross_vc(
        df,
        df_target,
        normalize=None,
        ):

    if normalize is None:
        normalize = False

    df_vc = df.apply(lambda v: (v.value_counts(),), axis=0).T.iloc[:, 0]
    df_target_unique = df_target.apply(lambda v: v.unique(), axis=0)

    vc = df_vc.apply(
        lambda v:
        df_target_unique.apply(
            lambda w:
            (v.index.isin(w).astype(int) * v.values).sum()
            )
        )

    if normalize:
        vc /= df.shape[0]

    return vc


def reset_index(
                df: Optional[Union[
                            pd.core.frame.DataFrame,
                            pd.core.series.Series,
                            ]] = None,
                reindex: Optional[bool] = None,
                ) -> pd.core.frame.DataFrame:

    df = defaults.df(df)
    index = df.index
    df = df.reset_index()
    if defaults.reindex(reindex):
        df.index = index

    return df


def index_to_df(
                df: Optional[Union[
                                    pd.core.frame.DataFrame,
                                    pd.core.series.Series,
                                    ]] = None,
                index: Optional[Union[
                                            pd.core.indexes.base.Index,
                                            pd.core.indexes.multi.MultiIndex,
                                            ]] = None,
                reindex: Optional[bool] = None,
                ) -> pd.core.frame.DataFrame:

    if df is None:
        df = pd.Series(index=defaults.index(index))

    index_nlevels = df.index.nlevels
    df = reset_index(df=df, reindex=reindex)
    level_drop = [*(1 + np.arange(df.columns.nlevels - 1))]

    df = df.iloc[:, :index_nlevels].droplevel(level=level_drop, axis=1)

    return df


def index_df_is(
                df: pd.core.frame.DataFrame,
                cv: Optional[dict] = None,
                incl_index: Optional[bool] = None,
                ) -> pd.core.frame.DataFrame:

    cv = defaults.cv(cv)
    if defaults.incl_index(incl_index):
        df = reset_index(df, reindex=True)

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
                incl_index: Optional[bool] = None,
                incl_index_ret: Optional[bool] = None,
                ) -> pd.core.frame.DataFrame:

    df = df[index_df_is(df=df, cv=cv, incl_index=incl_index)]
    if defaults.incl_index_ret(incl_index_ret):
        df = reset_index(df, reindex=True)
    return df


def where_is(
                is_Series: pd.core.series.Series,
                ) -> Union[
                            pd.core.indexes.base.Index,
                            np.ndarray,
                            ]:

    return is_Series.index[is_Series]


def index_df_where(
                    df: pd.core.frame.DataFrame,
                    cv: Optional[dict] = None,
                    incl_index: Optional[bool] = None,
                    ) -> Union[
                                pd.core.indexes.base.Index,
                                np.ndarray,
                                ]:

    return where_is(is_Series=index_df_is(df=df, cv=cv, incl_index=incl_index))


def move_row_in_group(
                        df: Union[
                                    pd.core.frame.DataFrame,
                                    pd.core.series.Series,
                                    ],
                        group_index: Optional[Union[
                                                    pd.core.indexes.base.Index,
                                                    np.ndarray,
                                                    ]] = None,
                        source: Optional[Union[
                                                pd.core.indexes.base.Index,
                                                np.ndarray,
                                                object,
                                                ]] = None,
                        group_source: Optional[Union[
                                                    pd.core.indexes.base.Index,
                                                    np.ndarray,
                                                    int,
                                                    ]] = None,
                        destination: Optional[Union[
                                                    pd.core.indexes.base.Index,
                                                    np.ndarray,
                                                    object,
                                                    ]] = None,
                        group_destination: Optional[Union[
                                                    pd.core.indexes.base.Index,
                                                    np.ndarray,
                                                    int,
                                                    ]] = None,
                        conserve_index: Optional[bool] = None,
                        ) -> Union[
                                    pd.core.frame.DataFrame,
                                    pd.core.series.Series,
                                    ]:

    conserve_index = defaults.conserve_index(conserve_index)

    index_Series = pd.Series(df.index, index=df.index)

    if group_index is None:
        group_index = index_Series.values
    group_index_Series = pd.Series(group_index, index=group_index)

    if source is None:
        group_source = defaults.group_source(group_source)
        source = group_index[group_source]

    if destination is None:
        group_destination = defaults.group_destination(group_destination)
        destination = group_index[group_destination]

    not_source = arr.filter(group_index, a_rm=source)
    not_destination = arr.filter(group_index, a_rm=destination)

    new_group_index_Series = group_index_Series.copy()
    new_group_index_Series[destination] = group_index_Series[source]
    new_group_index_Series[not_destination] = group_index_Series[not_source]

    index_Series[new_group_index_Series.index] = new_group_index_Series.values

    df = df.loc[index_Series.values]
    if not conserve_index:
        df.index = index_Series.index

    return df


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

    overwrite_level_names = defaults.overwrite_level_names(overwrite_level_names)

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
            a_shape = [None for k in range(len(arr_index_vals))]
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

    df = defaults.df(df)

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

    appear_order = defaults.appear_order(appear_order)
    index = defaults.index(index)
    level_names = arr.iterarr(index.names)
    N_levels = level_names.size

    return {level_names[i]: arr.unique(index.get_level_values(i),
                                       appear_order=appear_order)
            for i in range(N_levels)}


def merge(
        dfs: list,
        merge_index: Optional[bool] = None
        ) -> pd.DataFrame:

    if merge_index is None:
        merge_index = True

    df = reduce(lambda left, right: left.merge(right, how="cross"), dfs)

    if merge_index:
        df = df.set_axis(pd.MultiIndex.from_product(map(lambda v: v.index, dfs)))

    return df


def srs_to_arr(
                srs: Optional[pd.core.series.Series] = None,
                ) -> np.ndarray:

    srs = defaults.srs(srs)

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

    srs = defaults.srs(srs)
    col_levels = defaults.col_levels(col_levels)
    # name = defaults.name(srs.name)

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


def _clean_index_func(s):

    null_prefix = "Unnamed: "
    fill_null = ""

    return fill_null if s.startswith(null_prefix) else s


def isscalar(a):

    if a is None:
        return True
    elif isinstance(a, pint.Quantity):
        return True
    else:
        return np.isscalar(a)


def safelen(x):
    if isscalar(x):
        return 1
    else:
        return len(x)


def df_from_kwargs(
        **kwargs,
        ):

    values = {
                k: v if not isscalar(v) else [v]
                for k, v in kwargs.items()
                }

    n = max(map(safelen, values.values()), default=0)

    cycle_values = {k: cycle(v) for k, v in values.items()}

    df = pd.DataFrame({
                    k: [next(v) for _ in range(n)]
                    for k, v in cycle_values.items()
                    })
    return df


def df_from_kwargs_keys(
        kwargs=None,
        dkwargs=None,
        keys=None,
        app_fill=None,
        fill=None,
        skip_dkwargs_if_empty=None,
        ):

    if skip_dkwargs_if_empty is None:
        skip_dkwargs_if_empty = True

    if kwargs is None:
        kwargs = {}

    if len(kwargs) > 0 or not skip_dkwargs_if_empty:
        kwargs = set_default_kwargs(
                        kwargs,
                        dkwargs=dkwargs,
                        )

    if app_fill is None:
        app_fill = fill is not None

    _keys = kwargs.keys()

    if keys is None:
        keys = _keys
    else:
        if np.isscalar(keys):
            keys = [keys]
        if not app_fill:
            keys = [k for k in keys if k in _keys]

    values = {
                k: kwargs.get(k, fill)
                for k in keys
                }

    df = df_from_kwargs(**values)

    if len(kwargs) == 0:
        df = df.iloc[:0]

    return df


def kwargs_list_from_srs(
        srs,
        collapse=None,
        ):

    if collapse is None:
        collapse = True

    if isscalar(srs):
        return srs
    else:
        if not collapse:
            N_unique = 2
        else:
            try:
                N_unique = np.unique(srs).size
            except Exception:
                N_unique = 2
        if N_unique == 1:
            return srs[0]
        else:
            return [*srs]


def kwargs_from_df(
        df,
        collapse=None,
        ):

    if collapse is None:
        collapse = True

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = {
            k: kwargs_list_from_srs(v, collapse=collapse)
            for k, v in df.items()
            }

    return df


def is_dt_dtype(
        dtypes
        ):

    return dtypes.astype(str).str.contains("datetime64|timedelta64")


def to_subdtype(
        df,
        dtypes=None,
        app=None,
        incl_index=None,
        ):

    if app is None:
        app = True

    if incl_index is None:
        incl_index = False

    if app:
        if isinstance(df, pd.Index):
            is_multi = isinstance(df, pd.MultiIndex)

            names = df.names

            df = pd.MultiIndex.from_frame(
                    to_subdtype(
                        df=df.to_frame(index=False),
                        dtypes=dtypes,
                        app=app,
                        incl_index=False
                        )
                    )
            df = df.set_names(names)

            if not is_multi:
                df = df.get_level_values(0)

            return df

        elif isinstance(df, pd.Series):
            return to_subdtype(
                    df.to_frame(),
                    dtypes=dtypes,
                    app=app,
                    ).iloc[:, 0]

        if dtypes is None:
            dtypes = df.dtypes.rename("dtype")
        dt_cols = is_dt_dtype(dtypes)
        subdtype = dtypes.apply(lambda v: getattr(v, "subdtype", None))
        subdtype.loc[dt_cols] = "int64"
        if True:
            subdtype[subdtype.apply(pd.core.arrays.string_.is_string_dtype)] = str
            subdtype[subdtype.apply(pd.core.arrays.integer.is_integer_dtype)] = int
            subdtype[subdtype.apply(pd.core.arrays.floating.is_float_dtype)] = float
        df = df.astype(subdtype)

        if incl_index:
            df.index = to_subdtype(
                        df=df.index,
                        dtypes=dtypes,
                        app=app,
                        )

    return df


def unitquant(
        dtypes,
        prod=None,
        dt_to_pint=None,
        ):

    if prod is None:
        prod = False

    if dt_to_pint is None:
        dt_to_pint = prod

    if not isinstance(dtypes, pd.Series):
        dtypes = pd.Series(dtypes)

    if dt_to_pint:
        is_dt = is_dt_dtype(dtypes)
        if is_dt.any():
            dtypes = dtypes.astype(str)
            dtypes.loc[is_dt] = (
                            dtypes.loc[is_dt]
                            .str.replace(r",.*\]", "]", regex=True)
                            .str.replace("timedelta64|datetime64", "pint", regex=True)
                            )

    uq = pd.DataFrame(
                1, index=[0], columns=dtypes.index
                ).astype(dtypes).iloc[0]

    if prod:
        return reduce(mul, uq)
    else:
        return uq


def sfunc(
        df,
        func=None,
        keep_dtype=None,
        otype=None,
        ):

    if keep_dtype is None:
        keep_dtype = True
    if keep_dtype:
        otype = getattr(df, "dtypes", otype)

    if func is None:
        def func(df):
            return df

    df = func(to_subdtype(df))
    if otype is not None:
        return df.astype(otype)
    else:
        return df


def sprod(
        df,
        dt_to_pint=None,
        ):

    is_srs = isinstance(df, pd.Series)
    kwargs = {}
    if is_srs:
        df = pd.DataFrame([[v for v in df]])
    else:
        kwargs["axis"] = 1

    uq_prod = unitquant(dtypes=df.dtypes, prod=True, dt_to_pint=dt_to_pint)

    sdf = to_subdtype(df)
    if is_srs:
        prod = sdf.iloc[0].prod()
    else:
        prod = sdf.prod(axis=1)

    return prod * uq_prod


def dtype_to_columns(
        df: pd.DataFrame,
        app: Optional[bool] = None,
        as_index: Optional[bool] = None,
        app_to_subdtype: Optional[bool] = None,
        ):

    if app is None:
        app = True
    if as_index is None:
        as_index = False

    dtypes = df.dtypes.rename("dtype")

    columns = pd.MultiIndex.from_frame(dtypes.reset_index())
    if as_index:
        return columns
    else:
        return to_subdtype(
                    df, dtypes=dtypes, app=app_to_subdtype,
                    ).set_axis(columns, axis=1)


def dtype_from_columns(
        df: pd.DataFrame,
        app: Optional[bool] = None,
        dtype_level: Optional[str] = None,
        ):

    if app is None:
        app = True

    if dtype_level is None:
        dtype_level = "dtype"

    level_df = calc_level_df(df.columns, level=dtype_level, ret="int")
    has_dtype_level = level_df.shape[1] > 0
    if has_dtype_level:
        dtypes = level_df.iloc[:, 0]
    else:
        app = False
        dtypes = df.dtypes.rename("dtype")
    if app:
        df = df.droplevel([*level_df.columns], axis=1).astype(dtypes)

    return df


def clean_index(
            index: pd.Index,
            app_clean: Optional[bool] = None,
            clean_func: Optional[Callable] = None,
            ) -> pd.Index:
    if app_clean is None:
        app_clean = True

    if clean_func is None:
        clean_func = _clean_index_func

    if app_clean:
        index = pd.MultiIndex.from_frame(
                        index.to_frame().map(clean_func)
                        )
        if index.nlevels == 1:
            index = index.get_level_values(0)

    return index


def read_csv(
            *args,
            app_col_clean: Optional[bool] = None,
            col_clean_func: Optional[Callable] = None,
            app_dtype_from_columns: Optional[bool] = None,
            dtype_level: Optional[str] = None,
            **kwargs,
            ) -> pd.DataFrame:

    df = pd.read_csv(*args, **kwargs)

    df = df.set_axis(
                clean_index(
                    df.columns,
                    app_clean=app_col_clean,
                    clean_func=col_clean_func,
                    ),
                axis=1,
                )
    df = dtype_from_columns(
            df=df,
            app=app_dtype_from_columns,
            dtype_level=dtype_level,
            )

    return df


def calc_level(
        index,
        level=None,
        ret=None,
        ):

    if level is None:
        level = index.nlevels - 1

    if ret is None:
        ret = ""
    ret = str(ret).lower()[:1]

    level_names = np.array([*index.names])

    level = np.atleast_1d(level).reshape(-1)
    if level.dtype == int:
        level_i = level[level < index.nlevels]
        level_names = level_names[level_i]
        if not ret:
            ret = "i"
    else:
        isin = np.isin(level_names, level)
        level_names = level_names[isin]
        level_i = np.where(isin)[0]
        if not ret:
            ret = "n"

    if (level_names == None).any():     # noqa: E711
        ret = "i"

    if ret == "n":
        return level_names
    else:
        return level_i


def drop_level(
        index,
        level=None,
        ret=None,
        drop=None,
        ):

    if drop is None:
        drop = True

    if drop and index.nlevels > 1:

        level = calc_level(index=index, level=level, ret=ret)
        if level.size > 0:
            index = index.droplevel([*level])

    return index


def calc_level_df(
        index,
        level=None,
        ret=None,
        ):

    level = calc_level(index=index, level=level, ret=ret)
    if level.dtype == int:
        method = "iloc"
    else:
        method = "loc"

    level_df = getattr(index.to_frame().droplevel([*level], axis=0), method)[:, level]

    return level_df


def capp_level(
        index,
        level_df=None,
        master_index=None,
        level=None,
        ret=None,
        app=None,
        null=None,
        ):

    if app is None:
        app = True

    if app:

        if level_df is not None or master_index is not None:
            if level_df is None:
                level_df = calc_level_df(index=master_index, level=level, ret=ret)

            if null is None:
                null = ""

        level_df = pd.concat([
                    level_df,
                    pd.DataFrame(
                        null,
                        index=index[~index.isin(level_df.index)],
                        columns=level_df.columns,
                        ),
                    ], axis=0).loc[index].reset_index()

        index = pd.MultiIndex.from_arrays(level_df.values.T, names=level_df.columns)

    return index


def dt_str(
            dt: datetime.datetime,
            ) -> str:

    return str(dt.strftime("%Y-%m-%d_%H%M%S"))


def multiindex_align(
                        df_style,
                        align: Literal["top", "bottom", "center"] = None,
                        ):

    return df_style.set_table_styles([
                                        {
                                            "selector": "th",
                                            "props": [
                                                        ("vertical-align", "top"),
                                                        ("text-align", "left"),
                                                         ],
                                        }])


def cartesian_product(
                        df: Union[pd.DataFrame, dict, np.ndarray],
                        ):

    if isinstance(df, pd.DataFrame):
        clmns = df.columns
    elif isinstance(df, dict):
        clmns = df.keys()
    else:
        clmns = np.arange(df.shape[0])

    return pd.DataFrame(list(product(*(df[c] for c in clmns))), columns=clmns)


def calc_metacats(
                    metacats: Optional[Union[
                                pd.DataFrame, list, np.ndarray, object
                                ]] = None,
                    cats: Optional[Union[pd.Index, np.ndarray, list,]] = None,
                    ) -> pd.DataFrame:

    if cats is None:
        cats = np.arange(1)

    if metacats is None:
        metacats = pd.DataFrame(np.eye(len(cats)).astype(bool), columns=cats)
    elif not isinstance(metacats, pd.DataFrame):
        metacats_list = [metacats] if np.isscalar(metacats) else metacats
        metacats = pd.DataFrame(
                                False,
                                index=np.arange(len(metacats_list)),
                                columns=cats)
        for r, c in enumerate(metacats_list):
            metacats.loc[r, c] = True

    return metacats


def index_num_convert(
                index: Union[pd.Index, pd.MultiIndex],
                incl_int=None,
                ):

    iarrays = [
                [
                    *map(
                        lambda v: _num_convert(s=v, incl_int=incl_int),
                        index.get_level_values(i).astype(str)
                        )]
                for i in range(index.nlevels)
            ]

    if isinstance(index, pd.MultiIndex):
        nindex = pd.MultiIndex.from_arrays(iarrays, names=index.names)
    else:
        nindex = pd.Index(iarrays[0], name=index.name)

    return nindex


def describe(
            df,
            percentiles=None,
            include=None,
            exclude=None,
            drop_count=None,
            qformat=None,
            termini=None,
            via_subdtype=None,
            groupby=None,
            transforms=None,
            merge_transform_idx=None,
            ):

    _val = "val"

    if via_subdtype is None:
        via_subdtype = True

    if drop_count is None:
        drop_count = True

    if percentiles is None:
        percentiles = 5

    if isinstance(percentiles, int):
        percentiles = np.linspace(0, 1, percentiles)[1:-1]
    else:
        percentiles = np.atleast_1d(percentiles).reshape(-1)

    if isinstance(qformat, str):
        q = qformat[:1]
        if q.lower() == "q":
            def qformat(v):
                return q + "_" + v.astype(str)
        elif q == "":
            def qformat(v):
                return v.astype(str)
        else:
            qformat = None

    N_percentiles = percentiles.size

    if isinstance(df, pd.DataFrame):
        dtypes = df.dtypes
        if include is not None or exclude is not None:
            if include is None:
                _incl = pd.Series(True, index=dtypes.index)
            else:
                include = pd.concat([
                            pd.Series(None, dtype=d)
                            for d in np.atleast_1d(include).reshape(-1)
                            ], axis=1).dtypes
                _incl = dtypes.isin(include)
            if exclude is not None:
                exclude = pd.concat([
                            pd.Series(None, dtype=d)
                            for d in np.atleast_1d(exclude).reshape(-1)
                            ], axis=1).dtypes
                _incl &= ~dtypes.isin(exclude)

            if groupby is not None:
                _incl[_incl.index.isin(groupby)] = True

            df = df.loc[:, _incl]
            dtypes = dtypes.loc[_incl]
    else:
        dtypes = df.dtype

    if via_subdtype:
        df = to_subdtype(df)

    if termini is None:
        termini = False

    if isinstance(termini, bool):
        if termini:
            termini = [0, 1]
        else:
            termini = None

    if termini is not None:
        if isinstance(termini, dict):
            termini = pd.Series(termini)

        if not isinstance(termini, pd.Series):
            termini = (
                            np.atleast_1d(termini).reshape(-1)
                            .astype(bool).astype(int)
                            )
            termini = pd.Series(
                        {"start": 0, "end": -1}
                        ).iloc[termini]

        termini = termini.astype(int).rename("iloc").rename_axis(_val)

    if transforms is not None:
        if isinstance(transforms, pd.Series):
            tnames = transforms.index.names
        else:
            tnames = ["transform"]
            if not isinstance(transforms, dict):
                tlist = np.atleast_1d(transforms).reshape(-1)
                transforms = {}
                for t in tlist:
                    if callable(t):
                        v = t
                        k = getattr(t, "__name__", "t")
                    elif isinstance(t, str):
                        if t == "":
                            if t in transforms:
                                k = None
                            else:
                                k = t
                            v = None
                        else:
                            v = t
                            k = t
                    else:
                        v = None
                        k = None

                    if k is not None:
                        if k in transforms:
                            i_k = 1
                            while (f"{k}_{i_k}" in transforms):
                                i_k += 1
                            k = f"{k}_{i_k}"

                        transforms[k] = v

        for k, v in transforms.items():

            if not callable(v):
                if v is None:
                    transforms[k] = return_x
                elif isinstance(v, str):
                    transforms[k] = getattr(np, v, None)
                else:
                    transforms[k] = None

        transforms = pd.Series(
                        {k: v for k, v in transforms.items() if v is not None}
                        ).rename_axis(tnames, axis=0)

        if isinstance(merge_transform_idx, bool):
            if merge_transform_idx:
                def merge_transform_idx(v):
                    return "_".join([w for w in v if w])
            else:
                merge_transform_idx = None

    if groupby is not None:
        desc = df.groupby(groupby).apply(
                        lambda v:
                        describe(
                            df=v,
                            percentiles=percentiles,
                            drop_count=drop_count,
                            qformat=qformat,
                            termini=termini,
                            via_subdtype=False,
                            transforms=transforms,
                            merge_transform_idx=merge_transform_idx,
                            )
                        )
    elif transforms is not None:
        scols = dtype_in_sdtypes(
                    dtype=dtypes,
                    sdtypes=[
                            "numeric", "bool",
                            "timedelta64", "datetime64",
                            ],
                    ).any(axis=0)
        print(scols)
        desc = pd.concat(
                transforms.apply(
                    lambda v:
                    describe(
                            df=transform_scols(
                                    df=df, transform=v,
                                    scols=scols,
                                    ),
                            percentiles=percentiles,
                            drop_count=drop_count,
                            qformat=qformat,
                            termini=termini,
                            via_subdtype=False,
                            transforms=None,
                            )
                    ).values,
                axis=0, keys=transforms.index
                )
        if callable(merge_transform_idx):
            mnames = [*tnames, _val]
            idx = desc.iloc[:, :0].reset_index()
            idx = pd.concat([
                        idx.drop(columns=mnames),
                        idx[mnames].apply(merge_transform_idx, axis=1).rename(_val)
                        ], axis=1)
            desc = desc.set_axis(
                    pd.MultiIndex.from_frame(idx),
                    axis=0,
                    )
    else:
        desc = df.describe(
            percentiles=percentiles,
            ).rename_axis(_val, axis=0)

        if drop_count:
            desc = desc.drop(index="count")

        if qformat is not None and N_percentiles > 0:
            if callable(qformat):
                desc = desc.rename(index=dict(zip(
                                    desc.index[-(1 + N_percentiles):-1],
                                    qformat(percentiles),
                                    )))

        if termini is not None:
            desc = pd.concat([
                        desc,
                        df.iloc[termini.values].set_axis(termini.index, axis=0)
                        ])

    if via_subdtype:
        if isinstance(desc, pd.DataFrame):
            dtypes = dtypes.loc[desc.columns]
        desc = desc.astype(dtypes)

    return desc


def dtype_in_sdtypes(
        dtype,
        sdtypes=None,
        ):

    if sdtypes is None:
        sdtypes = ["numeric", "bool", "timedelta64", "datetime64"]
    sdtypes = np.atleast_1d(sdtypes).reshape(-1)

    if len(sdtypes) == 0:
        if isinstance(dtype, pd.Series):
            return pd.Series(False, index=dtype.index)
        else:
            return False
    else:

        sdtypes = pd.Index(sdtypes, name="sdtype")
        sdtypes = sdtypes.to_frame()["sdtype"].apply(
                                    lambda v:
                                    getattr(pd.api.types, f"is_{v}_dtype", False),
                                    )
        if isinstance(dtype, pd.Series):
            return sdtypes.apply(lambda v: dtype.apply(v))
        else:
            return sdtypes.apply(lambda v: v(dtype))


def transform_scols(
        df,
        transform=None,
        scols=None,
        transform_kwargs=None,
        ):

    if transform is None:
        return df

    transform_kwargs = set_default_kwargs(transform_kwargs)

    if isinstance(df, pd.Series):
        if not np.isscalar(scols):
            scols = scols[0]

        if scols:
            return transform(df, **transform_kwargs)
        else:
            return df
    else:
        return pd.concat([
                transform(df.loc[:, scols], **transform_kwargs),
                df.loc[:, ~scols]
                ], axis=1).loc[:, df.columns]


def transform_sdtypes(
        df,
        transform=None,
        sdtypes=None,
        dtype=None,
        transform_kwargs=None
        ):

    if transform is None:
        return df

    if sdtypes is None:
        sdtypes = ["numeric", "bool", "timedelta64", "datetime64"]

    if dtype is None:
        if isinstance(df, pd.Series):
            dtype = df.dtype
        else:
            dtype = df.dtypes

    scols = dtype_in_sdtypes(dtype=dtype, sdtypes=sdtypes).any(axis=0)

    return transform_scols(
            df=df,
            transform=transform,
            scols=scols,
            transform_kwargs=transform_kwargs,
            )


def _calc_val_return(
        val,
        restore_srs=None,
        ):
    if restore_srs is None:
        restore_srs = False

    if restore_srs:
        if isinstance(val, pd.DataFrame):
            val = val.iloc[0]
    return val


def calc_val(
        df_meta,
        val=None,
        app_to_subdtype=None,
        set_columns=None,
        columns_names=None,
        val_columns_names=None,
        sortby=None,
        fill_val=None,
        c_val=None,
        c_dtype=None,
        if_none_in_meta=None,
        restore_srs=None,
        ):

    if restore_srs is None:
        restore_srs = True

    if c_val is None:
        c_val = "val"

    if c_dtype is None:
        c_dtype = "dtype"

    if fill_val is None:
        fill_val = False

    if sortby is None:
        sortby = "val"
    sortby = str(sortby).lower()[:1]

    if app_to_subdtype:
        app_to_subdtype = True

    if set_columns is None:
        if columns_names is None:
            set_columns = False
        else:
            set_columns = False

    has_val = c_val in df_meta.columns
    has_dtype = c_dtype in df_meta.columns

    _drop_cols = []
    if has_val:
        _drop_cols.append(c_val)
    if has_dtype:
        _drop_cols.append(c_dtype)

    _meta_index_cols = df_meta.columns
    if _drop_cols:
        _meta_index_cols = _meta_index_cols.drop(_drop_cols)

    _meta_index = pd.MultiIndex.from_frame(df_meta[_meta_index_cols]).to_frame()
    df_meta = df_meta.set_axis(_meta_index.index, axis=0)

    ow_index = True
    if isinstance(val, pd.DataFrame):
        if val.shape[1] == 0:
            return pd.DataFrame([
                        calc_val(
                            df_meta=df_meta,
                            val=None,
                            app_to_subdtype=app_to_subdtype,
                            set_columns=set_columns,
                            columns_names=columns_names,
                            val_columns_names=val_columns_names,
                            sortby=sortby,
                            fill_val=fill_val,
                            c_val=c_val,
                            c_dtype=c_dtype,
                            if_none_in_meta=if_none_in_meta,
                            restore_srs=True,
                            )
                        ], index=val.index)
        restore_srs = False

    elif isinstance(val, pd.Series) or isinstance(val, dict):
        if len(val) == 0:
            val = None

    if val is None:
        if has_val:
            val = df_meta[[c_val]].T.reset_index(drop=True)
        else:
            val = np.nan
        sortby = "v"
        ow_index = False

    if np.isscalar(val):
        val = pd.Series(val, index=_meta_index.index)
        ow_index = False
    elif isinstance(val, dict):
        val = df_from_kwargs_keys(kwargs=val)

    val_series = isinstance(val, pd.Series)
    if val_series:
        val = val.to_frame().T

    if val.shape[0] != 1:
        restore_srs = False

    if if_none_in_meta is None:
        if_none_in_meta = val.shape[1] == 0
    if not if_none_in_meta:
        if_none_in_meta = "none"
    elif isinstance(if_none_in_meta, bool):
        if_none_in_meta = "all"
    if_none_in_meta = str(if_none_in_meta)[:1].lower()

    if val_columns_names is not None:
        if np.isscalar(val_columns_names):
            val_columns_names = [val_columns_names]
        L = len(val_columns_names)
        L_0 = len(val.columns.names)
        L_1 = L_0 - L
        if L_1 > 0:
            val_columns_names = [
                                    *val.columns.names[:L_1],
                                    *val_columns_names,
                                    ]
        has_name = np.array([n is not None for n in val_columns_names])
        val_columns_names = np.where(
                                    has_name,
                                    val_columns_names[:L_0],
                                    val.columns.names,
                                    )
        val.columns = val.columns.set_names(val_columns_names[:L_0])

    val_columns_names = val.columns.names
    has_name = np.array([n is not None for n in val_columns_names])
    has_all_names = np.all(has_name)

    if not has_all_names:
        vc = cross_vc(
                df=val.columns.to_frame(),
                df_target=_meta_index,
                )
        vc_key = vc.idxmax(axis=1)
        vc_max = vc.max(axis=1)

        val_columns_names = np.where(
                                    has_name | (vc_max == 0),
                                    val_columns_names,
                                    vc_key.values,
                                    )
        val.columns = val.columns.set_names(val_columns_names)
        val_columns_names = val.columns.names

    if ow_index:
        val_lvl_in_meta = np.isin(val.columns.names, df_meta.columns)
        if np.all(~val_lvl_in_meta):
            if if_none_in_meta == "r":
                return _calc_val_return(val, restore_srs=restore_srs)
            else:
                outp = calc_val(
                        df_meta=df_meta,
                        val=val.iloc[:, :0],
                        app_to_subdtype=app_to_subdtype,
                        set_columns=set_columns,
                        columns_names=columns_names,
                        val_columns_names=val_columns_names,
                        sortby=sortby,
                        fill_val=fill_val,
                        c_val=c_val,
                        c_dtype=c_dtype,
                        if_none_in_meta=if_none_in_meta,
                        restore_srs=False,
                        )
                if if_none_in_meta == "n":
                    return outp.iloc[:, :0]
                else:
                    return _calc_val_return(outp, restore_srs=restore_srs)

        elif np.any(~val_lvl_in_meta):
            where_out = np.where(~val_lvl_in_meta)[0]
            old_val_columns = val.columns
            new_val_columns = old_val_columns.drop([int(i) for i in where_out])
            if new_val_columns.nlevels == 1:
                new_val_columns = new_val_columns.get_level_values(0)
            val = val.set_axis(new_val_columns, axis=0)

        _meta_index = pd.MultiIndex.from_frame(df_meta[val.columns.names])

    if isinstance(val.columns, pd.MultiIndex):
        if val.columns.nlevels == 1:
            val = val.set_axis(val.columns.get_level_values(0), axis=1)

    if not isinstance(val.columns, pd.MultiIndex):
        _meta_index = _meta_index.get_level_values(0)

    df_meta = df_meta.set_axis(
                _meta_index, axis=0,
                )

    val_in_meta = val.columns.isin(df_meta.index)
    val = val.loc[:, val_in_meta]
    meta_in_val = df_meta.index.isin(val.columns)

    if fill_val and not meta_in_val.all():
        fill_bool = ~meta_in_val
        if not isinstance(fill_val, bool):
            if np.isscalar(fill_val):
                fill_val = [fill_val]
            fill_val = df_meta.columns[df_meta.columns.isin(fill_val)]
            codes = pd.factorize(df_meta[fill_val].apply(tuple, axis=1))[0]
            ucodes = np.unique(codes[meta_in_val])
            fill_code = np.isin(codes, ucodes)
            fill_bool &= fill_code

        if fill_bool.any():
            if has_val:
                _fill = df_meta.loc[fill_bool, [c_val]].T
            else:
                _fill = pd.Series(
                        np.nan,
                        index=df_meta.index[fill_bool],
                        ).to_frame(c_val).T

            val = val.merge(_fill, how="cross").set_axis(val.index, axis=0)
            meta_in_val |= fill_bool

    if sortby == "v":
        df_meta_val = df_meta.loc[val.columns]
    else:
        df_meta_val = df_meta.loc[meta_in_val]
        val = val.loc[:, df_meta_val.index]

    if has_dtype:
        val = val.astype(df_meta_val[c_dtype])
        if app_to_subdtype:
            val = to_subdtype(val)

    if set_columns:
        if columns_names is None:
            columns_names = df_meta_val.columns.drop(_drop_cols)

        if np.isscalar(columns_names):
            columns_names = [columns_names]

        _new_index = df_meta_val[columns_names].copy()
        _is_obj = _new_index.dtypes == object
        if False:
            _new_index.loc[:, _is_obj] = _new_index.loc[:, _is_obj].astype(str)
        else:
            _new_index.loc[:, _is_obj] = _new_index.loc[:, _is_obj].map(str)

        _new_index = pd.MultiIndex.from_frame(
                        _new_index
                        )
        if _new_index.nlevels == 1:
            _new_index = _new_index.get_level_values(0)

        val = val.set_axis(
                _new_index, axis=1,
                )

    return _calc_val_return(val, restore_srs=restore_srs)
