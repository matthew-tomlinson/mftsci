"""
Module for dictionary utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Literal, Callable

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils import arr
# from mpoint.utils.str import calc_ps, _app_ps


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def default(
            d: Optional[dict] = None,
            ) -> dict:

    if d is None:
        d = {}
    return d


def keys(
            d: Optional[dict] = None,
            array: Optional[bool] = None,
            ) -> Union[np.ndarray, object,]:

    d = default(d=d)

    if array is None:
        array = True

    k = [*d.keys()]
    if array:
        return np.array(k)
    else:
        return k


def vals(
            d: Optional[dict] = None,
            array: Optional[bool] = None,
            ) -> Union[np.ndarray, object,]:

    d = default(d=d)

    if array is None:
        array = True

    v = [*d.values()]
    if array:
        return np.array(v)
    else:
        return v


def f_keys(
            d: Optional[dict] = None,
            f: Optional[Callable] = None,
            args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            ret_dict: Optional[bool] = None,
            ) -> Union[dict, np.ndarray, object,]:

    d = default(d=d)

    if ret_dict is None:
        ret_dict = True

    keys_0 = keys(d=d)

    keys_1 = arr.f(a=keys_0, f=f, args=args, kwargs=kwargs)

    if ret_dict:
        return {keys_1[k]: d[keys_0[k]] for k in range(keys_0.size)}
    else:
        return keys_1


def f_vals(
            d: Optional[dict] = None,
            f: Optional[Callable] = None,
            args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            ret_dict: Optional[bool] = None,
            ) -> Union[np.ndarray, object,]:

    d = default(d=d)

    if ret_dict is None:
        ret_dict = True

    keys_0 = keys(d=d)
    vals_0 = vals(d=d)

    vals_1 = arr.f(a=vals_0, f=f, args=args, kwargs=kwargs)

    if ret_dict:
        return {keys_0[k]: vals_1[k] for k in range(keys_0.size)}
    else:
        return vals_1


def valsort_keys(
                    d: Optional[dict] = None,
                    array: Optional[bool] = None,
                    ) -> Union[np.ndarray, object,]:

    d = default(d=d)

    if array is None:
        array = True

    return keys(d=d, array=True)[np.argsort(vals(d=d))]


def valsort(
            d: Optional[dict] = None,
            ret: Optional[Literal["dict", "array", "list"]] = None,
            ) -> Union[dict, np.ndarray, object,]:

    d = default(d=d)

    sort_keys = valsort_keys(d=d)
    vs_d = {k: d[k] for k in sort_keys}

    if np.isin(ret, ["list", "array"]):
        return vals(vs_d)
    else:
        return vs_d


def filter_keys(
                d: Optional[dict] = None,
                d_keys: Optional[Union[np.ndarray, list, str,]] = None,
                keys_in: Optional[Union[np.ndarray, list, object,]] = None,
                keys_rm: Optional[Union[np.ndarray, list, object,]] = None,
                None_rm: Optional[Union[np.ndarray, list, object,]] = None,
                ret: Optional[Literal["dict", "array", "list"]] = None,
                repeat: Optional[int] = None,
                make_iterable: Optional[bool] = None,
                ignore_index: Optional[bool] = None,
                index: Optional[Union[
                                        pd.core.indexes.base.Index,
                                        np.ndarray,
                                        list,
                                        ]] = None,
                force_index: Optional[bool] = None,
                ) -> Union[dict, np.ndarray, object,]:

    d = default(d=d)
    if len(d) == 0:
        return d
    else:
        if None_rm is None:
            None_rm = True

        d_keys_0 = keys(d=d, array=True)
        if d_keys is None:
            d_keys = d_keys_0
        if keys_in is None:
            keys_in = d_keys

        if None_rm:
            None_keys = d_keys[np.array([d[k] is None for k in d_keys])]
            if None_keys.size > 0:
                if keys_rm is None:
                    keys_rm = None_keys
                else:
                    keys_rm = np.concatenate([keys_rm, None_keys])

        d_keys = arr.filter(a=d_keys, a_in=keys_in, a_rm=keys_rm)

        if np.size(d_keys) == 0:
            if ret == "keys":
                return d_keys
            elif ret == "concat":
                return np.concatenate([[]])
            else:
                ret_dict = {}
                if ret == "DataFrame":
                    return pd.DataFrame(ret_dict)
                else:
                    return ret_dict
        else:
            df_size = np.size(d[d_keys[0]])

            if ret == "keys":
                return d_keys
            elif ret == "concat":
                return np.concatenate([d[k] for k in d_keys])
            else:
                if ignore_index is None:
                    ignore_index = False
                    if isinstance(d, pd.core.frame.DataFrame) \
                            or isinstance(d, pd.core.series.Series):
                        ignore_index = True

                if make_iterable is None:
                    make_iterable = False
                    if isinstance(d, pd.core.frame.DataFrame) \
                            or isinstance(d, pd.core.series.Series):
                        if ignore_index:
                            make_iterable = True

                if repeat is not None:
                    ret_dict = {k: arr.iterarr(np.repeat(d[k], repeats=repeat))
                                for k in d_keys}
                elif make_iterable:
                    ret_dict = {k: arr.iterarr(d[k]) for k in d_keys}
                else:
                    ret_dict = {k: d[k] for k in d_keys}

                if ret == "DataFrame":
                    if index is None:
                        index = np.arange(df_size)
                        if not ignore_index:
                            if isinstance(d, pd.core.frame.DataFrame) \
                                    or isinstance(d, pd.core.series.Series):
                                if np.isin(type(d.index[0]), [str, np.str_]):
                                    index = [d.name]
                                else:
                                    index = d.index

                    if force_index is None:
                        force_index = False
                        if not ignore_index:
                            if df_size > 1:
                                force_index = True

                    if force_index:
                        return pd.DataFrame(ret_dict, index=index)
                    else:
                        return pd.DataFrame(ret_dict)
                else:
                    return ret_dict


def rename_keys(
                d: Optional[dict] = None,
                keys_rn: Optional[dict] = None,
                ):

    d = default(d=d)
    keys_rn = default(d=keys_rn)

    for k in keys_rn:
        d[keys_rn[k]] = d.pop(k)


def collapse(
                d: Optional[dict] = None,
                subdictkeys: Optional[Union[np.ndarray, list, object,]] = None,
                ) -> dict:

    d = default(d=d)

    if subdictkeys is None:
        subdictkeys = []
    subdictkeys = arr.iterarr(subdictkeys)

    return {**filter_keys(d=d, keys_rm=subdictkeys), **{b: d[a][b] for a in subdictkeys
                                                        for b in d[a]}}


def append_if_not_in(
                        d: Optional[dict] = None,
                        keys_check: Optional[Union[np.ndarray, list, object,]] = None,
                        key: Optional[object] = None,
                        value: Optional[object] = None,
                        ):

    d = default(d=d)

    if key is None and keys_check is not None:
        key = arr.iterarr(keys_check)[0]

    if key is not None:
        if keys_check is None:
            keys_check = key

        if not np.any(np.isin(keys_check, keys(d))):
            d[key] = value
