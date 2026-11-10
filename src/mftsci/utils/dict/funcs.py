"""
Functions module for dictionary utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal, Callable

# External packages
import numpy as np
import pandas as pd

# Internal packages
from .. import arr
from ..meta import return_args_, set_default_kwargs
from ..str import capp_ps as _capp_ps
from ..dict import defaults


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def keys(
            d: Optional[dict] = None,
            array: Optional[bool] = None,
            ) -> Union[np.ndarray, object,]:

    d = defaults.d(d)
    array = defaults.array(array)

    k = [*d.keys()]
    if array:
        return np.array(k)
    else:
        return k


def vals(
            d: Optional[dict] = None,
            array: Optional[bool] = None,
            ) -> Union[np.ndarray, object,]:

    d = defaults.d(d)
    array = defaults.array(array)

    v = [*d.values()]
    if array:
        return np.array(v)
    else:
        return v


def apply_funcs(
        d: Union[dict, Callable,],
        *args,
        **kwargs,
        ):

    if callable(d):
        return d(*args, **kwargs)
    elif isinstance(d, dict):
        return {k: apply_funcs(v, *args, **kwargs) for k, v in d.items()}
    else:
        return d


def has_store(
        node,
        get_store=None,
        store_type=None,
        ):

    return isinstance(
                defaults.get_store(get_store)(node),
                defaults.store_type(store_type),
                )


def node_type(
        node,
        get_store=None,
        store_type=None,
        is_node=None,
        is_leaf=None,
        leaf_if_no_store=None
        ):

    _node_type = ""

    _is_node = defaults.is_node(is_node)(node)
    if _is_node:
        _has_store = has_store(
                        node=node,
                        get_store=get_store,
                        store_type=store_type,
                        )

        if is_leaf is None:
            _is_leaf = not _has_store
        else:
            _is_leaf = is_leaf(node)

        if _is_leaf:
            _node_type = "leaf"
        else:
            if not _has_store:
                if leaf_if_no_store is None:
                    leaf_if_no_store = False
                if leaf_if_no_store:
                    _node_type = "leaf"
            else:
                _node_type = "parent"

    return _node_type


def path_to_key_all(path):
    return path


def path_to_key_last(path):
    return path[-1]


f_path_leaf_both = return_args_


def f_path_leaf_leaf(path, node, **kwargs):
    return node


def f_path_leaf_path(path, node, **kwargs):
    return path


def cond_path_leaf_res_True(path, leaf, res, **kwargs):
    return True


def cond_res_is_not_None(path, leaf, res, **kwargs):
    return res is not None


def pad_depth(
        d: dict,
        max_rdepth: Optional[int] = None,
        null_key: Optional[object] = None,
        ):

    if null_key is None:
        null_key = ""

    rdepth = np.clip([*map(len, d.keys())], a_min=0, a_max=max_rdepth)
    rdepth_max = max(rdepth)

    delta_rdepth = rdepth_max - rdepth
    if (delta_rdepth > 0).any():
        d = {
                (k + (("",) * delta_rdepth[i])): d[k]
                for i, k in enumerate(d.keys())
                }
    return d


def to_srs(
        d,
        init_depth: Optional[int] = None,
        max_rdepth: Optional[int] = None,
        null_key: Optional[object] = None,
        ):

    if init_depth is None:
        init_depth = 0

    d = pd.Series(pad_depth(
            d=d,
            max_rdepth=max_rdepth,
            null_key=null_key,
            ))

    lvl_names = (
            "level_" +
            (init_depth + np.arange(d.index.nlevels)).astype(str)
            )

    return d.rename_axis(lvl_names, axis=0)


def walk(
        node,
        path=None,
        f_path_leaf=None,
        cond_path_leaf_res=None,
        path_to_key=None,
        result=None,
        leaf_only=None,
        at_empty=None,
        get_store=None,
        is_node=None,
        is_leaf=None,
        leaf_if_no_store=None,
        app_to_srs=None,
        init_depth=None,
        max_rdepth=None,
        null_key=None,
        force_dag=None,
        f_kwargs=None,
        cond_kwargs=None,
        ):

    if result is None:
        result = {}

    if path is None:
        path = ()

    if not isinstance(path, tuple):
        if isinstance(path, str):
            path = (path,)
        else:
            path = tuple(path)

    if leaf_only is None:
        leaf_only = not isinstance(result, dict)

    if at_empty is None:
        at_empty = "e"
    if isinstance(at_empty, bool):
        if at_empty:
            at_empty = "i"
        else:
            at_empty = "e"
    at_empty = str(at_empty)[:1].lower()

    if path_to_key is None:
        if leaf_only:
            path_to_key = path_to_key_all
        else:
            path_to_key = path_to_key_last

    if f_path_leaf is None:
        if path_to_key == path_to_key_all:
            f_path_leaf = f_path_leaf_leaf
        else:
            f_path_leaf = f_path_leaf_both

    if isinstance(cond_path_leaf_res, bool):
        if cond_path_leaf_res:
            cond_path_leaf_res = cond_res_is_not_None
        else:
            cond_path_leaf_res = None

    f_kwargs = set_default_kwargs(f_kwargs)
    cond_kwargs = set_default_kwargs(cond_kwargs)

    store_type = dict
    get_store = defaults.get_store(get_store)
    is_node = defaults.is_node(is_node)

    _node_type = node_type(
                        node=node,
                        get_store=get_store,
                        store_type=store_type,
                        is_node=is_node,
                        is_leaf=is_leaf,
                        leaf_if_no_store=leaf_if_no_store,
                        )

    if not _node_type:
        res = None
        app_res = False
    else:
        if _node_type == "parent":
            if not leaf_only:
                res = type(result)()
                app_res = True
            else:
                res = result
                app_res = False

            for key, child in get_store(node).items():
                cpath = path + (key,)
                res = walk(
                            node=child,
                            path=cpath,
                            f_path_leaf=f_path_leaf,
                            cond_path_leaf_res=cond_path_leaf_res,
                            path_to_key=path_to_key,
                            result=res,
                            leaf_only=leaf_only,
                            at_empty=at_empty,
                            get_store=get_store,
                            is_node=is_node,
                            is_leaf=is_leaf,
                            leaf_if_no_store=leaf_if_no_store,
                            force_dag=False,
                            f_kwargs=f_kwargs,
                            cond_kwargs=cond_kwargs,
                            )

            if not leaf_only:
                if len(res) == 0:
                    if at_empty == "e":
                        app_res = False
                    elif at_empty == "l":
                        _node_type = "leaf"

        if _node_type == "leaf":
            try:
                res = f_path_leaf(path, node, **f_kwargs)
                if callable(cond_path_leaf_res):
                    app_res = cond_path_leaf_res(path, node, res, **cond_kwargs)
                else:
                    app_res = True
            except:     # noqa: E722
                res = None
                app_res = False

    if force_dag is None:
        force_dag = False

    ret_val = (
                (len(path) == 0 and not force_dag)
                or (_node_type == "parent" and leaf_only)
                )
    if ret_val:
        result = res
    else:
        if app_res:
            if isinstance(result, list):
                result.append(res)
            elif isinstance(result, tuple):
                result += res,
            else:
                result[path_to_key(path)] = res

    if isinstance(result, dict):

        if app_to_srs is None:
            app_to_srs = False

        if app_to_srs:
            result = to_srs(
                        d=result,
                        init_depth=init_depth,
                        max_rdepth=max_rdepth,
                        null_key=null_key,
                        )

    return result


def f_keys(
            d: Optional[dict] = None,
            f: Optional[Callable] = None,
            args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            ret_dict: Optional[bool] = None,
            ) -> Union[dict, np.ndarray, object,]:

    d = defaults.d(d)
    ret_dict = defaults.ret_dict(ret_dict)

    keys_0 = keys(d=d)
    keys_1 = arr.fmap(a=keys_0, f=f, args=args, kwargs=kwargs)

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

    d = defaults.d(d)
    ret_dict = defaults.ret_dict(ret_dict)

    keys_0 = keys(d=d)
    vals_0 = vals(d=d)

    vals_1 = arr.fmap(a=vals_0, f=f, args=args, kwargs=kwargs)

    if ret_dict:
        return {keys_0[k]: vals_1[k] for k in range(keys_0.size)}
    else:
        return vals_1


def valsort_keys(
                    d: Optional[dict] = None,
                    array: Optional[bool] = None,
                    ) -> Union[np.ndarray, object,]:

    d = defaults.d(d)
    array = defaults.array(array)

    return keys(d=d, array=True)[np.argsort(vals(d=d))]


def valsort(
            d: Optional[dict] = None,
            ret: Optional[Literal["dict", "array", "list"]] = None,
            ) -> Union[dict, np.ndarray, object,]:

    d = defaults.d(d)

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

    d = defaults.d(d)
    if len(d) == 0:
        return d
    else:
        None_rm = defaults.None_rm(None_rm)

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

    d = defaults.d(d)
    keys_rn = defaults.d(d=keys_rn)

    for k in keys_rn:
        d[keys_rn[k]] = d.pop(k)


def capp_ps(
            d: Optional[dict] = None,
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
            ):

    d = defaults.d(d)
    _keys = keys(d)
    _keys_new = _capp_ps(
                            s=_keys, ps=ps, pre=pre, suff=suff,
                            app_ps=app_ps, app_pre=app_pre, app_suff=app_suff,
                            app_null=app_null, mode=mode,
                            constant_values=constant_values,
                            *args, **kwargs,
                            )

    keys_rn = dict(np.transpose([_keys, _keys_new]))

    rename_keys(d=d, keys_rn=keys_rn)


def collapse(
                d: Optional[dict] = None,
                subdictkeys: Optional[Union[np.ndarray, list, object,]] = None,
                ) -> dict:

    d = defaults.d(d)
    subdictkeys = defaults.subdictkeys(subdictkeys)

    return {**filter_keys(d=d, keys_rm=subdictkeys), **{b: d[a][b] for a in subdictkeys
                                                        for b in d[a]}}


def append_if_not_in(
                        d: Optional[dict] = None,
                        keys_check: Optional[Union[np.ndarray, list, object,]] = None,
                        key: Optional[object] = None,
                        value: Optional[object] = None,
                        ):

    d = defaults.d(d)

    if key is None and keys_check is not None:
        key = arr.iterarr(keys_check)[0]

    if key is not None:
        if keys_check is None:
            keys_check = key

        if not np.any(np.isin(keys_check, keys(d))):
            d[key] = value
