
"""
Core module for config_cls.dag_frame.dag_frame
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from functools import reduce

# External packages
import numpy as np
import pandas as pd

# Internal packages
from ..core import Configurable, hybridmethod
from ...meta import set_default_kwargs
from ...pd import (   # noqa: F401
                    kwargs_from_df, df_from_kwargs_keys, df_from_kwargs,
                    to_subdtype, cross_vc,
                    calc_val,
                    )
from ...meta import (   # noqa: F401
                    return_x
                    )

# =====================================
# CLASSES
# =====================================


class Meta:

    _dag_levels = None
    _init_depth = 0

    @classmethod
    def _levels(cls):

        dag_levels = getattr(cls, "_dag_levels", None)
        if dag_levels is None:
            return None
        else:
            init_depth = getattr(cls, "_init_depth", 0)
            return dag_levels[init_depth:]

    @classmethod
    def kw(
            cls,
            hkey=None,
            ):
        _kw = cls._kw()
        if hkey is None:
            hkey = _kw.keys()

        return [v for hk in hkey for v in _kw[hk]]

    @classmethod
    def _kw(
            cls,
            ):

        return {}


class Node(Meta, Configurable):

    _frame_id = True

    @classmethod
    def _export_spec_visit_state(cls):
        state = {
                    **super()._export_spec_visit_state(),
                    "_kw": cls._kw(),
                    "_frame_id": cls._frame_id,
                    }
        return state

    @classmethod
    def _import_spec_set_state(
            ccls,
            key,
            value,
            ):

        if key == "_kw":
            @classmethod
            def _kw(cls):
                return value
            setattr(ccls, key, _kw)
        else:
            super()._import_spec_set_state(key, value)

    @classmethod
    def df(
            cls,
            data=None,
            data_0=None,
            force_kw=None,
            app_post_df=None,
            ):

        if app_post_df is None:
            app_post_df = True

        if app_post_df:
            force_kw = True

        if force_kw is None:
            force_kw = False

        if force_kw:
            keys = cls.kw()
        else:
            keys = None

        if data is None:
            data = cls._data()

        if data_0 is None:
            data_0 = cls._data_0()

        df = df_from_kwargs_keys(
                kwargs=data,
                dkwargs=data_0,
                keys=keys,
                fill="",
                skip_dkwargs_if_empty=True,
                )

        if app_post_df:
            df = cls.post_df(df=df)

        return df

    @classmethod
    def post_df(
            cls,
            df=None,
            ):

        if df is None:
            df = cls.df(app_post_df=False, force_kw=True)

        return df

    @classmethod
    def val(
            cls,
            val=None,
            app_to_subdtype=None,
            set_columns=None,
            columns_names=None,
            val_columns_names=None,
            sortby=None,
            fill_val=None,
            c_val=None,
            c_dtype=None,
            df_meta=None,
            if_none_in_meta=None,
            restore_srs=None,
            ):

        if df_meta is None:
            df_meta = cls.post_df()

        return calc_val(
                    df_meta=df_meta,
                    val=val,
                    app_to_subdtype=app_to_subdtype,
                    set_columns=set_columns,
                    columns_names=columns_names,
                    val_columns_names=val_columns_names,
                    sortby=sortby,
                    fill_val=fill_val,
                    c_val=c_val,
                    c_dtype=c_dtype,
                    if_none_in_meta=if_none_in_meta,
                    restore_srs=restore_srs,
                    )

    @classmethod
    def collapse_data(
            cls,
            data,
            collapse=None,
            ):
        if collapse is None:
            collapse = True

        return kwargs_from_df(data, collapse=collapse)

    @classmethod
    def ndata_val(
            cls,
            val=None,
            c_val=None,
            odata=None,
            excl_data_0=None,
            ):

        if excl_data_0 is None:
            excl_data_0 = True

        if c_val is None:
            c_val = "val"

        if odata is None:
            odata = cls._data()
        df = cls.df(data=odata)

        val = cls.val(
                    val=val,
                    df_meta=df,
                    app_to_subdtype=True,
                    c_val=c_val,
                    restore_srs=False,
                    ).T.iloc[:, 0]
        icol = val.index.names
        if len(icol) == 1 and not isinstance(val.index, pd.MultiIndex):
            icol = icol[0]
        df = df.set_axis(df[icol], axis=0)
        val = val.loc[val.index.isin(df.index)]
        df.loc[val.index, c_val] = val

        if excl_data_0:
            df = df.loc[:, ~df.columns.isin(cls._data_0().keys())]

        return cls.collapse_data(df.reset_index(drop=True))

    @classmethod
    def _data(cls):
        return {}

    @classmethod
    def _data_0(cls):
        return {}

    @classmethod
    def _dag_df(
            cls,
            leaf_index=None,
            max_rdepth=None,
            ):

        df = cls.df()
        if leaf_index is not None:
            df = df.set_index(leaf_index)
        else:
            if df.index.nlevels == 1:
                if df.index.name is None:
                    df = df.rename_axis("leaf_index", axis=0)
        return df

    @classmethod
    def _is_node(
            cls,
            object,
            ):

        return getattr(object, "_frame_id", False) == cls._frame_id

    @hybridmethod
    def walk(
            cls,
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
            force_dag=None,
            app_to_srs=None,
            init_depth=None,
            max_rdepth=None,
            null_key=None,
            f_kwargs=None,
            cond_kwargs=None,
            ):

        if is_node is None:
            is_node = cls._is_node

        if is_leaf is None:
            is_leaf = is_leaf_

        return super().walk(
                    path=path,
                    f_path_leaf=f_path_leaf,
                    cond_path_leaf_res=cond_path_leaf_res,
                    path_to_key=path_to_key,
                    result=result,
                    leaf_only=leaf_only,
                    at_empty=at_empty,
                    get_store=get_store,
                    is_node=is_node,
                    is_leaf=is_leaf,
                    leaf_if_no_store=leaf_if_no_store,
                    force_dag=force_dag,
                    app_to_srs=app_to_srs,
                    init_depth=init_depth,
                    max_rdepth=max_rdepth,
                    null_key=null_key,
                    f_kwargs=f_kwargs,
                    cond_kwargs=cond_kwargs,
                    )

    @classmethod
    def calc_merge_data(
            cls,
            new_data=None,
            joinby=None,
            new_rows=None,
            new_cols=None,
            return_if_same=None,
            to_kwargs=None,
            collapse=None,
            data=None,
            walk_kwargs=None,
            force_dag=None,
            ):

        walk_kwargs = set_default_kwargs(
                        walk_kwargs,
                        dkwargs=dict(
                            leaf_only=True,
                            force_dag=force_dag,
                            ),
                        )
        if return_if_same is None:
            return_if_same = False

        walk_kwargs["f_path_leaf"] = call_calc_merge_data_
        walk_kwargs["f_kwargs"] = set_default_kwargs(
                                    walk_kwargs.get("f_kwargs", None),
                                    dkwargs=dict(
                                            new_data=new_data,
                                            joinby=joinby,
                                            new_rows=new_rows,
                                            new_cols=new_cols,
                                            return_if_same=return_if_same,
                                            to_kwargs=to_kwargs,
                                            collapse=collapse,
                                            data=data,
                                            )
                                    )

        ndata = cls.walk(
                **walk_kwargs
                )

        return ndata

    @classmethod
    def update_data(
            cls,
            new_data=None,
            joinby=None,
            new_rows=None,
            new_cols=None,
            return_if_same=None,
            to_kwargs=None,
            collapse=None,
            data=None,
            walk_kwargs=None,
            force_dag=None,
            suffix=None,
            copy_child=None,
            ):

        walk_kwargs = set_default_kwargs(
                        walk_kwargs,
                        dkwargs=dict(
                                cond_path_leaf_res=True,
                                )
                        )

        new_data = cls.calc_merge_data(
                    new_data=new_data,
                    joinby=joinby,
                    new_rows=new_rows,
                    new_cols=new_cols,
                    return_if_same=return_if_same,
                    to_kwargs=to_kwargs,
                    collapse=collapse,
                    data=data,
                    walk_kwargs=walk_kwargs,
                    force_dag=force_dag,
                    )

        if copy_child is None:
            copy_child = False

        if Leaf.typecheck(cls):
            return cls.update_dfuncs(
                        data=new_data,
                        suffix=suffix,
                        copy_child=copy_child,
                        )
        else:
            new_cls = {
                            k:
                            reduce(
                                lambda c, ck: c._config[ck],
                                k,
                                cls
                                ).update_dfuncs(
                                    data=v,
                                    suffix=suffix,
                                    copy_child=copy_child,
                                    )
                            for k, v in new_data.items()
                        }

            return cls.config(
                            new_class=True,
                            suffix=suffix,
                            kwargs=new_cls,
                            copy_child=copy_child
                            )


class Parent(Node):

    @classmethod
    def _child_nodes(cls):

        if hasattr(cls, "_instances"):
            keys = cls._instamces.keys()
        else:
            keys = cls._config.keys()

        keys = [
                        k for k in keys
                        if getattr(cls.get(k), "_frame_id", False) == cls._frame_id
                        ]

        return keys

    @classmethod
    def _data(cls):

        keys = cls._child_nodes()
        data = {
                    **pd.concat([
                        cls.get(c).post_df()
                        for c in keys
                        ])
                    }

        return data

    @classmethod
    def _data_alt(cls):

        return cls._dag_df().reset_index(drop=True)

    @classmethod
    def _dag_df(
            cls,
            max_rdepth=None,
            leaf_index=None,
            ):

        levels = cls._levels()
        init_depth = getattr(cls, "_init_depth", 0)

        if isinstance(max_rdepth, bool):
            if max_rdepth and levels is not None:
                max_rdepth = len(levels)
            else:
                max_rdepth = None

        f_kwargs = dict(
                        leaf_index=leaf_index,
                        )

        df = cls.walk(
                leaf_only=True,
                app_to_srs=True,
                init_depth=init_depth,
                max_rdepth=max_rdepth,
                f_path_leaf=call_dag_df_,
                f_kwargs=f_kwargs,
                )

        N_leaf_levels = df.iloc[0].index.nlevels
        df = pd.concat(df.values, keys=df.index, axis=0)

        if levels is not None:
            N_cut = df.index.nlevels - N_leaf_levels
            df = df.rename_axis(
                        [*levels[:N_cut], *df.index.names[N_cut:]],
                        axis=0,
                        )

        return df


class Leaf(Node):

    @classmethod
    def _export_spec_visit_state(cls):
        state = {
                    **super()._export_spec_visit_state(),
                    "_kw": cls._kw(),
                    "_data": cls.collapse_data(cls._data(), collapse=True),
                    "_data_0": cls.collapse_data(cls._data_0(), collapse=True),
                    }

        return state

    @classmethod
    def _import_spec_set_state(
            ccls,
            key,
            value,
            ):

        dfuncs = ["_kw", "_data", "_data_0"]
        if key in dfuncs:
            @classmethod
            def dfunc(cls):
                return value
            setattr(ccls, key, dfunc)
        else:
            super()._import_spec_set_state(key, value)

    @classmethod
    def _update_defs_dfunc(
            cls,
            defs=None,
            dfunc=None,
            ndict=None,
            clear=None,
            ):

        if clear is None:
            clear = False

        if dfunc is None:
            dfunc = "dfunc"

        if defs is None:
            defs = {}

        if ndict is None:
            ndict = {}

        if len(ndict) > 0 or clear:
            old_dfunc = getattr(cls, dfunc, None)
            if clear or old_dfunc is None:
                @classmethod
                def old_dfunc(cls):
                    return {}

            @classmethod
            def new_dfunc(cls):
                return {
                            **old_dfunc(),
                            **ndict,
                            }

            defs[dfunc] = new_dfunc

        return defs

    @classmethod
    def _dfuncs_defs(
            cls,
            data=None,
            data_0=None,
            kw=None,
            defs=None,
            ):

        defs = cls._update_defs_dfunc(dfunc="_data", ndict=data, defs=defs)
        defs = cls._update_defs_dfunc(dfunc="_data_0", ndict=data_0, defs=defs)
        defs = cls._update_defs_dfunc(dfunc="_kw", ndict=kw, defs=defs)

        return defs

    @classmethod
    def update_dfuncs(
            cls,
            data=None,
            data_0=None,
            kw=None,
            defs=None,
            suffix=None,
            copy_child=None,
            ):

        defs = cls._dfuncs_defs(
                        data=data,
                        data_0=data_0,
                        kw=kw,
                        defs=defs,
                        )

        if len(defs) == 0 and suffix is None:
            return cls

        if copy_child is None:
            copy_child = False

        ncls = cls._clscopy(
                        suffix=suffix,
                        defs=defs,
                        copy_child=copy_child,
                        )

        return ncls

    @classmethod
    def _calc_merge_data(
            cls,
            new_data=None,
            joinby=None,
            new_rows=None,
            new_cols=None,
            return_if_same=None,
            to_kwargs=None,
            collapse=None,
            data=None,
            ):

        if data is None:
            data = cls._data()

        _data_0 = cls._data_0()
        origin_data = cls.df(
                data=data,
                data_0=_data_0,
                app_post_df=False,
                )

        if new_data is None:
            merged = False
        else:
            if new_rows is None:
                new_rows = False
            if new_cols is None:
                new_cols = False

            if not isinstance(new_data, pd.DataFrame):
                new_data = pd.DataFrame(new_data)

            if joinby is not None:
                if np.isscalar(joinby):
                    joinby = [joinby]
                new_data = new_data.set_index([*joinby])

            joinby = new_data.index.names

            joinby = origin_data.columns[origin_data.columns.isin(new_data.index.names)]
            old_data = origin_data.set_index([*joinby])

            new_data = new_data.reset_index().set_index([*joinby])

            new_in_old_rows = new_data.index.isin(old_data.index)
            new_in_old_cols = new_data.columns.isin(origin_data.columns)

            if new_rows:
                _new_rows = new_data.index[~new_in_old_rows]
            else:
                _new_rows = []
                new_data = new_data.loc[new_in_old_rows]

            if new_cols:
                _new_cols = new_data.columns[~new_in_old_cols]
            else:
                _new_cols = []
                new_data = new_data.loc[:, new_in_old_cols]

            new_data_shape = np.array(new_data.shape)
            new_rc_shape = np.array([len(_new_rows), len(_new_cols)])
            any_new_rc = (new_rc_shape > 0).any()

            if not any_new_rc and (new_data_shape == 0).any():
                merged = False
            else:
                merged = True
                merge_data = new_data.combine_first(old_data)
                if not any_new_rc:
                    merge_data = merge_data.loc[old_data.index, old_data.columns]
                    merged = not merge_data.equals(old_data)
                    if merged:
                        try:
                            merged = not merge_data.astype(
                                            old_data.dtypes
                                            ).equals(old_data)
                        except:     # noqa: E722
                            merged = True
                    if merged:
                        merge_data = merge_data.reset_index()
                else:
                    merge_data = pd.concat([
                                merge_data.loc[idx]
                                for idx in [old_data.index, _new_rows]
                                ], axis=0).reset_index()

        if to_kwargs is None:
            to_kwargs = True

        if not merged:
            if return_if_same is None:
                return_if_same = True

            if not return_if_same:
                return None
            else:
                merge_data = data
                if not to_kwargs:
                    merge_data = pd.DataFrame(merge_data)
        else:
            merge_data = pd.concat([
                                merge_data.loc[:, idx]
                                for idx in [origin_data.columns, _new_cols]
                                ], axis=1)

            keys_0 = merge_data.columns[merge_data.columns.isin(_data_0.keys())]
            drop_key = [k for k in keys_0 if np.all(origin_data[k] == _data_0[k])]
            merge_data = merge_data.drop(columns=drop_key)
            merge_data = merge_data.loc[:, ~(merge_data == "").all(axis=0)]

        if to_kwargs:
            merge_data = kwargs_from_df(merge_data, collapse=collapse)

        return merge_data


# =====================================
# FUNCTIONS
# =====================================


def is_leaf_(
        object,
        ):

    return Leaf.typecheck(object)


def call_dag_df_(
        path,
        node,
        **kwargs
        ):

    dag_df = getattr(node, "_dag_df", None)
    if callable(dag_df):
        return dag_df(**kwargs)
    else:
        return pd.DataFrame()


def call_calc_merge_data_(
        path,
        node,
        **kwargs,
        ):

    calc_merge_data = getattr(node, "_calc_merge_data", None)
    if callable(calc_merge_data):
        kwargs = set_default_kwargs(
                    kwargs=kwargs,
                    dkwargs=dict(
                        to_kwargs=True,
                        collapse=True,
                        return_if_same=False,
                        ),
                    )

        return calc_merge_data(
                    **kwargs,
                    )
    else:
        return None
