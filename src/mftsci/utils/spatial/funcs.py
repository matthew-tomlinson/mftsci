"""
Functions module for spatial.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal     # noqa: F401
import itertools

# External packages
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.spatial
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.utils.validation import _is_fitted


# Local
from mftsci.utils.meta import require_optional
from mftsci.utils.spatial import defaults    # noqa: F401
from mftsci.utils.colour import c_vadd_diff
from mftsci.utils.plot import edge_scatter
from mftsci.utils.sklearn.preprocessing import UnitaryScaler, MinMaxScaler
from mftsci.utils.pd import calc_val


# =====================================
# CLASSES
# =====================================


class _PointCloud_:

    def prep_X(
            self,
            X,
            transformer=None,
            app_transform=None,
            enforce_feature_names_in=None,
            ):

        if transformer is None:
            transformer = getattr(self, "_transformer", None)

        featuee_names_in = getattr(transformer, "feature_names_in_", None)
        ndim = getattr(self, "ndim", 1)

        if isinstance(X, pd.Series):
            X = X.to_frame()
            if ndim > 1:
                if X.shape[0] == ndim:
                    X = X.T

        if isinstance(X, pd.DataFrame):
            if enforce_feature_names_in is None:
                enforce_feature_names_in = False
            if featuee_names_in is None:
                enforce_feature_names_in = False
            if enforce_feature_names_in:
                X = X.reindex(featuee_names_in, axis=1)
        else:
            X = np.atleast_1d(X)
            if len(X.shape) == 1 and ndim == 1:
                X = X.reshape(-1, 1)

        if app_transform is None:
            app_transform = transformer is not None

        if app_transform:
            if transformer is None:
                transformer = UnitaryScaler()
            elif isinstance(transformer, type):
                transformer = transformer()

            if not _is_fitted(transformer, attributes=["n_features_in_"]):
                X = transformer.fit_transform(X)
            else:
                X = transformer.transform(X)

        if isinstance(X, pd.DataFrame):
            columns = X.columns
            index = X.index
            X = X.values
        else:
            index = None
            columns = None

        return X, index, columns, transformer


class _PointCloudBeta_:

    def __init__(
            self,
            key_pref=None,
            key_i_0=None,
            *args,
            **kwargs,
            ):

        self._set_key_info(
                key_pref=key_pref,
                key_i_0=key_i_0,
                reset=True,
                )

    @property
    def N_points(self):
        return 0

    def _set_key_info(
            self,
            key_pref=None,
            key_i_0=None,
            reset=None,
            ):

        reset = defaults.reset(reset)

        if key_pref is not None:
            self._key_pref = str(key_pref)
        elif reset:
            self._key_pref = "B_"

        if key_i_0 is not None:
            self._key_i_0 = int(key_i_0)
        elif reset:
            self._key_i_0 = 0

    def _get_keys(
            self,
            pref=None,
            i_0=None,
            ):

        i_s = np.arange(self.N_points)
        if pref is None:
            pref = getattr(self, "_key_pref", "")

        if i_0 is None:
            i_0 = int(getattr(self, "_key_i_0", 0))

        keys = i_0 + i_s
        if pref:
            keys = pref + keys.astype(str)

        keys = pd.Index(keys, name="key")

        return keys

    def _get_index(
            self,
            ):

        i_s = np.arange(self.points.shape[0])
        index = pd.Index(i_s, name="i")

        return index

    def _beta_df_meta(
            self,
            dval=None,
            dtype=None,
            drop_keys=None,
            drop_index=None,
            key_pref=None,
            key_i_0=None,
            ):

        if dval is None:
            dval = 0.0

        if dtype is None:
            dtype = "float"

        keys = self._get_keys(pref=key_pref, i_0=key_i_0)
        df_meta = self._get_index().to_frame().set_axis(keys, axis=0)

        if drop_keys is None:
            drop_keys = False

        if drop_index is None:
            drop_index = False

        if drop_index:
            df_meta = df_meta.iloc[:, :0]

        df_meta["dtype"] = dtype
        df_meta["val"] = dval

        return df_meta.reset_index(drop=drop_keys)

    def beta(
            self,
            val=None,
            dval=None,
            dtype=None,
            drop_keys=None,
            drop_index=None,
            fill_val=None,
            restore_srs=None,
            key_pref=None,
            key_i_0=None,
            ):

        if val is not None:
            if np.isscalar(val):
                if val.ndim == 0:
                    dval = val
                    val = None

        df_meta = self._beta_df_meta(
                        dval=dval,
                        dtype=dtype,
                        drop_keys=drop_keys,
                        drop_index=drop_index,
                        key_pref=key_pref,
                        key_i_0=key_i_0,
                        )

        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                val = val.reshape(1, val.shape[0])
            elif val.ndim > 2:
                val = np.moveaxis(val, 1, 0).reshape(val.shape[1], -1).T
            index = df_meta.iloc[:val.shape[1]].drop(columns="val")
            val = pd.DataFrame(val, columns=index)

        val = calc_val(
                    df_meta=df_meta,
                    val=val,
                    set_columns=True,
                    fill_val=fill_val,
                    sortby="m",
                    restore_srs=restore_srs,
                    )
        return val


class _PointCloudSetBeta(_PointCloudBeta_):

    def __init__(
            self,
            P,
            key_pref=None,
            key_i_0=None,
            *args,
            **kwargs):

        if not isinstance(P, pd.Series):
            P = pd.Series(P)

        self.P = P

        super().__init__(
                key_pref=key_pref,
                key_i_0=key_i_0,
                )

    @property
    def N_points(self):
        return self.P.apply(lambda v: v.N_points).sum()

    def _get_index(
            self,
            ):

        indexes = self.P.apply(lambda v: v._get_index())

        return indexes

    def _set_key_info(
            self,
            key_pref=None,
            key_i_0=None,
            reset=None,
            ):

        super()._set_key_info(key_pref=key_pref, key_i_0=key_i_0, reset=reset)
        N_points = self.P.apply(lambda v: v.N_points)
        key_i_0_s = self._key_i_0 + N_points.cumsum() - N_points.iloc[0]

        for idx in self.P.index:
            self.P.loc[idx]._set_key_info(
                                key_pref=self._key_pref,
                                key_i_0=key_i_0_s.loc[idx],
                                reset=False
                                )

    def _beta_df_meta(
            self,
            dval=None,
            dtype=None,
            drop_keys=None,
            drop_index=None,
            key_pref=None,
            key_i_0=None,
            ):

        if dval is None:
            dval = 0.0

        if dtype is None:
            dtype = "float"

        if drop_keys is None:
            drop_keys = False

        if drop_index is None:
            drop_index = False

        self._set_key_info(
                key_pref=key_pref,
                key_i_0=key_i_0,
                reset=False,
                )

        tabs = self.P.apply(lambda v: v._beta_df_meta(
                                dval=dval,
                                dtype=dtype,
                                drop_keys=drop_keys,
                                drop_index=drop_index,
                                ))
        df_meta = pd.concat(
                    tabs.values,
                    keys=tabs.index
                    ).droplevel(-1, axis=0).reset_index()

        return df_meta

    def beta(
            self,
            val=None,
            dval=None,
            dtype=None,
            drop_keys=None,
            drop_index=None,
            fill_val=None,
            restore_srs=None,
            key_pref=None,
            key_i_0=None,
            ):

        if fill_val is None:
            fill_val = self.P.index.names

        return super().beta(
                        val=val,
                        dval=dval,
                        dtype=dtype,
                        drop_keys=drop_keys,
                        drop_index=drop_index,
                        fill_val=fill_val,
                        restore_srs=restore_srs,
                        key_pref=key_pref,
                        key_i_0=key_i_0,
                        )


class _PointCloud(_PointCloudBeta_, _PointCloud_):

    @classmethod
    def from_beta_srs(
            cls,
            beta,
            *args,
            **kwargs,
            ):

        points = beta.index.to_frame().reset_index(drop=True)
        beta = beta.reset_index(drop=True)

        instance = cls(points=points, *args, **kwargs)
        # instance.beta = instance.prep_beta(beta=beta)[0]

        return instance

    def __init__(
                self,
                points=None,
                transformer=None,
                key_pref=None,
                key_i_0=None,
                *args,
                **kwargs,
                ):

        _PointCloudBeta_.__init__(self, key_pref=key_pref, key_i_0=key_i_0)

    @property
    def N_points(self):
        return self.points.shape[0]

    def _get_index(
            self,
            ):

        index = getattr(self, "_index", None)

        if index is None:
            index = super()._get_index()

        return index

    def prep_beta(
            self,
            beta,
            v_index=None,
            na_value=None
            ):

        if v_index is None:
            v_index = self._index

        beta_columns = getattr(beta, "columns", None)
        if hasattr(beta, "values"):
            if v_index is not None:
                if na_value is None:
                    na_value = 0.0
                beta = beta.reindex(v_index).fillna(na_value).values
            else:
                beta = beta.values

        beta = np.atleast_1d(beta)

        beta_1d = beta.ndim == 1
        if beta_1d:
            beta = beta.reshape(-1, 1)

        if beta.dtype == int:
            beta = beta.astype(float)

        return beta, beta_columns

    def get_points(
            self,
            app_transform=None,
            force_df=None
            ):

        if app_transform is None:
            app_transform = False

        if force_df is None:
            force_df = False

        P = self.points
        if not app_transform:
            P = self._transformer.inverse_transform(self.points)

        if not isinstance(P, pd.DataFrame):
            if force_df:
                if app_transform:
                    columns = self._columns
                else:
                    columns = None
                P = pd.DataFrame(P, index=self._index, columns=columns)
        else:
            P = P.set_axis(self._index, axis=0)

        return P

    @property
    def Points(
            self,
            ):

        return self.get_points(app_transform=False)

    @property
    def edges(
            self,
            ):

        return np.array(list({
                tuple(sorted(edge))
                for s in self.simplices
                for edge in itertools.combinations(s, 2)
                }))

    def _set_points_1D(
            self,
            points,
            ):

        self.ndim = 1
        self._points = points
        self._argsort = np.argsort(self.points[:, 0])
        self._argsort_r = np.argsort(self._argsort)

    def plot_graph(
            self,
            beta=None,
            beta_to_points=None,
            edges_kwargs=None,
            points_kwargs=None,
            plot_edges=None,
            plot_points=None,
            app_transform=None,
            ax=None,
            ):

        if app_transform is None:
            app_transform = False

        edges_kwargs = defaults.set_default_kwargs(edges_kwargs)
        points_kwargs = defaults.set_default_kwargs(points_kwargs)

        plot_edges = plot_edges or edges_kwargs.pop("plot_edges", None)
        plot_points = plot_points or points_kwargs.pop("plot_points", None)

        res_edges = self.plot_edges(
                            ax=ax,
                            plot_edges=plot_edges,
                            beta=beta, beta_to_points=beta_to_points,
                            app_transform=app_transform,
                            **edges_kwargs
                            )
        res_points = self.plot_points(
                        ax=ax,
                        plot_points=plot_points,
                        beta=beta, beta_to_points=beta_to_points,
                        app_transform=app_transform,
                        **points_kwargs
                        )

        return res_edges, res_points

    def _calc_N_axes(
            self,
            ax=None,
            ):

        if ax is None:
            N_axes = np.clip(self.points.shape[1], a_min=2, a_max=3)
        else:
            N_axes = 3 if isinstance(ax, Axes3D) else 2

        return N_axes

    def _convex_hull_edge(self):

        if self.ndim < self.ndim_embed:
            M = self.points.shape[0]
            if M == 0:
                return np.zeros((0, 0), dtype=int)
            elif M == 1:
                return np.array([[0]])
            else:
                arange = np.arange(M)
                return np.array([
                                    np.delete(arange, i)
                                    for i in arange[::-1]
                                    ])
        elif self.ndim == 1:
            if self._argsort.shape[0] == 1:
                return self._argsort[[0]].reshape(-1, 1)
            else:
                return self._argsort[[0, -1]].reshape(-1, 1)
        else:
            return None

    def _make_points(
            self,
            points=None,
            beta=None,
            beta_to_points=None,
            N_axes=None,
            ax=None,
            ):

        if points is None:
            points = self.points

        if N_axes is None:
            N_axes = self._calc_N_axes(ax=ax)

        if beta_to_points is None:
            beta_to_points = points.shape[-1] < N_axes

        if beta_to_points:
            points = np.hstack([points, beta[:, [0]]])

        return points

    def plot_points(
            self,
            beta=None,
            beta_to_points=None,
            plot_points=None,
            app_transform=None,
            ax=None,
            **kwargs,
            ):

        if plot_points is None:
            plot_points = True

        if app_transform is None:
            app_transform = False

        if plot_points:
            kwargs.setdefault("N_edges", 3)
            N_axes = self._calc_N_axes(ax=ax)

            points = self.get_points(
                            app_transform=app_transform, force_df=True
                            ).astype(float).values
            if False:
                if beta is None:
                    beta = getattr(self, "beta", None)

                if beta is not None:
                    beta, beta_columns = self.prep_beta(beta)
                    kwargs.setdefault("c", beta[:, 0])
                    points = self._make_points(
                                points=points, beta=beta,
                                beta_to_points=beta_to_points, N_axes=N_axes
                                )
            else:
                if beta is not None:
                    beta = self.beta(
                                val=beta, fill_val=True, restore_srs=False,
                                ).iloc[0].values
                    kwargs.setdefault("c", beta)
                    points = self._make_points(
                                    points=points, beta=beta.reshape(-1, 1),
                                    beta_to_points=beta_to_points, N_axes=N_axes
                                    )

            scatters = edge_scatter(*points.T, ax=ax, **kwargs)
        else:
            scatters = None

        return scatters

    def plot_edges(
            self,
            beta=None,
            beta_to_points=None,
            plot_edges=None,
            app_transform=None,
            ax=None,
            **kwargs,
            ):

        if plot_edges is None:
            plot_edges = True

        if app_transform is None:
            app_transform = False

        if isinstance(plot_edges, bool) and plot_edges:
            plot_edges = "line"

        if plot_edges:
            plot_edges = str(plot_edges)[:1].lower()

            edgecolor = c_vadd_diff(v=1.0)
            edgecolor[..., -1] = 0.75

            dkwargs = {
                        "lw": 1,
                        "color": edgecolor,
                        }

            if plot_edges == "p":
                _mode_attr = "Poly"
                _i_attr = "simplices"
                dkwargs["lw"] = 0
                dkwargs["facecolor"] = tuple([0.0]*4)
            else:
                _mode_attr = "Line"
                _i_attr = "edges"
                dkwargs["zorder"] = 1

            kwargs = defaults.set_default_kwargs(
                kwargs=kwargs,
                dkwargs=dkwargs,
                )

            N_axes = self._calc_N_axes(ax=ax)
            if N_axes == 3:
                _D_attr = "3D"
            else:
                _D_attr = ""

            _attr = f"{_mode_attr}{_D_attr}Collection"

            Collect = require_optional(module="mpl_toolkits.mplot3d.art3d", attr=_attr)
            _pmethod = f"add_collection{_D_attr.lower()}"

            points = self.get_points(
                            app_transform=app_transform, force_df=True
                            ).astype(float).values
            if False:
                if beta is None:
                    beta = getattr(self, "beta", None)
                if beta is not None:
                    beta, beta_columns = self.prep_beta(beta)
                    points = self._make_points(
                                points=points, beta=beta,
                                beta_to_points=beta_to_points, N_axes=N_axes
                                )
            else:
                if beta is not None:
                    beta = self.beta(
                                val=beta, fill_val=True, restore_srs=False,
                                ).iloc[0].values
                    points = self._make_points(
                                    points=points, beta=beta.reshape(-1, 1),
                                    beta_to_points=beta_to_points, N_axes=N_axes
                                    )

            if False:
                return [ax.fill(*points[s].T, **kwargs) for s in self.simplices]

            collect = Collect(
                        points[getattr(self, _i_attr)],
                        **kwargs,
                        )

            getattr(ax, _pmethod)(collect)
            ax.autoscale()
        else:
            collect = None

        return collect

    def _calc_hull_pinv(
            self,
            hull=None,
            ):

        if hull is None:
            hull_pinv = None

        elif self.ndim > 1:

            V = self.points[hull]

            # Anchor point for each facet.
            V0 = V[:, :1, :]

            # Tangent basis.
            A = V[:, 1:, :] - V0

            # One pseudoinverse per facet.
            hull_pinv = np.array([
                np.linalg.pinv(A_i)
                for A_i in A
                ])

        else:
            hull_pinv = None

        return hull_pinv


class _Hull(_PointCloud):

    @classmethod
    def from_delaunay(
            cls,
            delaunay,
            *args,
            **kwargs,
            ):

        if hasattr(delaunay, "Points"):
            points = delaunay.Points
        else:
            points = delaunay.points

        self = cls(
                points=points,
                transformer=getattr(delaunay, "_transformer", None),
                *args,
                **kwargs,
                )

        for attr in ["_index", "_columns"]:
            setattr(self, attr, getattr(delaunay, attr, None))

        self._from_delaunay(delaunay=delaunay)

        M, ndim = self.points.shape
        if M <= ndim:
            self.equations = None
        else:
            if not hasattr(self, "equations"):
                self.equations = self._build_equations()

        return self

    def _from_delaunay(
            self,
            delaunay,
            ):
        pass

    @property
    def min_hull(self):
        return self.simplices

    @property
    def hull_equations(self):
        return self.equations

    def _build_equations(self):

        N = self.points.shape[1]
        centre = self.points.mean(axis=0)
        N_simplex = self.simplices.shape[0]

        if N == 1:
            sign = np.array([-1, 1])[:N_simplex].reshape(-1, 1)

            equations = np.hstack([
                                sign,
                                -sign * self.points[self.simplices, 0],
                                ])
        else:
            equations = np.full(
                        (self.simplices.shape[0], N + 1),
                        np.nan,
                        )
            for i, s in enumerate(self.simplices):
                X = self.points[s]
                c = X.mean(axis=0)
                T = X[1:] - X[0]
                vh = np.linalg.svd(T)[2]
                n = vh[-1]
                n /= np.linalg.norm(n)

                if np.dot(n, centre - c) > 0:
                    n *= -1

                b = -np.dot(n, X[0])

                equations[i, :-1] = n
                equations[i, -1] = b

        return equations

    def calc_s(
            self,
            X,
            transformer=None,
            app_transform=None
            ):

        X, index = self.prep_X(
                                X,
                                transformer=transformer,
                                app_transform=app_transform,
                                )[:2]

        equations = getattr(self, "hull_equations", None)
        if equations is None:
            s = np.full(X.shape[0], np.nan)
        else:
            s = X @ equations[:, :-1].T + equations[:, -1]

        if index is not None:
            s = pd.DataFrame(s, index=index)

        return s

    def find_facet(
            self,
            X,
            to_bool=None,
            transformer=None,
            app_transform=None,
            ):

        if to_bool is None:
            to_bool = False

        if not to_bool:
            s = self.calc_s(
                    X,
                    transformer=transformer,
                    app_transform=app_transform,
                    )
            if isinstance(s, pd.DataFrame):
                facet = s.idxmax(axis=1)
            else:
                facet = np.argmax(s, axis=-1)

            return facet
        else:
            X, index = self.prep_X(
                                X,
                                transformer=transformer,
                                app_transform=app_transform,
                                )[:2]

            # Primary criterion:
            # Maximum alignment with hull hyperplane
            s = self.calc_s(X, app_transform=False)
            s_max = np.max(s, axis=-1, keepdims=True)
            facet_bool = s == s_max

            if not to_bool:
                facet_bool = np.argmax(facet_bool, axis=-1)
                if index is not None:
                    facet_bool = pd.Series(facet_bool, index=index)
            else:
                if index is not None:
                    facet_bool = pd.DataFrame(facet_bool, index=index)

            return facet_bool

    def _set_dhull_pinv(
            self,
            ):

        if not hasattr(self, "_dhull_pinv"):
            self._dhull_pinv = self._calc_hull_pinv(
                                    hull=getattr(self, "dhull", None)
                                    )


class BoundaryHull(_Hull):

    def _from_delaunay(
            self,
            delaunay
            ):

        for attr in ["ndim", "ndim_embed"]:
            setattr(self, attr, getattr(delaunay, attr, None))

        self.simplices = delaunay.convex_hull


class ConvexHull(_Hull, scipy.spatial.ConvexHull):

    def __init__(
                    self,
                    points,
                    transformer=None,
                    *args,
                    **kwargs,
                    ):

        _Hull.__init__(
                    self, points=points, transformer=transformer,
                    *args, **kwargs,
                    )

        points, index, columns, transformer = self.prep_X(
                                                points,
                                                transformer=transformer,
                                                app_transform=True,
                                                )
        self._index = index
        self._columns = columns
        self._transformer = transformer

        M, ndim = points.shape
        self.ndim_embed = ndim
        self.ndim = min(ndim, M - 1)

        if ndim > 1 and M > ndim:
            scipy.spatial.ConvexHull.__init__(
                self, points=points,
                *args, **kwargs,
                )
        else:
            if ndim == 1:
                self._set_points_1D(points=points)
            else:
                self._points = points

            self.simplices = self._convex_hull_edge()
            if M <= 1:
                self.equations = None
            else:
                sign = np.array([-1, 1])[:self.simplices.size].reshape(-1, 1)
                self.equations = np.hstack([
                                    sign,
                                    -sign * self.points[self.simplices, 0],
                                    ])


class Delaunay(_PointCloud, scipy.spatial.Delaunay):

    @classmethod
    def from_arange_binary_tri(
            cls,
            *args,
            ndim: Optional[int] = None,
            transformer=None,
            X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            **kwargs,
            ):

        U, transformer = arange_binary_tri(
                            *args,
                            ndim=ndim,
                            transformer=transformer,
                            X=X,
                            ret_transformer=True,
                            app_transform=True
                            )

        self = cls(
                points=U,
                transformer=None,
                **kwargs,
                )
        if transformer is not None:
            self._transformer = transformer

        return self

    def __init__(
                    self,
                    points,
                    transformer=None,
                    set_mhull=None,
                    *args,
                    **kwargs,
                    ):

        if set_mhull is None:
            set_mhull = True

        _PointCloud.__init__(
                    self, points=points,
                    transformer=transformer,
                    *args, **kwargs,
                    )

        points, index, columns, transformer = self.prep_X(
                                            points,
                                            transformer=transformer,
                                            app_transform=True,
                                            )
        self._index = index
        self._columns = columns
        self._transformer = transformer

        M, ndim = points.shape
        self.ndim_embed = ndim

        if M <= ndim:
            self.ndim = M - 1
            self._points = points
            self.simplices = np.arange(M).reshape(1, -1)

            if M > 1:
                A = np.linalg.pinv((points[:-1] - points[-1]).T)
            else:
                A = np.empty((0, ndim), dtype=float)

            if M > 0:
                self._transform = np.vstack([
                                    A,
                                    points[[-1]],
                                    ]).reshape(1, M, ndim)
            else:
                self._transform = A.reshape(1, M, ndim)

            self.neighbors = -np.ones((1, M), dtype=int)
            self.equations = None

        else:
            if np.shape(points)[1] == 1:
                self._set_points_1D(points=points)

                self.simplices = np.c_[self._argsort[:-1], self._argsort[1:]]
                self.equations = None

                N_simplex = self.simplices.shape[0]
                self.neighbors = (
                                np.arange(N_simplex).reshape(-1, 1)
                                + np.array([-1, 1]).reshape(1, -1)
                                )
                self.neighbors[-1, -1] = -1
                x_simplex = self.points[self.simplices, 0]

                self._transform = np.hstack([
                    -1.0/np.diff(x_simplex, axis=1)[:, 0].reshape(-1, 1),
                    x_simplex[:, [-1]]
                    ]).reshape(N_simplex, 2, 1)

            else:
                scipy.spatial.Delaunay.__init__(
                    self, points=points, *args, **kwargs,
                    )

        if M <= ndim:
            N_chull = self.simplices.shape[0]
            self.chull_to_simplices = np.zeros(N_chull, dtype=int)
            self.chull_to_mhull = np.zeros(N_chull, dtype=int)
        else:
            idx_hull_simplex = np.where(np.any((self.neighbors == -1), axis=1))[0]
            hull_simplex = self.simplices[idx_hull_simplex]

            self.chull_to_simplices = np.array([
                idx_hull_simplex[
                    np.isin(hull_simplex, facet).astype(int).sum(axis=1).argmax()
                    ]
                for facet in self.convex_hull
                ]).ravel()

            self.chull_to_mhull = None

        if set_mhull:
            self.set_mhull()

    def set_mhull(self):

        if not hasattr(self, "mhull"):

            self.mhull = ConvexHull.from_delaunay(delaunay=self)
            M, ndim = self.points.shape
            if M > ndim:
                self.chull_to_mhull = self.mhull.find_facet(
                                        self.points[self.convex_hull].mean(axis=-2),
                                        to_bool=False,
                                        app_transform=False,
                                        )

    @property
    def min_hull(self):
        self.set_mhull()
        return self.mhull.simplices

    @property
    def hull_equations(self):
        self.set_mhull()
        return self.mhull.equations

    @property
    def convex_hull(self):
        hull = self._convex_hull_edge()
        if hull is None:
            hull = super().convex_hull
        return hull

    def find_facet(
            self,
            X,
            to_bool=None,
            to=None,
            tiebreak=None,
            transformer=None,
            app_transform=None,
            ):

        if to is None:
            to = "chull"
        to = str(to).lower()[:1]

        self.set_mhull()
        if to == "m":
            return self.mhull.find_facet(
                        X=X, to_bool=to_bool,
                        transformer=transformer,
                        app_transform=app_transform,
                        )

        if to_bool is None:
            to_bool = True

        X, index = self.prep_X(
                                X,
                                transformer=transformer,
                                app_transform=app_transform,
                                )[:2]
        # Primary criterion:
        # Maximum alignment with hull hyperplane
        mhull_facet_bool = self.mhull.find_facet(X=X, to_bool=True, app_transform=False)

        facet_bool = mhull_facet_bool[:, self.chull_to_mhull]

        if not to_bool:
            tiebreak = True

        if tiebreak is None:
            tiebreak = True

        if tiebreak:
            N_cand_facet = facet_bool.sum(axis=1)
            ambiguous = N_cand_facet > 1
            hull = self.convex_hull

            if np.any(ambiguous):
                if not hasattr(self, "_points_on_hull"):
                    self._points_on_hull = self.mhull.calc_s(
                                                self.points,
                                                app_transform=False,
                                                ) == 0
                    self._hull_coplanar_points = np.where(
                                                np.any(
                                                    self._points_on_hull,
                                                    axis=1,
                                                    )
                                                )[0]

                cand_points = (
                        mhull_facet_bool[ambiguous] @
                        self._points_on_hull[self._hull_coplanar_points].T
                        )
                any_cand_points = cand_points.any(axis=0)
                where_any_cand_points = np.where(any_cand_points)[0]

                ndim = self.points.shape[-1]

                _X = X[ambiguous]
                cand_distance_sq = np.where(
                    cand_points[:, where_any_cand_points],
                    np.sum(
                        (
                            getattr(_X, "values", _X).reshape(-1, 1, ndim)
                            - (
                                self.points[
                                    self._hull_coplanar_points
                                    ][where_any_cand_points]
                                ).reshape(1, -1, ndim)
                            )**2, axis=-1),
                    np.nan,
                    )
                nearest_point = self._hull_coplanar_points[
                                    where_any_cand_points[
                                        np.nanargmin(cand_distance_sq, axis=1)
                                        ]
                                    ]

                facet_bool[ambiguous] = np.vstack([
                                    facet_bool[i] & np.any(hull == v, axis=1)
                                    for i, v in enumerate(nearest_point)
                                    ])

        if to == "s":
            facet_bool = np.vstack([
                                np.any(
                                    facet_bool[:, self.chull_to_simplices == i],
                                    axis=-1,
                                    )
                                for i in range(self.simplices.shape[0])
                                ]).T
            if tiebreak:
                N_cand_simplex = facet_bool.sum(axis=1)
                ambiguous = N_cand_simplex > 1
                if np.any(ambiguous):
                    where_ambiguous = np.where(ambiguous)[0]
                    s = np.where(np.any(facet_bool[ambiguous], axis=0))[0]
                    bary = self._barycentric(
                            X=X[ambiguous],
                            simplex_ids=s.reshape(*[1]*(X.ndim - 1), -1),
                            app_transform=False,
                            )
                    bary = np.maximum(bary, 0)
                    bary /= bary.sum(axis=1, keepdims=True)

                    x_proj = np.einsum(
                        "...si,sid->...sd",
                        bary,
                        self.points[self.simplices[s]],
                        )

                    simplex = s[
                            np.argmin(
                                np.sum((X[ambiguous][:, None, :] - x_proj)**2, axis=-1),
                                axis=-1,
                                )
                            ]

                    facet_bool[ambiguous, :] = False
                    facet_bool[where_ambiguous, simplex] = True

        if not to_bool:
            facet_bool = np.argmax(facet_bool, axis=-1)
            if index is not None:
                facet_bool = pd.Series(facet_bool, index=index)
        else:
            if index is not None:
                facet_bool = pd.DataFrame(facet_bool, index=index)

        return facet_bool

    def find_simplex(
                    self,
                    xi,
                    extrap=None,
                    transformer=None,
                    app_transform=None,
                    *args,
                    **kwargs,
                    ):

        if extrap is None:
            extrap = False

        xi, index = self.prep_X(
                                    xi,
                                    enforce_feature_names_in=True,
                                    transformer=transformer,
                                    app_transform=app_transform,
                                    )[:2]

        if self.ndim < self.ndim_embed:
            N = xi.shape[0]
            if extrap:
                simplex = np.zeros(N, dtype=int)
            else:
                simplex = -np.ones(N, dtype=int)
        else:
            if self.ndim == 1:
                X_shape = xi.shape
                xi = xi.reshape(-1)

                N_points = np.size(self._points)

                simplex = np.sum(
                            np.greater.outer(xi, self._points[:, 0]).astype(int),
                            axis=-1
                            ) - 1
                if extrap:
                    simplex = np.clip(simplex, a_min=0, a_max=N_points - 2)
                else:
                    simplex[simplex == N_points - 1] = -1

                if False:
                    simplex = simplex.reshape(X_shape)
            else:
                simplex = super().find_simplex(xi=xi, *args, **kwargs)
                if extrap:
                    outside = simplex < 0
                    if np.any(outside):
                        simplex[outside] = (
                                self._extrap_simplex(xi[outside], app_transform=False)
                                )

        if index is not None:
            simplex = pd.Series(simplex, index=index)

        return simplex

    def _barycentric(
            self,
            X,
            simplex_ids=None,
            transformer=None,
            app_transform=None,
            ):

        X, index = self.prep_X(
                                X,
                                transformer=transformer,
                                app_transform=app_transform,
                                enforce_feature_names_in=True,
                                )[:2]

        ndim = self.ndim

        if simplex_ids is None:
            simplex_ids = self.find_simplex(X, extrap=True, app_transform=False)

        T = self.transform[simplex_ids]

        delta_ndim = X.ndim - (T.ndim - 1)
        if delta_ndim != 0:
            edims = [1] * np.abs(delta_ndim)
            if delta_ndim < 0:
                X = X.reshape(*[-1, *edims, ndim])
            else:
                T = T.reshape(*[-1, *edims, ndim+1, ndim])

        c = np.einsum(
                "...ij,...j->...i",
                T[..., :ndim, :],
                X - T[..., ndim, :]
                )

        bary = np.concat([
                c,
                1.0 - c.sum(axis=-1, keepdims=True)
                ], axis=-1)

        return bary

    def _extrap_simplex(
            self,
            xi,
            transformer=None,
            app_transform=None,
            ):

        if self.ndim < self.ndim_embed:
            N = xi.shape[0]
            simplex = np.zeros(N, dtype=int)
        else:

            if True:
                X, index = self.prep_X(
                                        xi,
                                        transformer=transformer,
                                        app_transform=app_transform,
                                        )[:2]

                s = np.where(
                        np.any(
                            np.isin(self.simplices, np.unique(self.convex_hull)),
                            axis=1,
                            )
                        )[0]
                bary = self._barycentric(
                            X=X, simplex_ids=s.reshape(*[1]*(X.ndim - 1), -1),
                            app_transform=False,
                            )

                if False:
                    simplex = s[np.argmax(np.min(bary, axis=-1), axis=-1)]

                else:
                    sbary = np.sort(bary, axis=-1)
                    for i in range(sbary.shape[-1]):
                        sbary_max = np.nanmax(sbary[..., i], axis=-1, keepdims=True)
                        sbary[~(sbary[..., i] == sbary_max), (1+i):] = np.nan

                    simplex = s[np.argmax(~np.isnan(sbary[..., -1]), axis=-1)]

                if index is not None:
                    simplex = pd.Series(simplex, index=index)
            else:
                simplex = self.find_facet(
                            X=xi, to_bool=False, tiebreak=True, to="s",
                            transformer=transformer, app_transform=app_transform,
                            )

        return simplex

    def basis(
            self,
            X,
            beta=None,
            extrap=None,
            to_csr=None,
            y_min=None,
            y_max=None,
            na_value=None,
            transformer=None,
            app_transform=None,
            ):

        if extrap is None:
            extrap = True

        if to_csr is None:
            to_csr = False

        X, index, columns = self.prep_X(
                                X,
                                transformer=transformer,
                                app_transform=app_transform,
                                enforce_feature_names_in=True,
                                )[:3]
        columns = self._index

        N = X.shape[0]
        M = self._points.shape[0]
        Phi_shape = (N, M)

        simplices = self.find_simplex(xi=X, extrap=extrap, app_transform=False)
        inside = simplices >= 0

        if not np.any(inside):
            if beta is None:
                if to_csr:
                    Phi = csr_matrix(Phi_shape)
                    Phi.index = index
                    Phi.columns = columns
                else:
                    Phi = np.zeros(Phi_shape, dtype=float)
                    if index is not None or columns is not None:
                        Phi = pd.DataFrame(Phi, index=index, columns=columns)
                return Phi

            else:
                y = np.zeros(N, dtype=float)
                if index is not None:
                    y = pd.Series(y, index=index)
                return y

        rows = np.where(inside)[0]
        verts = self.simplices[simplices[inside]]

        bary = self._barycentric(
                X=X[inside],
                simplex_ids=simplices[inside],
                app_transform=False,
                )

        if extrap == "hull":
            outside = np.any(bary < 0, axis=1)
            bary[outside] = np.clip(bary[outside], a_min=0.0, a_max=None)
            denom = np.sum(bary[outside], axis=1).reshape(-1, 1)
            bary[outside] /= np.where(denom == 0, 1, denom)

        if beta is not None:

            beta = self.beta(
                        beta,
                        fill_val=True,
                        restore_srs=True,
                        )
            beta_1d = isinstance(beta, pd.Series)
            if beta_1d:
                beta = beta.to_frame()
            else:
                beta = beta.T

            beta_columns = beta.columns
            beta = beta.values

            if isinstance(y_min, bool):
                if y_min:
                    y_min = np.min(beta)
                else:
                    y_min = None
            if isinstance(y_max, bool):
                if y_max:
                    y_max = np.max(beta)
                else:
                    y_max = N

            if na_value is None:
                na_value = np.nan

            y = np.full(
                    (N, beta.shape[1]),
                    na_value,
                    dtype=beta.dtype,
                    )

            y[inside] = np.einsum(
                            "ij,ijk->ik",
                            bary,
                            beta[verts],
                            )
            if y_min or y_max:
                y = np.clip(y, a_min=y_min, a_max=y_max)

            if beta_1d:
                y = y[:, 0]
                if index is not None:
                    y = pd.Series(y, index=index)
            else:
                if index is not None or beta_columns is not None:
                    y = pd.DataFrame(
                            y, index=index, columns=beta_columns,
                            )

            return y

        if to_csr:
            Phi = csr_matrix(
                    (
                        bary.ravel(),
                        (
                            np.repeat(rows, self.ndim + 1),
                            verts.ravel()
                            ),
                        ),
                    shape=Phi_shape,
                    )
            Phi.index = index
            Phi.columns = columns
        else:
            N = len(verts)
            M = len(self.points)
            Phi = np.zeros((N, M), dtype=float)
            Phi[np.arange(N)[:, None], verts] = bary

            if index is not None or columns is not None:
                Phi = pd.DataFrame(Phi, index=index, columns=columns)

        return Phi


class DelaunaySet(_PointCloudSetBeta):
    pass


# =====================================
# FUNCTIONS
# =====================================

def _binary_tri_q(
        level: Optional[int] = None,
        ):

    if level is None:
        level = 0

    return np.linspace(0, 1, 1+(2**level))


def _binary_tri_V_l(
        level: Optional[int] = None,
        ndim: Optional[int] = None,
        ):

    if level is None:
        level = 0

    if ndim is None:
        ndim = 1

    if ndim == 1:
        lvl = level
        c = 0
    else:
        c = int((level - 1) % ndim)
        lvl = 1 + int((level - 1) // ndim)

    q = _binary_tri_q(level=lvl)

    if lvl == 0:
        q_split = [q for i in range(2)]
        c = 0
    else:
        q_split = [q[i:][::2] for i in range(2)]

    cs = itertools.combinations(range(ndim), c)

    return np.concat([
        [*itertools.product(*[
                    *[q_split[int(j not in i)] for j in range(ndim)],
                    ])]
        for i in cs
        ])


def _binary_tri_level_index(*args, **kwargs):
    return pd.Index(np.arange(*args, **kwargs), name="level")


def _arange_binary_tri(
        *args,
        ndim: Optional[int] = None,
        **kwargs,
        ):
    if ndim is None:
        ndim = 1

    levels = _binary_tri_level_index(*args, **kwargs)
    names = [*levels.names, "i"]
    if len(levels) == 0:
        df = pd.DataFrame(
                np.zeros((0, ndim), dtype=int),
                index=pd.MultiIndex.from_tuples([], names=names)
                )
    else:
        df = pd.concat([
                pd.DataFrame(
                    _binary_tri_V_l(level=i, ndim=ndim)
                    )
                for i in levels], axis=0, keys=levels, names=names)

    return df


def make_transformer(
        transformer=None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        ndim: Optional[int] = None,
        ):

    if transformer is None:
        if X is not None:
            transformer = MinMaxScaler()

    is_transformer = transformer is not None

    if is_transformer:
        if not _is_fitted(transformer, attributes=["n_features_in_"]):
            if X is not None:
                if isinstance(X, pd.Series):
                    X = X.to_frame()
                if not isinstance(X, pd.DataFrame):
                    X = np.atleast_1d(X)
                    if X.ndim != 2:
                        X = X.reshape(-1, 1)
                    # X = pd.DataFrame(X)
            else:
                if ndim is None:
                    ndim = 1
                X = pd.DataFrame([[0, 1]], index=np.arange(ndim)).T

            transformer.fit(X)

    return transformer


def arange_binary_tri(
        *args,
        ndim: Optional[int] = None,
        transformer=None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        ret_transformer: Optional[bool] = None,
        app_transform: Optional[bool] = None,
        ):

    transformer = make_transformer(
                    transformer=transformer,
                    X=X,
                    ndim=ndim,
                    )

    is_transformer = transformer is not None
    if ret_transformer is None:
        # ret_transformer = is_transformer
        ret_transformer = False

    if is_transformer:
        c = transformer.get_feature_names_out()
        ndim = len(c)

    U = _arange_binary_tri(*args, ndim=ndim)
    if is_transformer:
        if app_transform is None:
            app_transform = False
        if not app_transform:
            U = transformer.inverse_transform(U)
        else:
            U = U.set_axis(c, axis=1)

    if ret_transformer:
        return U, transformer
    else:
        return U


def make_child(cell, child):
    return


def canonical_diag(diag, ndim):
    """
    Identify d and complement(d).
    """
    mask = (1 << ndim) - 1
    if diag & (1 << (ndim - 1)):
        diag ^= mask
    return diag


def child_diagonal(parent_diag, child_index, ndim):

    return canonical_diag(
            parent_diag ^ child_index,
            ndim
            )


def refine(cell, diag, level, max_level, ndim):
    if level == max_level:
        yield cell, diag
        return
    for child in range(1 << ndim):
        child_cell = make_child(cell, child)
        child_diag = child_diagonal(
           diag,
           child,
           ndim,
            )
        yield from refine(
            child_cell,
            child_diag,
            level + 1,
            max_level,
            )


def reorient_kuhn(kuhn_simplices, diag):
    return kuhn_simplices ^ diag


def kuhn_simplices(ndim):
    """
    Kuhn triangulation of [0,1]^ndim.
    Returns
    -------
    simplices : (ndim!, ndim+1) ndarra
        Vertex indices into cube_vertices(ndim).
    """
    verts = list(itertools.product([0, 1], repeat=ndim))
    lookup = {v: i for i, v in enumerate(verts)}
    simplices = []
    for perm in itertools.permutations(range(ndim)):
        x = np.zeros(ndim, dtype=int)
        simplex = [lookup[tuple(x)]]
        for d in perm:
            x[d] = 1
            simplex.append(lookup[tuple(x)])
        simplices.append(simplex)
    return np.asarray(simplices, dtype=int)


def mirrored_cube_vertices(ndim):
    verts = np.array(list(itertools.product([0, 1], repeat=ndim)))
    verts[:, 0] = 1 - verts[:, 0]
    return verts


def mirror_permutation(ndim):
    verts = list(itertools.product([0, 1], repeat=ndim))
    reflected = [
       (1 - v[0], *v[1:])
       for v in verts
    ]

    lookup = {v: i for i, v in enumerate(verts)}

    return np.array(
       [lookup[v] for v in reflected],
       dtype=int,
    )


def mirrored_kuhn_simplices(ndim):
    perm = mirror_permutation(ndim)
    return perm[kuhn_simplices(ndim)]


def cell_parity(index):
    xor = 0
    for i in index:
        xor ^= int(i)
    return xor.bit_count() & 1


def cell_simplices(cell_index, ndim):
    if cell_parity(cell_index) == 0:
        return kuhn_simplices(ndim)
    return mirrored_kuhn_simplices(ndim)


def cell_vertices(index):
    ndim = len(index)
    return [
       tuple(index[d] + bit[d]
             for d in range(ndim))
       for bit in itertools.product([0, 1], repeat=ndim)
    ]
