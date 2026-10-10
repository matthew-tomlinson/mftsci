"""
Functions module for colour utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from __future__ import annotations
from typing import Optional, Union, Callable

# External packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Internal packages
from mftsci.utils.meta import return_x
from mftsci.utils.pd import cartesian_product
from mftsci.utils.colour import defaults
from mftsci.utils.colour.core import *  # noqa: F401, F403
from mftsci.utils.colour.core import is_dark_theme


# =====================================
# CLASSES
# =====================================

class Colormap(matplotlib.colors.Colormap):

    def inverted(
                self,
                name: Optional[str] = None,
                invert_bad: Optional[bool] = None,
                ) -> Colormap:

        """
        Return an inverted instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the inverted colormap. If None, the
            name is set to ``self.name + "_i"`` if ``self.name[:-2] != "_i"``
            or otherwise to ``self.name[:-2]``.


        Returns
        -------
        Colormap
            An inverted instance of the colormap.
        """

        return invert_cmap(self=self, name=name, invert_bad=invert_bad)


class ListedColormap(matplotlib.colors.ListedColormap):

    def inverted(
                    self,
                    name: Optional[str] = None,
                    invert_bad: Optional[bool] = None,
                    ) -> ListedColormap:
        """
        Return an inverted instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the inverted colormap. If None, the
            name is set to ``self.name + "_i"`` if ``self.name[:-2] != "_i"``
            or otherwise to ``self.name[:-2]``.


        Returns
        -------
        ListedColormap
            An inverted instance of the colormap.
        """

        return invert_cmap(self=self, name=name, invert_bad=invert_bad)


class ArrayNorm(matplotlib.colors.Normalize):

    def __init__(
                self,
                vmin=None,
                vmax=None,
                x=None,
                y=None,
                clip=None
                ):

        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.x = np.clip(np.atleast_1d(x), self.vmin, self.vmax)
        self.y = np.clip(np.atleast_1d(y), 0.0, 1.0)

    def __call__(
            self,
            value,
            clip=None,
            ):

        x = [self.vmin, *self.x, self.vmax]
        y = [0, *self.y, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class MidpointNorm(matplotlib.colors.Normalize):

    def __init__(
                self,
                vmin=None,
                vmax=None,
                midpoint=None,
                clip=None
                ):

        self.midpoint = midpoint
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)

    def __call__(
            self,
            value,
            clip=None,
            ):

        x = [self.vmin, self.midpoint, self.vmax]
        y = [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# =====================================
# FUNCTIONS
# =====================================


def calc_invert_suffs():

    cats = pd.Series(["colors", "bad"])
    cats.index = cats.values
    suffs_True = "_i" + cats.str.split("_").apply(lambda v: "".join([w[0] for w in v]))
    suffs_bool = suffs_True.apply(lambda v: pd.Series(["", v], index=[False, True])).T
    suffs_comb = cartesian_product(suffs_bool).sum(axis=1)
    suffs_comb.index = pd.MultiIndex.from_product(
                            [suffs_bool.index for c in cats],
                            names=cats,
                            )

    return suffs_comb


def infer_invert_suffs(
                        name: Optional[str] = None,
                        ):

    if name is None:
        name = ""
    if not isinstance(name, str):
        name = name.name

    invert_suffs = calc_invert_suffs()

    l_suffs = invert_suffs.apply(lambda v: len(v))
    if False:
        l_suffs = invert_suffs.apply(lambda v: len(v))[1:]
        find_suffs = invert_suffs.apply(lambda v: name.find(v)).iloc[1:]
        found_suffs = find_suffs > 0
        if found_suffs.any():
            min_find = find_suffs[found_suffs].min()
            where_min_find = find_suffs == min_find
            idx = l_suffs.loc[where_min_find].idxmax()
        else:
            idx = invert_suffs.index[0]

        d_idx = dict(zip(invert_suffs.index.names, idx))
    else:
        ew = invert_suffs.apply(lambda v: name.endswith(v))
        idx = l_suffs.loc[ew].idxmax()
        d_idx = pd.Series(dict(zip(invert_suffs.index.names, idx)))

    return d_idx


def invert_cmap_colors(
                self: Colormap,
                ) -> Colormap:

    self.colors = invert(self.colors)[:, :3]

    return self


def invert_cmap_bad(
                self: Colormap,
                ) -> Colormap:

    self.set_bad(invert(self.get_bad())[0, :3])

    return self


def _invert_cmap(
                self,
                invert_colors: Optional[bool] = None,
                invert_bad: Optional[bool] = None,
                ) -> Colormap:

    if defaults.invert_colors(invert_colors):
        self = invert_cmap_colors(self)
    if defaults.invert_bad(invert_bad):
        self = invert_cmap_bad(self)

    return self


def invert_cmap(
                self: Colormap,
                name: Optional[str] = None,
                invert_colors: Optional[bool] = None,
                invert_bad: Optional[bool] = None,
                ) -> Colormap:
    """
    Return an inverted instance of the Colormap.

    Parameters
    ----------
    name : str, optional
        The name for the inverted colormap. If None, the
        name is set to ``self.name + "_i"`` if ``self.name[:-2] != "_i"``
        or otherwise to ``self.name[:-2]``.

    Returns
    -------
    Colormap
        The reversed colormap.
    """

    invert_bool = pd.Series({
                                "colors":  defaults.invert_colors(invert_colors),
                                "bad": defaults.invert_bad(invert_bad),
                            })

    if name is None:
        source_invert_suffs = infer_invert_suffs(self.name)
        dest_invert_suffs = source_invert_suffs.copy()

        for inv in invert_bool.index:
            if invert_bool.loc[inv]:
                dest_invert_suffs.loc[inv] = not source_invert_suffs.loc[inv]

        invert_suffs = calc_invert_suffs()
        source_suff = invert_suffs.loc[tuple(source_invert_suffs.values)]
        dest_suff = invert_suffs.loc[tuple(dest_invert_suffs.values)]

        self.name = self.name.removesuffix(source_suff) + dest_suff

    invert_bool.index = "invert_" + invert_bool.index
    self = _invert_cmap(self, **invert_bool)

    return self


def get_cmap(
                name: Colormap | str | None = None,
                lut: int | None = None,
                invert_with_theme: Optional[int] = None,
                ) -> Colormap:
    """
    Get a colormap instance, defaulting to rc values if *name* is None.

    Parameters
    ----------
    name : `~matplotlib.colors.Colormap` or str or None, default: None
        If a `.Colormap` instance, it will be returned. Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*. The
        default, None, means :rc:`image.cmap`.
    lut : int or None, default: None
        If *name* is not already a Colormap instance and *lut* is not None, the
        colormap will be resampled to have *lut* entries in the lookup table.

    Returns
    -------
    Colormap
    """
    if name is None:
        name = matplotlib.rcParams['image.cmap']

    if not isinstance(name, str):
        name = name.name
    invert_bool = infer_invert_suffs(name)
    invert_suffs = calc_invert_suffs()
    source_suff = invert_suffs.loc[tuple(invert_bool.values)]
    name = name.removesuffix(source_suff)

    name = plt.get_cmap(name, lut)

    invert_with_theme = defaults.invert_with_theme(invert_with_theme)
    do_invert_with_theme = (
                            (invert_with_theme < 0 and not is_dark_theme()) or
                            (invert_with_theme > 0 and is_dark_theme())
                            )
    if do_invert_with_theme:
        invert_bool = ~invert_bool

    invert_bool.index = "invert_" + invert_bool.index
    name = invert_cmap(name, **invert_bool)

    return name


def calc_greyscale_val(
                        cs: Optional[Union[np.ndarray, list, str,]] = None,
                        ):

    cs = defaults.cs(cs=cs, app_to_rgba=True)

    val = 0.2989 * cs[..., 0] + 0.5870 * cs[..., 1] + 0.1140 * cs[..., 2]

    return val


def to_greyscale(
                    cs: Optional[Union[np.ndarray, list, str,]] = None,
                    ):

    cs = defaults.cs(cs=cs, app_to_rgba=True)

    val = calc_greyscale_val(cs=cs)

    for i in range(3):
        cs[..., i] = val

    return cs


def invert(
            cs: Optional[np.ndarray] = None,
            invert_alpha: Optional[bool] = None,
            ) -> np.ndarray:

    if invert_alpha is None:
        invert_alpha = False

    cs = defaults.cs(cs=cs, app_to_rgba=True)
    if invert_alpha:
        cs = 1.0 - cs
    else:
        cs[..., :3] = 1.0 - cs[..., :3]

    return cs


def c_diff(
            cs: Optional[np.ndarray] = None,
            ) -> np.ndarray:

    cs = defaults.cs(cs=cs, app_to_rgba=True)
    return np.diff(cs, axis=0)


def c_vadd_diff(
                cs: Optional[np.ndarray] = None,
                v: Optional[Union[float, int,]] = None,
                ) -> np.ndarray:

    v = defaults.v(v)

    cs = defaults.cs(cs=cs, app_to_rgba=True)
    diff = c_diff(cs=cs)

    v_diff = np.multiply.outer(v, diff)

    return v_diff + cs[:-1, ...]


def c_vadd_diff_back_fore(
                            v: Optional[Union[float, int,]] = None,
                            ) -> np.ndarray:

    return c_vadd_diff(cs=defaults.cs(app_to_rgba=True), v=v)


def c_grad_cycle(
                    cs: Optional[np.ndarray] = None,
                    N_c: Optional[int] = None,
                    N_grad: Optional[int] = None,
                    step: Optional[Union[float, int,]] = None,
                    i_0: Optional[int] = None,
                    max_contrast_0: Optional[bool] = None,
                    ) -> np.ndarray:

    max_contrast_0 = defaults.max_contrast_0(max_contrast_0)
    N_c = defaults.N_c(N_c)
    i_0 = defaults.i_0(i_0)
    cs = defaults.cs(cs=cs, app_to_rgba=True)

    if N_grad is None:
        N_grad = N_c

    if step is None:
        if calc_greyscale_val(cs[0, :3]) < 0.5:
            # step = 30/255
            step = 40/255
        else:
            step = 20/255

    c_grad = c_vadd_diff(v=(i_0+np.arange(N_grad))*step)[:, 0, ...]
    if max_contrast_0:
        c_grad = np.flip(c_grad, axis=0)

    N_cycle = 1+((N_c-1)//N_grad)

    c_grad_cyc = np.concatenate([c_grad for k in range(N_cycle)])[:N_c]

    if False:
        _c_grad_cyc = np.concatenate([c_grad for k in range(N_cycle)])
        if max_contrast_0:
            c_grad_cyc = np.flip(_c_grad_cyc[-N_c:], axis=0)
        else:
            c_grad_cyc = _c_grad_cyc[:N_c]

    return c_grad_cyc


def reverse_cmap(
                    reverse: Optional[bool] = None,
                    flip: Optional[bool] = None,
                    ) -> bool:

    reverse = defaults.reverse(reverse)
    flip = defaults._set_default(val=flip, default_call=is_dark_theme)

    if flip:
        reverse = not True

    return reverse


def reverse_cmap_suff(
                        reverse: Optional[bool] = None,
                        flip: Optional[bool] = None,
                        ) -> str:

    reverse = reverse_cmap(reverse=reverse, flip=flip)

    if reverse:
        suff = "_r"
    else:
        suff = ""

    return suff


def make_cmap_excepts(
                        excepts: Optional[Union[dict, list]] = None,
                        cmap_base: Optional[Callable] = None,
                        ) -> Callable:

    if excepts is None:
        excepts = {}
    elif isinstance(excepts, list):
        excepts = {e: None for e in excepts}

    for e in excepts:
        if excepts[e] is None:
            excepts[e] = c_vadd_diff(v=0.5)

    if cmap_base is None:
        cmap_base = get_cmap()
    elif isinstance(cmap_base, str):
        cmap_base = get_cmap(cmap_base)

    def cmap(val):
        c = cmap_base(val)
        for e in excepts:
            if np.isscalar(val):
                if val == e:
                    c = excepts[e]
            else:
                c[val == e] = excepts[e]

        return c

    return cmap


def _srs_colour(
                s: pd.Series,
                mode=None,
                vmin=None,
                vmax=None,
                cmap=None,
                norm=None,
                invert_colour=None,
                rank_cols=None,
                u_cols=None,
                app_to_hex=None,
                ) -> pd.Series:

    if mode is None:
        mode = ""

    ineq_modes = {
                    "less": ["<", "less"],
                    "less_equal": ["<=", "less_equal", "less_eq", "leq"],
                    "greater": [">", "greater"],
                    "greater_equal": [">=", "greater_equal", "greater_eq", "geq"],
                }
    # ineq_modes_list_g=[m for d in ["greater", "greater_equal"] for m in ineq_modes[d]]
    # ineq_modes_list_l = [m for d in ["less", "less_equal"] for m in ineq_modes[d]]
    # ineq_modes_list = [*ineq_modes_list_g, *ineq_modes_list_l]

    for d in ineq_modes:
        if np.isin(mode, ineq_modes[d]):
            mode = d
            break

    invert_colour = defaults.invert_colour(invert_colour)

    s_float = s.astype(float)
    N = s_float.size

    if cmap is None:
        if mode[:7] == "greater":
            cmap = defaults._set_default(cmap, default="winter")

        elif mode[:4] == "less":
            cmap = defaults._set_default(cmap, default="autumn")
        else:
            cmap = defaults.cmap(cmap)
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    if norm is None:
        norm = return_x

    def norm_cmap(v):
        return cmap(norm(v))

    if np.isin(mode, [*ineq_modes.keys()]):

        if u_cols is None:
            u_cols = 0.0
        if np.isscalar(u_cols):
            u_cols = [u_cols]
        if not isinstance(u_cols, pd.Series):
            _u_cols = norm_cmap(np.linspace(0.0, 1.0, len(u_cols)))
            u_cols = pd.Series([c for c in _u_cols], index=u_cols)

        if mode[:4] == "less":
            sort_asc = False
        else:
            sort_asc = True

        u_cols = u_cols.sort_index(ascending=sort_asc)

        N_exceed = pd.DataFrame(
                    getattr(np, mode).outer(s.values, u_cols.index),
                    index=s.index,
                    columns=u_cols.index).astype(int).sum(axis=1) - 1

        active = N_exceed >= 0

        cols = u_cols.copy().iloc[N_exceed[active].values]
        cols.index = N_exceed[active].index

    elif np.isin(mode, ["rank", "min", "max"]):

        argsort_ = s_float.argsort()
        if mode != "min":
            argsort_.loc[:] = argsort_[::-1].values
        rank = argsort_.argsort()
        rank.name = "rank"

        if rank_cols is None:
            val = np.linspace(1, 0, N).astype(float)
            rank_cols = [c for c in norm_cmap(val)]
        if np.isscalar(rank_cols):
            rank_cols = [rank_cols]
        if not isinstance(rank_cols, pd.Series):
            rank_cols = pd.Series(rank_cols)
        if rank_cols.dtype == int or rank_cols.dtype == float:
            rank_cols = rank_cols.apply(lambda v: norm_cmap(v))
        else:
            rank_cols = rank_cols.apply(
                            lambda v: matplotlib.colors.to_rgba_array(v)[0, ...]
                            )

        active = pd.Series(np.isin(rank.values, rank_cols.index), index=rank.index)
        cols = rank_cols.copy().loc[rank[active].values]
        cols.index = rank[active].index
    else:
        active = pd.Series(True, index=s_float.index)

        if vmin is None:
            vmin = s_float.min()
        if vmax is None:
            vmax = s_float.max()
        val = ((s_float - vmin) / (vmax - vmin))
        cols = val.apply(lambda v: norm_cmap(v))

    if invert_colour:
        cols = cols.apply(lambda v: invert(v))

    if app_to_hex:
        cols = cols.apply(lambda v: matplotlib.colors.to_hex(v))
    else:
        cols = cols.apply(lambda v: matplotlib.colors.to_rgba_array(v)[0, ...])

    return cols, active


def pd_colour(
                s: pd.Series,
                mode=None,
                vmin=None,
                vmax=None,
                cmap=None,
                norm=None,
                invert_colour=None,
                u_cols=None,
                rank_cols=None,
                app_to_hex=None,
                default=None,
                default_v=None,
                ):

    app_to_hex = defaults.app_to_hex(app_to_hex)

    if default is None:
        default_v = defaults.v(default_v)
        default = c_vadd_diff(v=default_v)
    default = defaults.cs(cs=default, app_to_rgba=True)[0, ...]

    cols = s.apply(lambda v: default)

    _cols, _active = _srs_colour(
                                s=s,
                                mode=mode,
                                vmin=vmin,
                                vmax=vmax,
                                cmap=cmap,
                                norm=norm,
                                invert_colour=invert_colour,
                                u_cols=u_cols,
                                rank_cols=rank_cols,
                                )
    cols[_active] = _cols

    if app_to_hex:
        cols = cols.apply(lambda v: matplotlib.colors.to_hex(v))

    return cols


def pd_scolour(
                s: pd.Series,
                mode=None,
                vmin=None,
                vmax=None,
                cmap=None,
                norm=None,
                invert_colour=None,
                u_cols=None,
                rank_cols=None,
                cmode=None
                ):

    cols, active = _srs_colour(
                                s=s,
                                mode=mode,
                                vmin=vmin,
                                vmax=vmax,
                                cmap=cmap,
                                norm=norm,
                                invert_colour=invert_colour,
                                u_cols=u_cols,
                                rank_cols=rank_cols,
                                app_to_hex=True,
                                )

    cmode = defaults.cmode(cmode)

    s_hl = s.apply(lambda v: "")
    if cmode[:10].lower() == "background":
        tcols = cols.apply(lambda v: "white")
        tcols[calc_greyscale_val(cols) > 0.5] = "black"

        s_hl[active] = "background-color: " + cols.values + "; color: " + tcols.values + ";"    # noqa: E501
    else:
        s_hl[active] = "color: " + cols.values + ";"

    return s_hl
