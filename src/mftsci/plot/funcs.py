"""
Functions module for plot.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import (
                        Optional, Union, Literal, Callable, List, Tuple, Any,
                    )
from collections.abc import (
        Mapping,
    )

# External packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Internal packages
from mftsci import utils
from . import defaults  # noqa: F401
from mftsci.utils.colour import get_cmap
from mftsci.utils.sklearn.preprocessing import Reshepe2Dto1D, Reshepe1Dto2D
from mftsci.utils.sklearn.neighbors import make_kde_pipeline


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def calc_density(
        data: Union[np.ndarray, pd.DataFrame, pd.Series,],
        kde_kwargs: Optional[dict] = None,
        scaler: Optional[Union[Callable, str]] = None,
        ret_pipeline: Optional[bool] = None,
        ):

    ret_pipeline = defaults.set_default(ret_pipeline, True)

    kde_pipeline = make_kde_pipeline(kde_kwargs=kde_kwargs, scaler=scaler, data=data)
    density = np.exp(kde_pipeline.score_samples(data))

    if ret_pipeline:
        return density, kde_pipeline
    else:
        return density,


def dplot(
        data: Union[np.ndarray, pd.DataFrame, pd.Series,],
        xs: Optional[Union[np.ndarray, pd.Series]] = None,
        num_xs: Optional[int] = None,
        bound_xs: Optional[np.array] = None,
        plot_mode: Optional[Literal["plot", "fill_between"]] = None,
        plot_kwargs: Optional[dict] = None,
        kde_kwargs: Optional[dict] = None,
        scaler: Optional[Union[Callable, str]] = None,
        dcoef: Optional[float] = None,
        ret_pipeline: Optional[bool] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
        ):

    ret_pipeline = defaults.set_default(ret_pipeline, True)
    dcoef = defaults.set_default(dcoef, 1.0)
    plot_mode = defaults.set_default(plot_mode, "fill_between")

    if plot_mode[:12] == "fill_between":
        dkwargs = {"lw": 0.0}
    else:
        dkwargs = {}

    plot_kwargs = utils.meta.set_default_kwargs(
                kwargs=plot_kwargs,
                dkwargs=dkwargs,
                )

    data = Reshepe2Dto1D().fit_transform(data)
    if xs is None:
        num_xs = defaults.set_default(num_xs, 401)
        if bound_xs is None:
            bound_xs = [np.min(data), np.max(data)]
        xs = np.linspace(*bound_xs, num_xs)

    data_pd = hasattr(data, "values")
    xs_pd = hasattr(xs, "values")
    if data_pd and not xs_pd:
        xs = pd.Series(xs, name=data.name)
    elif not data_pd and xs_pd:
        xs = xs.values

    kde_pipeline = make_kde_pipeline(
                    kde_kwargs=kde_kwargs,
                    scaler=scaler,
                    ).fit(data)

    density = np.exp(kde_pipeline.score_samples(xs))

    if ax is None:
        fig, ax = plt.subplots()

    if plot_mode[:12] == "fill_between":
        xy = ["x", "y"]
        if plot_mode[-1:] == "x":
            xy = xy[::-1]
        ax.fill_between(
                **{xy[0]: xs, f"{xy[1]}1": dcoef*density},
                **plot_kwargs,
                )
    else:
        ax.plot(xs, dcoef*density, **plot_kwargs)

    if ret_pipeline:
        return ax, kde_pipeline
    else:
        return ax,


def dscatter(
        data: Union[np.ndarray, pd.DataFrame, pd.Series,],
        scatter_kwargs: Optional[dict] = None,
        kde_kwargs: Optional[dict] = None,
        scaler: Optional[Union[Callable, str]] = None,
        ret_pipeline: Optional[bool] = None,
        app_density: Optional[bool] = None,
        app_czorder: Optional[Literal[-1, 0, 1]] = None,
        app_ax_transScale: Optional[bool] = None,
        fill_y: Optional[float] = None,
        flip_1D: Optional[bool] = None,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
        ):
    # axis transformations

    scatter_kwargs = utils.meta.set_default_kwargs(
                kwargs=scatter_kwargs,
                dkwargs=dict(marker=".", lw=0),
                )
    kde_pipeline = None

    has_c = scatter_kwargs.get("c", None) is not None

    app_density = defaults.set_default(app_density, not has_c)
    app_czorder = defaults.set_default(app_czorder, 1)
    app_ax_transScale = defaults.set_default(app_ax_transScale, True)
    ret_pipeline = defaults.set_default(ret_pipeline, True)
    flip_1D = defaults.set_default(flip_1D, False)

    if ax is None:
        fig, ax = plt.subplots()

    if app_ax_transScale:
        if len(np.shape(data)) == 1:
            if flip_1D:
                f_trans = ax.get_yaxis_transform().inverted().transform
            else:
                f_trans = ax.get_xaxis_transform().inverted().transform
        else:
            f_trans = ax.transScale.transform
    else:
        f_trans = utils.meta.return_x

    if app_density:
        scatter_kwargs["c"], kde_pipeline = calc_density(
                    data=f_trans(data),
                    kde_kwargs=kde_kwargs,
                    scaler=scaler,
                    ret_pipeline=True
                    )
        has_c = True
    else:
        ret_pipeline = False

    if hasattr(data, "values"):
        data = data.values

    if has_c:
        if app_czorder:
            order = np.argsort(scatter_kwargs["c"])
            if app_czorder == -1:
                order = order[::-1]
            data = data[order, ...]
            scatter_kwargs["c"] = scatter_kwargs["c"][order]
    else:
        scatter_kwargs.pop("cmap", None)

    data = Reshepe1Dto2D().fit_transform(data)
    xydata = [*data.T[:2, :]]

    if len(xydata) == 1:
        fill_y = defaults.set_default(fill_y, 1.0)
        xydata.append(np.full(data.shape[0], fill_y))

        if flip_1D:
            xydata = xydata[::-1]

    ax.scatter(*xydata, **scatter_kwargs)

    if ret_pipeline:
        return ax, kde_pipeline
    else:
        return ax,


def pairs(
        data: pd.DataFrame,
        rows: Optional[Union[pd.Index, np.ndarray, list,]] = None,
        columns: Optional[Union[pd.Index, np.ndarray, list,]] = None,
        c_density: Optional[Union[pd.Series, np.ndarray, list, bool,]] = None,
        c: Optional[Union[pd.Series, np.ndarray, list, str,]] = None,
        app_czorder: Optional[Literal[-1, 0, 1]] = None,
        cat: Optional[Union[pd.Series, np.ndarray, list, str,]] = None,
        metacats: Optional[Union[pd.DataFrame, list, np.ndarray, object]] = None,
        metacats_scatter: Optional[Union[pd.Index, list, np.ndarray, object]] = None,
        metacats_kde: Optional[Union[pd.Index, list, np.ndarray, object]] = None,
        metacats_corr: Optional[Union[pd.Index, list, np.ndarray, object]] = None,
        cmap_metacats: Optional[Union[pd.Series, Callable]] = None,
        cmap_metacats_all: Optional[Callable] = None,
        kde_dcoef_metacats: Optional[Union[pd.Series, float,]] = None,
        kde_match_ybounds: Optional[bool] = None,
        v_metacats: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
        v_metacats_all: Optional[float] = None,
        change_c_all: Optional[bool] = None,
        cmap_corr: Optional[Callable] = None,
        pmode: Optional[np.ndarray] = None,
        smode: Optional[int] = None,
        vsmode: Optional[int] = None,
        bounds: Optional[pd.DataFrame] = None,
        plot_child: Optional[bool] = None,
        scatter_kwargs: Optional[dict] = None,
        kde_2D_kwargs: Optional[dict] = None,
        kde_kwargs: Optional[dict] = None,
        fill_between_kwargs: Optional[dict] = None,
        num_xs: Optional[int] = None,
        show_ticks: Optional[Literal["", "all", "outer", "outer_x"]] = None,
        show_ticklabels: Optional[Literal["", "all", "outer", "outer_x"]] = None,
        overwrite_ticks_over_pmode: Optional[bool] = None,
        ax_parent: Optional[np.ndarray] = None,
        subplots_kwargs: Optional[dict] = None,
        ):
    """
    Returns a pairs plot.

    Parameters
    ----------
    data : pd.DataFrame
        The data.

    rows : Optional[Union[np.ndarray, list,]]
        Features to be ploted on the y axes.

    columns : Optional[Union[np.ndarray, list,]]
        Features to be ploted on the c axes.

    c_density: Optional[Union[pd.Series, np.ndarray, list, bool,]]
        Use 2D kde density to set colourmap value in scatter plots.

    c: Optional[Union[pd.Series, np.ndarray, list, str,]]
        Series used for colourmap in scatter plots.

    app_czorder: Literal[-1, 0, 1]
        Controls whether to sort data according to c to affect scatter z order

    cat: Optional[Union[pd.Series, np.ndarray, list, str,]]
        Series used for catagorical partition of data.

    metacats: Optional[Union[pd.DataFrame, list, np.ndarray, object]]
        Defines metacatagories from unions of catagorical partitions. Defaults to
        identiy matrix, which preserves original partitions.

    metacats_scatter: Optional[Union[pd.Index, list, np.ndarray, object]]
        Selects which metacatagories are included in scatter plots.

    metacats_kde: Optional[Union[pd.Index, list, np.ndarray, object]]
        Selects which metacatagories are included in kde plots.

    metacats_corr: Optional[Union[pd.Index, list, np.ndarray, object]]
        Selects which metacatagories are included in correlation plots.

    cmap_metacats: Optional[Union[pd.Series, Callable]]
        Colourmaps used for each metacatogry.

    kde_dcoef_metacats: Optional[Union[pd.Series, float]]
        Relative height of kde density plots. Set to sample size by default.

    kde_match_ybounds: Optional[Union[bool]]
        Transform kde to ybounds of other plots.

    cmap_metacats_all: Optional[Callable]
        Colourmaps used for union of all catagories if change_c_all is True.

    v_metacats: Optional[float]
        Colourmap value used to to extract identifying colour. Default is 1.0.

    v_metacats_all: Optional[float]
        Colourmap value Colourmaps used for union of all catagories if change_c_all
        is True.

    change_c_all: Optional[bool]
        Controls whether to overwrite colourmap for union of all catagories

    cmap_corr: Optional[Callable]
        Colourmap for facecolor of correlation plots

    pmmode: Optional[Literal[0, 1]]
        Boolean array for whether to plot on axis or not.

    smode: Optional[Literal[0, 1]]
        Sets type of plot when row == column.
        0: name
        1: kde
        If None then is set to 1

    vsmode: Optional[Literal[0, 1]]
        Sets type of plot when row != column.
        0: scatter
        1: correlation
        If None and rows == columns, scatter plots will be placed in the upper
        diagonal and correlation plots in the lower diagonal

    bounds: Optional[pd.DataFrame]
        Used to set axis limits in scatter and kde plots.

    plot_child: Optional[bool]
        Option to return each scatter plot as a bespoke figure.

    scatter_kwargs: Optional[dict]
        Additional keyword arguments for scatter plot. By default
        marker = "."
        s = 1

    kde_2D_kwargs: Optional[dict]
        Additional keyword arguments for 2D (scatter) kde. By default
        kernel = "epanechnikov"
        bandwidth = "silverman"

    kde_kwargs: Optional[dict]
        Additional keyword arguments for kde. By default
        kernel = "epanechnikov"
        bandwidth = "silverman"

    fill_between_kwargs: Optional[dict]
        Additional keyword arguments for fill_between on kde plot. By defult
        alpha = 0.5
        lw = 0.0

    num_xs: Optional[int]
        Number of linspace points for fill_between on kde plot.

    show_ticks: Optional[Literal["", "all", "outer", "outer_x"]]
        Controls whether to remove all ticks from all panels
        in ax_parent

    show_ticklabels: Optional[Literal["", "all", "outer", "outer_x"]]
        Controls whether to remove all ticklabels from all panels
        in ax_parent

    overwrite_ticks_over_pmode: Optional[bool]
        Clear ticks and ticklabels on non-plotted axes
        in ax_parent

    ax_parent: Optional[np.ndarray]
        A pre-instantiated array of axes on which to add plots.

    Returns
    -------
    fig_parent
        matplotlib figure
    ax_parent
        Array of matplotlib axes

    """

    # Series of catagories
    if cat is None:
        cat = pd.Series(True, index=data.index)
    elif isinstance(cat, str):
        cat = data[cat]
    elif not isinstance(cat, pd.Series):
        cat = pd.Series(cat, index=data.index)

    # If catagories boolean, need to change to integer for later pandas compatibility
    cat_bool = False
    if cat.dtype == bool:
        cat_bool = True     # noqa: F841
        cat = cat.astype(int)

    # Default colourmap
    if cmap_metacats is None:
        cmap_metacats = get_cmap("viridis" + utils.colour.reverse_cmap_suff(
                                                    reverse=True
                                                    ))

    # If scalar, same colourmap used for all metacatagories
    cmap_metacats_scalar = not (
                                isinstance(cmap_metacats, pd.Series) or
                                isinstance(cmap_metacats, list) or
                                isinstance(cmap_metacats, np.ndarray)
                                )

    app_czorder = defaults.set_default(app_czorder, 1)
    app_c = c is not None
    if c_density is None:
        if app_c or cmap_metacats_scalar:
            c_density = False
        else:
            c_density = True

    if np.all(c_density):
        app_c = False

    _app_czorder = app_czorder
    if app_c:
        if isinstance(c, str):
            c = data[c]
        elif not isinstance(c, pd.Series):
            c = pd.Series(c, index=data.index)

        if app_czorder:
            order = np.argsort(c)
            if app_czorder == -1:
                order = order[::-1]
            data = data.iloc[order]
            c = c.iloc[order]
            _app_czorder = 0

    # DataFrame of dummies for cat Series
    cat_dummies = pd.get_dummies(cat)

    # Set rows and columns. If only one is specified, then plot is square.
    square = False
    if rows is None:
        if columns is not None:
            rows = columns
            square = True
        else:
            rows = data.columns
    rows = utils.arr.iterarr(rows)

    if columns is None:
        columns = rows
        square = True
    columns = utils.arr.iterarr(columns)

    # Axes shape
    N_rows = len(rows)
    N_cols = len(columns)

    # Find boundaries from data if not given
    if bounds is None:
        bounds = data[[*set([*rows, *columns])]].agg(["min", "max"])
        bounds.index = [0, 1]

    # Set mode for plot
    if pmode is None:
        pmode = np.full((N_rows, N_cols), True)

    # Set mode for row == column plots
    if smode is None:
        smode = np.ones((N_rows, N_cols), dtype=int)
    if np.isscalar(smode):
        smode = np.full((N_rows, N_cols), smode).astype(int)

    # Set mode for row != column plots
    if vsmode is None:
        vsmode = np.zeros((N_rows, N_cols), dtype=int)
        if square:
            vsmode[np.tril_indices_from(vsmode, k=-1)] = 1
    if np.isscalar(vsmode):
        vsmode = np.full((N_rows, N_cols), vsmode).astype(int)

    # Return bespoke scatter plots
    if plot_child is None:
        plot_child = False

    scatter_kwargs = utils.meta.set_default_kwargs(
                    kwargs=scatter_kwargs,
                    dkwargs={"marker": ".", "s": 1},
                    )

    kde_2D_kwargs = utils.meta.set_default_kwargs(
                kwargs=kde_2D_kwargs,
                dkwargs={"kernel": "epanechnikov", "bandwidth": "silverman"},
                )

    kde_kwargs = utils.meta.set_default_kwargs(
                kwargs=kde_kwargs,
                dkwargs={"kernel": "epanechnikov", "bandwidth": "silverman"},
                )

    fill_between_kwargs = utils.meta.set_default_kwargs(
            kwargs=fill_between_kwargs,
            dkwargs={"alpha": 0.5, "lw": 0.0},
            )

    # Metacatagories create unions from the cat partitions
    metacats = utils.pd.calc_metacats(
                            metacats=metacats,
                            cats=cat_dummies.columns,
                            )

    # Is metacatagory a union of all catagories
    metacats_all = metacats.all(axis=1)

    # Metacatagoreies shown on each scatter plot
    if metacats_scatter is None:
        metacats_scatter = metacats.index
    metacats_scatter = utils.arr.iterarr(metacats_scatter)

    # Metacatagoreies shown on each kde plot
    if metacats_kde is None:
        metacats_kde = metacats.index
    metacats_kde = utils.arr.iterarr(metacats_kde)

    # Metacatagoreies shown on each correlation plot
    if metacats_corr is None:
        metacats_corr = metacats.index
    metacats_corr = utils.arr.iterarr(metacats_corr)

    if not isinstance(c_density, pd.Series):
        c_density = pd.Series(c_density, index=metacats.index)

    # Overwrite colourmap for union of all catagories
    if change_c_all is None:
        change_c_all = True

    # c_neutral used in kde plots when only when metacatagory shown
    c_neutral = utils.colour.c_vadd_diff(v=0.5)[0]

    if not isinstance(cmap_metacats, pd.Series):
        cmap_metacats = pd.Series(cmap_metacats, index=metacats.index)

    if cmap_metacats_all is not None and change_c_all:
        cmap_metacats[metacats_all] = cmap_metacats_all

    if v_metacats is None:
        v_metacats = pd.Series(1.0, index=metacats.index)
        if cmap_metacats_scalar:
            if change_c_all:
                mc_grad = ~metacats_all
            else:
                mc_grad = pd.Series(True, index=metacats.index)
            N_grad = mc_grad.sum()
            if N_grad == 1:
                l_0 = 1.0
            else:
                l_0 = 0.0

            if N_grad > 0:
                v_metacats[mc_grad] = np.linspace(l_0, 1.0, N_grad)

    if not isinstance(v_metacats, pd.Series):
        v_metacats = pd.Series(v_metacats, index=metacats.index)

    if v_metacats_all is not None and change_c_all:
        v_metacats[metacats_all] = v_metacats_all

    c_metacats = pd.concat([cmap_metacats, v_metacats],
                           axis=1).apply(lambda v: v[0](v[1]), axis=1)

    if len(metacats_scatter) > 1 or len(metacats_kde) > 1:
        c_kde = c_metacats
    else:
        c_kde = c_metacats.apply(lambda v: c_neutral)

    N_metacats = metacats.apply(lambda v:
                                cat_dummies.iloc[:, v.values].any(axis=1).sum(),
                                axis=1)

    if kde_dcoef_metacats is None:
        kde_dcoef_metacats = N_metacats
    elif not isinstance(kde_dcoef_metacats, pd.Series):
        kde_dcoef_metacats = pd.Series(kde_dcoef_metacats, index=metacats.index)

    # Default colourmap for correlation plot facecolor
    if cmap_corr is None:
        if utils.colour.is_dark_theme():
            cmap_corr = get_cmap("berlin_r")
        else:
            cmap_corr = get_cmap("seismic_r")

    if plot_child:
        fig_children = [[None for j in range(N_cols)] for i in range(N_rows)]
        ax_children = [[None for j in range(N_cols)] for i in range(N_rows)]

    if ax_parent is None:
        _dkwargs = dict(sharex="col")
        if square:
            _dkwargs["figsize"] = (7, 7)

        subplots_kwargs = defaults.set_default_kwargs(
                                    subplots_kwargs,
                                    dkwargs=_dkwargs,
                                    )

        fig_parent, ax_parent = utils.plot.subplots(
                                            nrows=N_rows,
                                            ncols=N_cols,
                                            **subplots_kwargs,
                                            )

    else:
        fig_parent = ax_parent[0, 0].get_figure()

    if False:
        if show_ticks is None:
            show_ticks = pmode.copy()
        if np.isscalar(show_ticks):
            show_ticks = np.full((N_rows, N_cols), show_ticks).astype(bool)

    show_ticks = defaults.set_default(show_ticks, "")
    show_ticklabels = defaults.set_default(show_ticklabels, "")
    _show = {"": show_ticks, "label": show_ticklabels}
    for lab in _show:
        if not isinstance(_show[lab], str):
            if _show[lab]:
                _show[lab] = "all"
            else:
                _show[lab] = ""
        _show[lab] = str(_show[lab]).lower()

    overwrite_ticks_over_pmode = defaults.set_default(overwrite_ticks_over_pmode, True)
    kde_match_ybounds = defaults.set_default(kde_match_ybounds, False)

    _clear_x = []
    for lab in _show:
        if _show[lab] == "outer_x":
            _clear_x.append(lab)

    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            if pmode[i, j] or overwrite_ticks_over_pmode:
                _outer = {
                            "top": i == 0,
                            "bottom": i == N_rows - 1,
                            "left": j == 0,
                            "right": j == N_cols - 1,
                            }

                ax_parent[i, j].tick_params(**{
                            lab+k: (
                                        True if _show[lab] == "all"
                                        else _outer[k] if _show[lab][:5] == "outer"
                                        else False
                                        )
                            for lab in _show for k in _outer
                            })
            if pmode[i, j]:
                if row == col:
                    if smode[i, j] == 0:
                        # name
                        ax_parent[i, j].set_ylim(bounds[row].values)
                        ax_parent[i, j].set_xlim(bounds[col].values)
                        ax_parent[i, j].annotate(
                            row,
                            xycoords="axes fraction", xy=[0.5, 0.5],
                            horizontalalignment="center",
                            verticalalignment="center",
                            )

                    else:
                        # kde plot

                        for idx in metacats_kde:
                            active = cat_dummies.iloc[
                                        :, metacats.loc[idx].values
                                        ].any(axis=1)

                            dplot(
                                data=data[col][active],
                                num_xs=num_xs,
                                bound_xs=bounds[col],
                                dcoef=kde_dcoef_metacats.loc[idx],
                                plot_kwargs=dict(
                                                **fill_between_kwargs,
                                                color=c_kde.loc[idx],
                                                ),
                                ax=ax_parent[i, j],
                                )

                        ax_parent[i, j].set_xlim(bounds[col].values)
                        ax_parent[i, i].set_ylim(bottom=0)

                        if kde_match_ybounds:
                            def tfunc(v):
                                loc = bounds[col].values[0]
                                scale = (
                                            np.diff(bounds[col].values)[0]
                                            / np.diff(ax_parent[i, i].get_ylim())[0]
                                            )
                                return loc + (scale * v)

                            for k, idx in enumerate(metacats_kde):
                                utils.plot.transform_obj_ax(
                                                obj=ax_parent[i, i].get_children()[k],
                                                axis=1,
                                                tfunc=tfunc,
                                                )
                            ax_parent[i, j].set_ylim(bounds[col].values)

                        ax_parent[i, j].tick_params(**{
                                                lab+d: False
                                                for lab in _clear_x
                                                for d in ["left", "right"]
                                                })

                else:
                    if vsmode[i, j] == 1:
                        # correlation plot

                        corr = pd.Series(np.nan, index=metacats_corr)
                        for idx in metacats_corr:
                            active = cat_dummies.iloc[
                                                :,
                                                metacats.loc[idx].values
                                                ].any(axis=1)
                            corr.loc[idx] = np.corrcoef(
                                            data[row][active],
                                            data[col][active],
                                            )[0, 1]

                        corr = corr.iloc[::-1]

                        corr_str = utils.scif.calc_scif(
                                                corr, pow10=0, dp=2,
                                                val_plus=True,
                                                pltf=True,
                                                app_dlrs=True,
                                                )
                        corr_str.iloc[1:] = corr_str.iloc[1:].apply(
                                            lambda v: utils.ps.capp_dlrs(
                                                    utils.ps.capp_brac(
                                                        v[1:-1]
                                                        )))

                        corr_str_merge = "\n".join(corr_str.values)

                        fcols = cmap_corr((1.0 + corr)/2.0)

                        tcols = np.repeat("white", len(fcols))
                        tcols[utils.colour.calc_greyscale_val(fcols) > 0.5] = "black"

                        ax_parent[i, j].set_facecolor(fcols[0])

                        ax_parent[i, j].set_ylim(bounds[row].values)
                        ax_parent[i, j].set_xlim(bounds[col].values)
                        ax_parent[i, j].annotate(
                                                corr_str_merge,
                                                xycoords="axes fraction", xy=[0.5, 0.5],
                                                horizontalalignment="center",
                                                verticalalignment="center",
                                                color=tcols[0],
                                                # color=annotate_cmap((1.0 + corr)/2.0),
                                                )
                    else:
                        # scatter plot
                        axs = [ax_parent[i, j]]
                        if plot_child:
                            fig_children[i][j], ax_children[i][j] = plt.subplots()
                            axs.append(ax_children[i][j])

                        for ax in axs:
                            for idx in metacats_scatter:
                                active = cat_dummies.iloc[:, metacats.loc[idx].values]\
                                            .any(axis=1)
                                data_active = data[[col, row]][active]

                                use_cmap = app_c
                                __app_czorder = app_czorder
                                if c_density.loc[idx] or use_cmap:
                                    scatter_c_kwargs = {
                                                        "cmap": cmap_metacats.loc[idx],
                                                        }
                                    if app_c:
                                        scatter_c_kwargs["c"] = c[active]
                                        __app_czorder = _app_czorder
                                else:
                                    scatter_c_kwargs = {
                                                        "color": c_metacats.loc[idx],
                                                        }

                                dscatter(
                                    data=data_active,
                                    app_density=c_density.loc[idx],
                                    app_czorder=__app_czorder,
                                    scatter_kwargs=dict(
                                                    **scatter_kwargs,
                                                    **scatter_c_kwargs,
                                                    ),
                                    ax=ax,
                                    )

                            ax.set_ylim(bounds[row].values)
                            ax.set_xlim(bounds[col].values)

                            if plot_child:
                                ax_children[i][j].set_ylabel(row)
                                ax_children[i][j].set_xlabel(col)
                                fig_children[i][j].set_size_inches(3.4, 3.4)
                                fig_children[i][j].tight_layout()

    for i, row in enumerate(rows):
        if pmode[i, -1]:
            ax_parent[i, -1].yaxis.set_label_position("right")
        for k in [0, -1]:
            if pmode[i, k]:
                ax_parent[i, k].set_ylabel(row)

    for j, col in enumerate(columns):
        if pmode[0, j]:
            ax_parent[0, j].xaxis.set_label_position("top")
        for k in [0, -1]:
            if pmode[k, j]:
                ax_parent[k, j].set_xlabel(col)

    if plot_child:
        fig_parent, ax_parent, fig_children, ax_children
    else:
        return fig_parent, ax_parent


def plot_X_y_samp(
                X_samp: Union[pd.DataFrame, pd.Series],
                y_samp: Union[pd.DataFrame, pd.Series],
                plot_kwargs: Optional[dict] = None,
                subplots_kwargs: Optional[dict] = None,
                ):

    if isinstance(X_samp, pd.Series):
        X_samp = X_samp.to_frame()
    if isinstance(y_samp, pd.Series):
        y_samp = y_samp.to_frame()

    plot_kwargs = utils.meta.set_default_kwargs(plot_kwargs)
    subplots_kwargs = utils.meta.set_default_kwargs(
                            subplots_kwargs,
                            dkwargs=dict(
                                    ncols=X_samp.shape[1],
                                    nrows=y_samp.shape[1],
                                    sharey=True,
                                    sharex="col",
                                    ))

    fig, ax = utils.plot.subplots(
                **subplots_kwargs,
                )

    ax2 = utils.arr.fmap(ax, lambda v: v.twinx())
    utils.arr.fmap(
            ax2,
            lambda v:
            v.tick_params(
                            **{
                                lab+d: False
                                for lab in ["", "label"]
                                for d in utils.plot.calc_edges().reshape(-1)
                                }
                            ))
    utils.arr.fmap(
            ax,
            lambda v:
            v.tick_params(
                            **{
                                lab+d: True
                                for lab in [""]
                                for d in utils.plot.calc_edges().reshape(-1)
                                }
                            ))

    if X_samp.index.nlevels == 3:

        plot_kwargs.setdefault("color", utils.colour.c_vadd_diff(v=1.0))
        plot_kwargs.setdefault("lw", 0.05)
        plot_kwargs.setdefault("alpha", 0.1)

        for i, m in enumerate(y_samp.columns):
            for j, c in enumerate(X_samp.columns):
                if i < ax.shape[0] and j < ax.shape[1]:
                    svals = X_samp.loc[c].index.get_level_values(0).unique()
                    for s in svals:
                        ax[i, j].plot(
                                    X_samp.loc[(c, s), c], y_samp.loc[(c, s), m],
                                    **plot_kwargs,
                                    )
    else:
        for i, m in enumerate(y_samp.columns):
            for j, c in enumerate(X_samp.columns):
                if i < ax.shape[0] and j < ax.shape[1]:
                    dscatter(
                        data=pd.concat([X_samp[c], y_samp[m]], axis=1),
                        ax=ax[i, j],
                        **plot_kwargs,
                        )

    for j, c in enumerate(X_samp.columns):
        if j < ax.shape[1]:
            ax[-1, j].set_xlabel(c)

    for i, m in enumerate(y_samp.columns):
        if i < ax.shape[0]:
            ax2[i, -1].set_ylabel(m)

    return fig, ax, ax2


def parallel_coordinates(
                frame: pd.DataFrame,
                class_column: str,
                cols: Optional[List[str]] = None,
                ax: matplotlib.axes.Axes = None,
                color: Optional[Union[List[str], Tuple[str, ...]]] = None,
                use_columns: Optional[bool] = None,
                xticks: Optional[Union[list, tuple,]] = None,
                colormap: Optional[Union[matplotlib.colors.Colormap, str]] = None,
                axvlines: Optional[bool] = None,
                axvlines_kwds: Optional[Mapping[str, Any]] = None,
                sort_labels: Optional[bool] = None,
                **kwargs,
            ) -> matplotlib.axes.Axes:
    """
    Parallel coordinates plotting.

    Parameters
    ----------
    frame : DataFrame
        The DataFrame to be plotted.
    class_column : str
        Column name containing class names.
    cols : list, optional
        A list of column names to use.
    ax : matplotlib.axis, optional
        Matplotlib axis object.
    color : list or tuple, optional
        Colors to use for the different classes.
    use_columns : bool, optional
        If true, columns will be used as xticks.
    xticks : list or tuple, optional
        A list of values to use for xticks.
    colormap : str or matplotlib colormap, default None
        Colormap to use for line colors.
    axvlines : bool, optional
        If true, vertical lines will be added at each xtick.
    axvlines_kwds : keywords, optional
        Options to be passed to axvline method for vertical lines.
    sort_labels : bool, default False
        Sort class_column labels, useful when assigning colors.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the parallel coordinates plot.

    See Also
    --------
    plotting.andrews_curves : Generate a matplotlib plot for visualizing clusters
        of multivariate data.
    plotting.radviz : Plot a multidimensional dataset in 2D.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> df = pd.read_csv(
        ...     "https://raw.githubusercontent.com/pandas-dev/"
        ...     "pandas/main/pandas/tests/io/data/csv/iris.csv"
        ... )  # doctest: +SKIP
        >>> pd.plotting.parallel_coordinates(
        ...     df, "Name", color=("#556270", "#4ECDC4", "#C7F464")
        ... )  # doctest: +SKIP
    """
    if use_columns is None:
        use_columns = False

    if axvlines is None:
        axvlines = True

    if sort_labels is None:
        sort_labels = False

    axvlines_kwds = utils.meta.set_default_kwargs(
                        axvlines_kwds,
                        dkwargs=dict(color=plt.rcParams["axes.edgecolor"], lw=0.5),
                        )

    if color is None:
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    kwargs = utils.meta.set_default_kwargs(
                            dkwargs=dict(lw=0.5, alpha=0.5),
                            )

    ax = pd.plotting.parallel_coordinates(
                frame=frame,
                class_column=class_column,
                cols=cols,
                ax=ax,
                color=color,
                use_columns=use_columns,
                xticks=xticks,
                colormap=colormap,
                axvlines=axvlines,
                axvlines_kwds=axvlines_kwds,
                sort_labels=sort_labels,
                **kwargs,
            )
    ax.grid(False)

    return ax


def parallel_coordinates_column(
            ax: np.ndarray,
            frame: pd.DataFrame,
            class_column: str,
            row_column: Optional[str] = None,
            **kwargs,
            ):

    index_names = [class_column]
    if row_column is not None:
        rc_vals = frame.reset_index()[row_column].unique()
        index_names.append(row_column)
    else:
        rc_vals = [None]

    index_names_drop = np.array(frame.index.names)[
                            ~np.isin(frame.index.names, index_names)
                            ]
    if frame.index.names != pd.DataFrame().index.names:
        frame = frame.reset_index()
    frame = frame.drop(index_names_drop, axis=1).set_index(index_names)

    for i, s in enumerate(rc_vals):
        if row_column is not None:
            frame_plot = frame.loc[pd.IndexSlice[:, s], :].droplevel([1], axis=0)
        else:
            frame_plot = frame

        parallel_coordinates(
                        frame=frame_plot.reset_index(),
                        class_column=class_column,
                        ax=ax[i],
                        **kwargs
                        )

    return ax
