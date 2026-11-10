"""
Functions module for plot utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal, Tuple, Callable
import os
import warnings


# External packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.dates as mdates   # noqa: F401
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)
from matplotlib import ticker as mticker    # noqa: F401
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D


# Internal packages
from mftsci.utils import arr
from mftsci.utils.meta import default_kwargs
from mftsci.utils.str import get_matplotlib_use_tex     # noqa: F401
from mftsci.utils.char import calc_greek_roman_tex_preamble_print_lines \
                                as _cGR_tex_preamble_print_lines   # noqa: F401
from mftsci.utils.plot import defaults  # noqa: F401
from mftsci.utils.file import mkdir, WPath as _WPath, calc_path as _calc_path
from mftsci.utils.colour import c_vadd_diff


# =====================================
# CLASSES
# =====================================


class Path(_WPath):

    _default_subdir = os.path.normpath(
                        os.path.join(
                            _WPath.default_subdir(),
                            "figs",
                            )
                        )
    _default_file = "figure"


# =====================================
# FUNCTIONS
# =====================================


def calc_tex_preamble_print_lines():

    base_tex_preamble_print_lines = [f"\\usepackage{{siunitx}}"]    # noqa: F541
    return np.concatenate([
                            base_tex_preamble_print_lines,
                            # _cGR_tex_preamble_print_lines(
                            #                   app_omicron=True,
                            #                   def_Greek=True,
                            #                   )
                            ])


def calc_tex_preamble():

    return "\n".join([*calc_tex_preamble_print_lines(), ""])


def pyplot_setup(
                    font_family: Optional[str] = None,
                    font_text: Optional[Union[np.ndarray, list, str]] = None,
                    font_mathtext=None,
                    minor_visible: Optional[bool] = None,
                    use_tex: Optional[bool] = None,
                    SMALL_SIZE: Optional[Union[float, int]] = None,
                    MEDIUM_SIZE: Optional[Union[float, int]] = None,
                    LARGE_SIZE: Optional[Union[float, int]] = None,
                    style: Optional[str] = None,
                    supress_warnings: Optional[bool] = None,
                    ):

    use_tex = defaults.use_tex(use_tex)
    minor_visible = defaults.minor_visible(minor_visible)
    font_family = defaults.font_family(font_family)
    supress_warnings = defaults.supress_warnings(supress_warnings)

    if False:
        if SMALL_SIZE is None:
            SMALL_SIZE = 10
        if MEDIUM_SIZE is None:
            MEDIUM_SIZE = 11
        if LARGE_SIZE is None:
            LARGE_SIZE = 12

    if style is not None:
        plt.style.use(style)

    # Font
    if SMALL_SIZE is not None:
        plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
    if MEDIUM_SIZE is not None:
        plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    if LARGE_SIZE is not None:
        plt.rc("figure", titlesize=LARGE_SIZE)  # fontsize of the figure title

    plt.rc("figure", figsize=[7, 4])

    # Ticks
    plt.rc("xtick", top=True)
    plt.rc("ytick", right=True)

    if minor_visible:
        xys = ["x", "y"]
        for xy in xys:
            # plt.rc(f"{xy}tick", minor_visible=minor_visible)
            plt.rcParams[f"{xy}tick.minor.visible"] = minor_visible

    if use_tex:

        rc_fonts = {
                        "text.usetex": True,
                        # "text.latex.preview": True, # Gives correct legend alignment.
                        # "mathtext.default": "regular",
                        "text.latex.preamble": calc_tex_preamble(),
                    }
        plt.rcParams.update(rc_fonts)

    plt.rc("font", family=font_family)

    if font_text is None:
        font_text = plt.rcParams[f"font.{font_family}"]
    plt.rcParams[f"font.{font_family}"] = arr.iterarr(font_text, as_list=True)

    if font_mathtext is None:
        # font_mathtext = plt.rcParams[f"font.{font_family}"][0]
        if font_family == "sans-serif":
            font_mathtext = "dejavusans"
        else:
            font_mathtext = "dejavuserif"
            # font_mathtext = "cm"

    plt.rc("mathtext", fontset=font_mathtext)

    pd.plotting.register_matplotlib_converters()

    if supress_warnings:
        warnings.filterwarnings(
            action="ignore", module="matplotlib.figure", category=UserWarning,
            message=("""
                    This figure includes Axes that are not compatible with tight_layout,
                    so results might be incorrect.
                    """)
            )


def subplots(
                nrows: Optional[int] = None,
                ncols: Optional[int] = None,
                **kwargs,
                ) -> Tuple[matplotlib.figure.Figure, np.ndarray]:

    nrows = defaults.nrows(nrows)
    ncols = defaults.ncols(ncols)
    kwargs = defaults.set_default_kwargs(kwargs=kwargs, dkwargs=dict(layout="tight"))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)
    if ncols == 1:
        ax = np.expand_dims(ax, axis=1)

    return fig, ax


def integer_ticks(
                    ax: matplotlib.axes._axes.Axes,
                    xys: Optional[Union[np.ndarray, list, Literal["x", "y"]]] = None,
                    maj_mult: Optional[int] = None,
                    min_mult: Optional[int] = None,
                    app_minor: Optional[bool] = None,
                    maj_offset: Optional[Union[float, int]] = None,
                    min_offset: Optional[Union[float, int]] = None,
                    ):

    xys = defaults.xys(xys)
    maj_offset = defaults.maj_offset(maj_offset)

    if app_minor is None:
        if min_mult is not None or min_offset is not None:
            app_minor = np.repeat(True, xys.size)
        else:
            app_minor = np.array([plt.rcParams[f"{xy}tick.minor.visible"]
                                  for xy in xys])

    if min_offset is None:
        min_offset = maj_offset

    for k in range(xys.size):
        xy = xys[k]

        if maj_mult is None:
            getattr(ax, f"{xy}axis").set_major_locator(MaxNLocator(integer=True,
                                                                   nbins="auto"))
            maj_mult = int(np.diff(ax.__getattribute__(f"get_{xy}ticks")()[:2]))
        else:
            getattr(ax, f"{xy}axis").set_major_locator(MultipleLocator(maj_mult))

        # maj_ticks = getattr(ax, f"{xy}axis").get_major_ticks()

        if min_mult is None:
            if np.isin(maj_mult, [6, 8, 10]):
                min_mult = 2
            elif np.isin(maj_mult, [9, 12, 15]):
                min_mult = 3
            elif maj_mult < 8:
                min_mult = 1

        if app_minor[k]:
            if min_mult is None:
                getattr(ax, f"{xy}axis").set_minor_locator(AutoMinorLocator())
            else:
                getattr(ax, f"{xy}axis").set_minor_locator(MultipleLocator(min_mult))


def calc_path(
        file: Optional[str] = None,
        path: Optional[str] = None,
        ext: Optional[str] = None,
        app_ext: Optional[bool] = None,
        app_mkdir: Optional[bool] = None,
        app_print: Optional[bool] = None,
        dPath: Optional[Path] = None,
        ):

    if dPath is None:
        dPath = Path

    return _calc_path(
                file=file, path=path, ext=ext, app_ext=app_ext,
                app_mkdir=app_mkdir, app_print=app_print,
                dPath=dPath,
                )


def calc_savefig_kwargs(
            file: Optional[str] = None,
            fname: Optional[str] = None,
            savefig_kwargs: Optional[dict] = None,
            app_mkdir: Optional[bool] = None,
            app_print: Optional[bool] = None,
            dPath=None,
            ):

    app_mkdir = defaults.set_default(app_mkdir, False)
    app_print = defaults.set_default(app_print, False)

    savefig_kwargs = default_kwargs(savefig_kwargs)

    fname = calc_path(
                            file=file,
                            path=defaults.set_default(
                                        fname,
                                        savefig_kwargs.get("fname", None)
                                        ),
                            app_mkdir=False,
                            app_print=False,
                            dPath=dPath,
                            )
    # fname = savefig_kwargs.get("fname", defaults.fname(fname))
    fname_splitext = os.path.splitext(fname)
    fname_format = fname_splitext[1][1:]
    _format = savefig_kwargs.get("format", fname_format)

    # if not np.isin(_format, [*plt.gcf().canvas.get_supported_filetypes().keys()]):
    if not np.isin(_format, np.concat([*get_formats().values()])):
        _format = plt.rcParams["savefig.format"]

    savefig_kwargs["format"] = _format
    if np.isin(savefig_kwargs["format"], get_formats()["raster"]):
        savefig_kwargs.setdefault("dpi", defaults.dpi())

    fname = f"{fname_splitext[0]}.{_format}"
    savefig_kwargs["fname"] = mkdir(
                            fname,
                            app_mkdir=app_mkdir, app_print=app_print,
                            ret_dir=False,
                            )

    return savefig_kwargs


def savefig(
            fig: plt.Figure,
            file: Optional[str] = None,
            fname: Optional[str] = None,
            savefig_kwargs: Optional[dict] = None,
            app_savefig: Optional[bool] = None,
            app_mkdir: Optional[bool] = None,
            app_print: Optional[bool] = None,
            dPath=None,
            ):

    app_savefig = defaults.set_default(app_mkdir, True)

    if app_savefig:
        app_mkdir = defaults.set_default(app_mkdir, True)
        app_print = defaults.set_default(app_print, True)

        fig.savefig(**calc_savefig_kwargs(
                    file=file,
                    fname=fname,
                    savefig_kwargs=savefig_kwargs,
                    app_mkdir=app_mkdir,
                    app_print=app_print,
                    dPath=dPath,
                    ))


def get_formats():

    formats = {
                "raster": [
                            "png", "jpg", "jpeg", "tif", "tiff",
                            "bmp", "raw", "rgba", "webp",
                            ],
                "vector": ["pdf", "svg", "svgz", "eps", "ps", "pgf"],
                }

    return formats


def calc_edges(
        ):

    edges = np.array([
                ["bottom", "top"],
                ["left", "right"],
            ])

    return edges


def calc_tickl():

    tickl = ["", "label"]

    return tickl


def coloured_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def transform_obj_ax(
            obj,
            axis: Optional[int],
            tfunc: Optional[Callable] = None
            ):

    axes = pd.Series(["x", "y", "z"])
    if isinstance(axis, str):
        axis = axis[:1].lower()

    if tfunc is not None:
        if isinstance(obj, matplotlib.lines.Line2D):
            if isinstance(axis, str):
                ax = axis
            else:
                ax = axes.loc[axis]

            getattr(obj, f"set_{ax}data")(tfunc(getattr(obj, f"get_{ax}data")()))

        elif isinstance(obj, matplotlib.collections.FillBetweenPolyCollection):
            if isinstance(axis, int):
                ax = axis
            else:
                ax = axes.reset_index().set_index(0).iloc[:, 0].loc[axis]

            obj.get_paths()[0].vertices[:, ax] = tfunc(
                                                    obj.get_paths()[0].vertices[:, ax]
                                                    )

    return obj


def edge_scatter(
        *args,
        plot_marker=None,
        N_edges=None,
        edge_kwargs=None,
        edgewidth_factor=None,
        **kwargs,
        ):

    if plot_marker is None:
        plot_marker = True

    if edge_kwargs is None:
        if N_edges is not None:
            cols = c_vadd_diff(v=[0.0, 1.0])    # [-N_edges:]
            N_cols = len(cols)
            edge_kwargs = [{"c": cols[i % N_cols]} for i in range(N_edges)][::-1]
        else:
            edge_kwargs = []
    elif isinstance(edge_kwargs, dict):
        edge_kwargs = [edge_kwargs]

    N_edges = len(edge_kwargs)

    if edgewidth_factor is None:
        edgewidth_factor = 3
    edgewidths = (1 + np.arange(N_edges)) * edgewidth_factor
    for i, d in enumerate(edge_kwargs):
        d.setdefault("lw", edgewidths[-(1+i)])

    ax = kwargs.pop("ax", None)
    if ax is None:
        fig, ax = plt.subplots()

    is_3d = isinstance(ax, Axes3D)
    if is_3d:
        N_axes = 3
    else:
        N_axes = 2

    N_args = len(args)

    x_kw = np.array(["x", "y", "z"])[:N_axes]
    extra_args_kw = ["s", "c"]
    if is_3d:
        x_kw = x_kw + "s"
        extra_args_kw = ["zdir", *extra_args_kw, "depthshade"]

    X = [*args[:N_axes], *[None]*max(0, N_axes - N_args)]
    for i, x in enumerate(x_kw):
        _x = kwargs.pop(x, None)
        if X[i] is None:
            X[i] = _x

    for i, k in extra_args_kw[:max(0, N_args - N_axes)]:
        kwargs[k] = args[N_axes + i]

    # kwargs.setdefault("c", c_vadd_diff(1.0))
    e_kw = ["marker", "s", "zorder"]
    ekwargs = {
                k: v for k, v in
                {k: kwargs.get(k, None) for k in e_kw}.items()
                if v is not None
                }
    ret_edge = [ax.scatter(*X, **ekwargs, **e) for e in edge_kwargs]

    if plot_marker:
        kwargs.setdefault("lw", 0)
        ret_marker = ax.scatter(*X, **kwargs)
    else:
        ret_marker = None

    return ret_edge, ret_marker
