"""
Module for plot utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Literal
import warnings

# External packages
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.dates as mdates   # noqa: F401
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)
from matplotlib import ticker as mticker    # noqa: F401

# Internal packages
from mftsci.utils import arr
from mftsci.utils.str import str_concatenate
from mftsci.utils.char import calc_greek_roman_tex_preamble_print_lines \
                                as _cGR_tex_preamble_print_lines   # noqa: F401


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def get_matplotlib_use_tex():

    return matplotlib.rcParams["text.usetex"]


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

    return str_concatenate(calc_tex_preamble_print_lines(), fill_str="\n")


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

    if use_tex is None:
        use_tex = False

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
    if minor_visible is None:
        minor_visible = True
    if minor_visible is not None:
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

    if font_family is None:
        font_family = "sans-serif"
        # font_family = "serif"

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

    if supress_warnings is None:
        supress_warnings = True

    if supress_warnings:
        warnings.filterwarnings(
            action="ignore", module="matplotlib.figure", category=UserWarning,
            message=("""
                    This figure includes Axes that are not compatible with tight_layout,
                    so results might be incorrect.
                    """)
            )


def integer_ticks(
                    ax: matplotlib.axes._axes.Axes,
                    xys: Optional[Union[np.ndarray, list, Literal["x", "y"]]] = None,
                    maj_mult: Optional[int] = None,
                    min_mult: Optional[int] = None,
                    app_minor: Optional[bool] = None,
                    maj_offset: Optional[Union[float, int]] = None,
                    min_offset: Optional[Union[float, int]] = None,
                    ):

    if xys is None:
        xys = ["x", "y"]
    xys = arr.iterarr(xys)

    if app_minor is None:
        if min_mult is not None or min_offset is not None:
            app_minor = np.repeat(True, xys.size)
        else:
            app_minor = np.array([plt.rcParams[f"{xy}tick.minor.visible"]
                                  for xy in xys])

    if maj_offset is None:
        maj_offset = 0.0

    if min_offset is None:
        min_offset = maj_offset

    for k in range(xys.size):
        xy = xys[k]

        if maj_mult is None:
            ax.__dict__[f"{xy}axis"].set_major_locator(MaxNLocator(integer=True,
                                                                   nbins="auto"))
            maj_mult = int(np.diff(ax.__getattribute__(f"get_{xy}ticks")()[:2]))
        else:
            ax.__dict__[f"{xy}axis"].set_major_locator(MultipleLocator(maj_mult))

        # maj_ticks = ax.__dict__[f"{xy}axis"].get_major_ticks()

        if min_mult is None:
            if np.isin(maj_mult, [6, 8, 10]):
                min_mult = 2
            elif np.isin(maj_mult, [9, 12, 15]):
                min_mult = 3
            elif maj_mult < 8:
                min_mult = 1

        if app_minor[k]:
            if min_mult is None:
                ax.__dict__[f"{xy}axis"].set_minor_locator(AutoMinorLocator())
            else:
                ax.__dict__[f"{xy}axis"].set_minor_locator(MultipleLocator(min_mult))


# Defaults
# -------------------------------------


def default_dpi(
                dpi: Optional[Union[float, int]] = None,
                ) -> Union[float, int]:

    if dpi is None:
        dpi = 600
    return dpi
