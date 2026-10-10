"""
Defaults module for table utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401
from mftsci.utils.colour.defaults import invert_colour    # noqa: F401
import mftsci.utils.colour as clr

# =====================================
# DEFAULTS
# =====================================


def app_master(
       app_master: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_master, default=True)


def app_vpad(
       app_vpad: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_vpad, default=True)


def app_hpad(
       app_halign: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_hpad, default=True)


def app_valign(
       app_valign: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_valign, default=True)


def app_halign(
       app_halign: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_halign, default=True)


def app_font(
       app_font: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_font, default=True)


def app_font_weight(
       app_font_weight: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_font_weight, default=True)


def app_colour(
       app_colour: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=app_colour, default=True)


def line_height(
       line_height: Optional[Union[float, int]] = None,
        ) -> Union[float, int]:

    return set_default(val=line_height, default=1.2)


def valign(
       valign: Optional[str] = None,
        ) -> str:

    return set_default(val=valign, default="top")


def valign_index(
       valign_index: Optional[str] = None,
        ) -> str:

    return set_default(val=valign_index, default="top")


def halign(
       halign: Optional[str] = None,
        ) -> str:

    return set_default(val=halign, default="left")


def halign_index(
       halign_index: Optional[str] = None,
        ) -> str:

    return set_default(val=halign_index, default="right")


def halign_index_names(
       halign_index_names: Optional[str] = None,
        ) -> str:

    # return set_default(val=halign_index_names, default="left")
    return halign_index(halign_index_names)


def halign_columns(
       halign_columns: Optional[str] = None,
        ) -> str:

    return set_default(val=halign_columns, default="left")


def halign_columns_names(
       halign_columns_names: Optional[str] = None,
        ) -> str:

    return set_default(val=halign_columns_names, default="right")


def halign_data(
       halign_data: Optional[str] = None,
        ) -> Union[str]:

    return set_default(val=halign_data, default="right")


def halign_data_numeric(
       halign_data_numeric: Optional[str] = None,
        ) -> Union[str]:

    return set_default(val=halign_data_numeric, default="right")


def cond_ralign_col_lastlevel(
        cond_ralign_col_lastlevel: Optional[bool] = None,
        ) -> Union[bool]:

    return set_default(val=cond_ralign_col_lastlevel, default=False)


def colours_bg_header(
        colours_bg_header: Optional[Union[list, np.ndarray, pd.Series, str,]] = None,
        ) -> Union[list, np.ndarray, pd.Series,]:

    _colours = ["#2f2f2f", "#353535"]
    if not clr.is_dark_theme():
        _colours = [
                    clr.capp_to_hex(v, app_to_hex=True)
                    for v in clr.invert(_colours)
                    ]

    return set_default(val=colours_bg_header, default=_colours)


def colours_bg_data(
        colours_bg_data: Optional[Union[list, np.ndarray, pd.Series, str,]] = None,
        ) -> Union[list, np.ndarray, pd.Series, str,]:

    _colours = ["#1f1f1f", "#272727"]
    if not clr.is_dark_theme():
        _colours = [
                    clr.capp_to_hex(v, app_to_hex=True)
                    for v in clr.invert(_colours)
                    ]

    return set_default(val=colours_bg_data, default=_colours)


def colours_text(
        colours_text: Optional[Union[list, np.ndarray, pd.Series, str,]] = None,
        ) -> Union[list, np.ndarray, pd.Series, str,]:

    _colours = "#cccccc"
    if not clr.is_dark_theme():
        _colours = [
                    clr.capp_to_hex(v, app_to_hex=True)
                    for v in clr.invert(_colours)
                    ]

    return set_default(val=colours_text, default=_colours)


def colours_text_header(
        colours_text_header: Optional[Union[list, np.ndarray, pd.Series, str,]] = None,
        ) -> Union[list, np.ndarray, pd.Series, str,]:

    return colours_text(colours_text_header)


def colours_text_data(
        colours_text_data: Optional[Union[list, np.ndarray, pd.Series, str,]] = None,
        ) -> Union[list, np.ndarray, pd.Series, str,]:

    return colours_text(colours_text_data)
