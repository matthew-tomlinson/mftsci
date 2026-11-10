"""
Functions module for table utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional
import os

# External packages
import numpy as np

# Internal packages
from mftsci.utils.tab import defaults
from mftsci.utils.arr import fmap
from mftsci.utils import colour as clr
from mftsci.utils.colour import pd_colour, pd_scolour   # noqa: F401

from mftsci.utils.file import WPath as _WPath
from mftsci.utils.html import write as _write_html


# =====================================
# CLASSES
# =====================================


class Path(_WPath):

    _default_subdir = os.path.normpath(
                    os.path.join(
                        _WPath.default_subdir(),
                        "tabs",
                        )
                    )
    _default_file = "table"


# =====================================
# FUNCTIONS
# =====================================

def empty_kwargs(index):

    return {v: {} for v in index}


def df_style(
                df,
                app_master=None,
                app_default=None,
                app_vpad=None,
                app_hpad=None,
                app_valign=None,
                app_halign=None,
                app_font=None,
                app_font_weight=None,
                app_colour=None,
                colours_bg_header=None,
                colours_bg_index=None,
                colours_bg_data=None,
                colours_text_header=None,
                colours_text_index=None,
                colours_text_data=None,
                font_kwargs=None,
                invert_colour=None,
                line_height=None,
                valign=None,
                valign_index=None,
                halign=None,
                halign_index=None,
                halign_index_names=None,
                halign_columns=None,
                halign_columns_names=None,
                halign_data=None,
                cond_ralign_col_lastlevel=None,
                ):

    app_master = defaults.app_master(app_master)

    if False:
        app_vpad = defaults.app_vpad(app_vpad)
        app_hpad = defaults.app_hpad(app_hpad)
        app_valign = defaults.app_valign(app_valign)
        app_halign = defaults.app_halign(app_halign)
        app_font = defaults.app_font(app_font)
        app_font_weight = defaults.app_font_weight(app_font_weight)
        app_colour = defaults.app_colour(app_colour)
    else:
        app_vpad = defaults.set_default(
                    app_vpad, default=defaults.app_vpad(app_default)
                    )
        app_hpad = defaults.set_default(
                    app_hpad, default=defaults.app_hpad(app_default)
                    )
        app_valign = defaults.set_default(
                    app_valign, default=defaults.app_valign(app_default)
                    )
        app_halign = defaults.set_default(
                    app_halign, default=defaults.app_halign(app_default)
                    )
        app_font = defaults.set_default(
                    app_font, default=defaults.app_font(app_default)
                    )
        app_font_weight = defaults.set_default(
                    app_font_weight, default=defaults.app_font_weight(app_default)
                    )
        app_colour = defaults.set_default(
                    app_colour, default=defaults.app_colour(app_default)
                    )

    valign = defaults.valign(valign)
    valign_index = defaults.valign_index(valign_index)

    halign = defaults.halign(halign)
    halign_index = defaults.halign_index(halign_index)
    # halign_index_names = defaults.halign_index_names(halign_index_names)
    halign_index_names = defaults.set_default(halign_index_names, halign_index)
    halign_columns = defaults.halign_columns(halign_columns)
    halign_columns_names = defaults.halign_columns_names(halign_columns_names)
    halign_data = defaults.halign_data(halign_data)

    invert_colour = defaults.invert_colour(invert_colour)
    line_height = defaults.line_height(line_height)
    cond_ralign_col_lastlevel = defaults.cond_ralign_col_lastlevel(cond_ralign_col_lastlevel)   # noqa: E501

    N_index_head = int(np.any([i is not None for i in df.index.names]))
    N_levels = df.columns.nlevels
    N_head = N_index_head + N_levels

    cell_sel = {
                "all": "th, td",
                "columns": "thead th.col_heading",
                "columns_lastlevel": f"thead th.col_heading.level{N_levels-1}",
                "index": "tbody th",
                "data": "tbody td"
                }

    lname_sel = {
                "columns": [
                            f"thead tr:nth-child({1+i}) th.index_name"
                            for i in range(N_levels)
                            ],
                "index": [
                            "thead tr:last-child th.index_name"
                            for i in range(N_index_head)
                            ],
                }
    selectors = [*cell_sel.values(), *[v for lst in lname_sel.values() for v in lst]]
    kwargs_index = ["vpad", "hpad", "valign", "halign", "font", "font_weight"]

    kwargs = {
                s: empty_kwargs(kwargs_index)
                for s in selectors
                }
    index_band_styles = []
    data_band_styles = []
    header_band_styles = []

    def zebra_band_row(row):
        return [""] * df.shape[1]

    if app_master:
        if app_vpad:
            kwargs[cell_sel["all"]]["vpad"] = {
                                                "line-height": line_height,
                                                }

        if app_valign:
            kwargs[cell_sel["all"]]["valign"] = {
                                                "vertical-align": valign,
                                                }
            kwargs[cell_sel["index"]]["valign"] = {
                                                "vertical-align": valign_index,
                                                }

        if app_halign:
            kwargs[cell_sel["all"]]["halign"] = {
                                                "text-align": halign,
                                                }
            for c in lname_sel["columns"]:
                kwargs[c]["halign"] = {
                                        "text-align": halign_columns_names,
                                        }
            for c in lname_sel["index"]:
                kwargs[c]["halign"] = {
                                        "text-align": halign_index_names,
                                        }

            kwargs[cell_sel["columns"]]["halign"] = {
                                                    "text-align": halign_columns,
                                                    }
            if cond_ralign_col_lastlevel and N_index_head == 0:
                kwargs[cell_sel["columns_lastlevel"]]["halign"] = {
                                                    "text-align": "right"
                                                    }

            kwargs[cell_sel["index"]]["halign"] = {
                                                    "text-align": halign_index,
                                                    }
            kwargs[cell_sel["data"]]["halign"] = {
                                                    "text-align": halign_data,
                                                    }

        if app_hpad:
            kwargs[cell_sel["all"]]["hpad"] = {
                                                "padding-left": "8px",
                                                "padding-right": "8px",
                                                }

            if halign_data == "left":
                hpad_data_left = 6
                hpad_data_right = 10
            elif halign_data == "right":
                hpad_data_left = 10
                hpad_data_right = 6
            else:
                hpad_data_left = 8
                hpad_data_right = 8

            kwargs[cell_sel["data"]]["hpad"] = {
                                            "padding-left": f"{hpad_data_left}px",
                                            "padding-right": f"{hpad_data_right}px",
                                            }

        if app_font:
            if font_kwargs is None:
                font_kwargs = {}

            font_vscode_defaults_base = {
                                        "font-size": "10.5pt",
                                        "font-family": "'Segoe UI', sans",
                                    }
            extra_defaults = {
                                "font-feature-settings": "'kern' 1, 'liga' 0",
                                "-webkit-font-smoothing": "antialiased",
                                "-moz-osx-font-smoothing": "grayscale",
                                "text-rendering": "optimizeLegibility",
                                "font-weight": "400",
                            }
            for v in font_vscode_defaults_base:
                font_kwargs.setdefault(v, font_vscode_defaults_base[v])
            if False:
                for v in extra_defaults:
                    font_kwargs.setdefault(v, extra_defaults[v])

            kwargs[cell_sel["all"]]["font"] = font_kwargs

        if app_font_weight:
            for c in lname_sel["columns"]:
                kwargs[c]["font_weight"] = {
                                            "font-weight": "bolder",
                                            }
            for c in lname_sel["index"]:
                kwargs[c]["font_weight"] = {
                                            "font-weight": "bolder",
                                            }
            kwargs[cell_sel["columns"]]["font_weight"] = {
                                                        "font-weight": "bold",
                                                        }
            kwargs[cell_sel["index"]]["font_weight"] = {
                                            "font-weight": "normal",
                                            }

        if app_colour:

            colours_bg_header = defaults.colours_bg_header(colours_bg_header)
            if colours_bg_index is None:
                if colours_bg_data is not None:
                    colours_bg_index = colours_bg_data
                else:
                    colours_bg_index = defaults.colours_bg_data()
            if colours_bg_data is None:
                colours_bg_data = colours_bg_index

            colours_text_header = defaults.colours_text_header(colours_text_header)
            if colours_text_index is None:
                if colours_text_data is not None:
                    colours_text_index = colours_text_data
                else:
                    colours_text_index = defaults.colours_text_data()
            if colours_text_data is None:
                colours_text_data = colours_text_index

            colours = {
                        "bg_header": colours_bg_header,
                        "bg_index": colours_bg_index,
                        "bg_data": colours_bg_data,
                        "text_header": colours_text_header,
                        "text_index": colours_text_index,
                        "text_data": colours_text_data,
                    }

            for h in ["header", "index", "data"]:
                colours[f"bg_{h}"] = clr.capp_to_rgba(colours[f"bg_{h}"])
                if colours[f"text_{h}"] == "hc":
                    colours[f"text_{h}"] = np.repeat(
                                                "white",
                                                colours[f"bg_{h}"].shape[0],
                                                )
                    flip = clr.calc_greyscale_val(colours[f"bg_{h}"]) > 0.5
                    colours[f"text_{h}"][flip] = "black"

            for c in colours:
                colours[c] = clr.capp_to_rgba(colours[c])
                if invert_colour:
                    colours[c] = clr.invert(colours[c])
                colours[c] = fmap(
                                    colours[c],
                                    lambda v: clr.capp_to_hex(v, app_to_hex=True),
                                    axes=1,
                                    )

            # Create a mapping from index and data row position to band color
            def zebra_band_row(row):    # noqa: F811
                i = df.index.get_loc(row.name)
                data_i_colour = {
                            p: colours[f"{p}_data"][i % colours[f"{p}_data"].shape[0]]
                            for p in ["bg", "text"]
                            }
                return [
                            f"background-color: {data_i_colour['bg']}; color: {data_i_colour['text']}"    # noqa: E501, W505
                        ] * df.shape[1]

            for i in range(len(df.index)):
                index_i_colour = {
                            p: colours[f"{p}_index"][i % colours[f"{p}_index"].shape[0]]
                            for p in ["bg", "text"]
                            }
                index_band_styles.append({
                    "selector": f"tbody th.row{i}",  # Index row i
                    "props": [
                                ("background-color", index_i_colour["bg"]),
                                ("color", index_i_colour["text"]),
                                ]
                })

                data_i_colour = {
                            p: colours[f"{p}_data"][i % colours[f"{p}_data"].shape[0]]
                            for p in ["bg", "text"]
                            }
                data_band_styles.append({
                    "selector": f"tbody td.row{i}",  # Data row i
                    "props": [
                                ("background-color", data_i_colour["bg"]),
                                ("color", data_i_colour["text"]),
                                ]
                })

            # Create a mapping from header row position to band color
            for _i in range(N_head):
                i = (N_head - 1) - _i
                i_colour = {
                            p: colours[f"{p}_header"][
                                i % colours[f"{p}_header"].shape[0]
                                ]
                            for p in ["bg", "text"]
                            }
                header_band_styles.append({
                    "selector": f"thead tr:nth-child({_i+1}) th",
                    "props": [
                                ("background-color", i_colour["bg"]),
                                ("color", i_colour["text"]),
                                ]
                })

    base_styles = [{
            "selector": s,
            "props": [t for f in kwargs[s] for t in zip(kwargs[s][f].keys(), kwargs[s][f].values())]    # noqa: E501
            } for s in kwargs]

    d = kwargs[cell_sel["all"]]["vpad"]
    post = "; ".join([":".join(np.array(a)) for a in zip(d.keys(), d.values())])
    if len(post) > 0:
        post = " " + post + ";"

    df_style = (
        df.style
        .set_table_styles([
            *base_styles,
            *header_band_styles,
            *index_band_styles,
            # *data_band_styles,
        ])
        .set_table_attributes(f"style='border-collapse:collapse;{post}'")   # noqa: E501
        .apply(zebra_band_row, axis=1)
    )

    return df_style


def write_html(
            s: str,
            file: Optional[str] = None,
            path: Optional[str] = None,
            ext: Optional[str] = None,
            app_ext: Optional[bool] = None,
            app_write: Optional[bool] = None,
            app_mkdir: Optional[bool] = None,
            app_print_path: Optional[bool] = None,
            dPath: Optional[Path] = None,
            ):

    if dPath is None:
        dPath = Path

    return _write_html(
            s=s, file=file, path=path,
            ext=ext, app_ext=app_ext,
            app_write=app_write,
            app_mkdir=app_mkdir, app_print_path=app_print_path,
            dPath=dPath,
            )
