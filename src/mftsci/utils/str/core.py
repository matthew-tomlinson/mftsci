"""
Core module for string utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional

# External packages
import matplotlib


# Internal packages


# =====================================
# FUNCTIONS
# =====================================


def get_matplotlib_use_tex() -> bool:
    return matplotlib.rcParams["text.usetex"]


def num_convert(
                s: object,
                incl_int: Optional[bool] = None,
                ) -> object:

    try:
        if isinstance(s, bool):
            return s
        f = float(s)
        s = str(s)

        if incl_int is None:
            incl_int = True
        if incl_int:
            if str(s).find(".") < 0:
                return int(f)
        return f
    except:     # noqa: E722
        if s == "True" or s == "False":
            return bool(s)

        return s
