"""
Defaults call module for miscellaneous utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Union, Literal

# External packages


# Internal packages


# =====================================
# DEFAULTS
# =====================================


def val() -> Union[float, int,]:

    return 0


def factor() -> Union[float, int,]:

    return 1


def buffer() -> Union[float, int,]:

    return 0


def round_type() -> Literal[
                        "round", "round_math",
                        "floor", "ceil",
                        ]:

    return "round"


def app_val_1() -> bool:

    return None


def app_val_1_default() -> bool:

    return True


def val_1() -> object:

    return None


def val_0() -> object:

    return None
