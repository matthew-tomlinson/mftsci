"""
Defaults module for miscellaneous utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal

# External packages


# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.misc.defaults import _defaults


# =====================================
# DEFAULTS
# =====================================

def val(
        val: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=val, default_call=_defaults.val)


def factor(
        factor: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=factor, default_call=_defaults.factor)


def buffer(
        buffer: Optional[Union[float, int,]] = None,
        ) -> Union[float, int,]:

    return set_default(val=buffer, default_call=_defaults.buffer)


def round_type(
        round_type: Optional[Literal[
                                        "round", "round_math",
                                        "floor", "ceil",
                                        ]] = None,
        ) -> Literal[
                        "round", "round_math",
                        "floor", "ceil",
                        ]:

    return set_default(val=round_type, default_call=_defaults.round_type)


def app_val_1_default(
        app_val_1_default: Optional[bool] = None,
        ) -> bool:

    return set_default(val=app_val_1_default, default_call=_defaults.app_val_1_default)
