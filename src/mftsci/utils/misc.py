"""
Module for miscellaneous utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Literal

# External packages
import numpy as np

# Internal packages


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def _multiple_round_scalar(
                            val: Optional[Union[float, int,]] = None,
                            factor: Optional[Union[float, int,]] = None,
                            buffer: Optional[Union[float, int,]] = None,
                            buffer_sign: Optional[Literal[-1, 1]] = None,
                            round_type: Optional[Literal["round", "round_math",
                                                         "floor", "ceil",]] = None,
                            ) -> Union[float, int,]:

    if val is None:
        val = 0
    if factor is None:
        factor = 1
    if buffer is None:
        buffer = 0
    if round_type is None:
        round_type = "round"

    if buffer_sign is None:
        if round_type == "floor":
            buffer_sign = -1
        else:
            buffer_sign = 1

    if round_type == "round_math":
        return factor*int(np.floor((val/factor) + buffer + 0.5))
    else:
        return factor*int(np.__dict__[round_type]((val/factor)+buffer_sign*buffer))


_multiple_round_vector = np.vectorize(_multiple_round_scalar)


def multiple_round(
                    val: Optional[Union[np.ndarray, float, int,]] = None,
                    factor: Optional[Union[np.ndarray, float, int,]] = None,
                    buffer: Optional[Union[np.ndarray, float, int,]] = None,
                    buffer_sign: Optional[
                                            Union[
                                                np.ndarray,
                                                Literal[-1, 1],
                                                ]
                                        ] = None,
                    round_type: Optional[
                                            Union[
                                                    np.ndarray,
                                                    Literal[
                                                        "round", "round_math",
                                                        "floor", "ceil",
                                                        ],
                                                ]
                                        ] = None,
                    ) -> Union[np.ndarray, float, int,]:

    return _multiple_round_vector(**locals())


def calc_val_1_0(
                    app_val_1: Optional[bool] = None,
                    app_val_1_default: Optional[bool] = None,
                    val_1: Optional[object] = None,
                    val_0: Optional[object] = None,
                    ) -> object:

    if app_val_1 is None:
        if app_val_1_default is None:
            app_val_1_default = True
        app_val_1 = app_val_1_default

    if app_val_1:
        return val_1
    else:
        return val_0
