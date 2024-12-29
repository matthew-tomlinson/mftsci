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
    """
    Rounds a scalar value to the nearest multiple of a specified factor.

    Parameters
    ----------
    val : Union[float, int,], optional
        The value to round. Default is 0.
    factor : Union[float, int,], optional
        The factor to which the value is rounded. Default is 1.
    buffer : Union[float, int,], optional
        The buffer to add before rounding. Default is 0.
    buffer_sign : Literal[-1, 1], optional
        The sign of the buffer. Default is inferred from round_type.
    round_type : Literal["round", "round_math", "floor", "ceil",], optional
        The type of rounding to apply. Default is "round".

    Returns
    -------
    Union[float, int,]
        The rounded value.

    Examples
    --------
    >>> _multiple_round_scalar(val=5.6, factor=2)
    6

    >>> _multiple_round_scalar(val=5.6, factor=2, round_type="floor")
    4

    >>> _multiple_round_scalar(val=5.6, factor=2, buffer=0.1)
    8
    """

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
    """
    Rounds an array or scalar value to the nearest multiple of a specified factor.

    Parameters
    ----------
    val : Union[np.ndarray, float, int,], optional
        The values to round. Default is 0.
    factor : Union[np.ndarray, float, int,], optional
        The factors to which the values are rounded. Default is 1.
    buffer : Union[np.ndarray, float, int,], optional
        The buffers to add before rounding. Default is 0.
    buffer_sign : Union[np.ndarray, Literal[-1, 1]], optional
        The signs of the buffers. Default is inferred from round_type.
    round_type : Union[np.ndarray,
                        Literal["round", "round_math",
                                "floor", "ceil",]
                    ], optional
        The types of rounding to apply. Default is "round".

    Returns
    -------
    Union[np.ndarray, float, int,]
        The rounded values.

    Examples
    --------
    >>> multiple_round(val=np.array([5.6, 3.4]), factor=2)
    array([6, 4])

    >>> multiple_round(val=5.6, factor=2, round_type="floor")
    4

    >>> multiple_round(val=[5.6, 3.4], factor=2, buffer=0.1)
    array([8, 4])
    """

    return _multiple_round_vector(**locals())


def calc_val_1_0(
                    app_val_1: Optional[bool] = None,
                    app_val_1_default: Optional[bool] = None,
                    val_1: Optional[object] = None,
                    val_0: Optional[object] = None,
                    ) -> object:
    """
    Returns one of two values based on a boolean flag.

    Parameters
    ----------
    app_val_1 : bool, optional
        The boolean flag to determine the output value. Default is inferred from
        `app_val_1_default`.
    app_val_1_default : bool, optional
        The default value for `app_val_1`. Default is True.
    val_1 : object, optional
        The value to return when `app_val_1` is True. Can be of any type.
    val_0 : object, optional
        The value to return when `app_val_1` is False. Can be of any type.

    Returns
    -------
    object
        The selected value based on `app_val_1`.

    Examples
    --------
    >>> calc_val_1_0(app_val_1=True, val_1=1, val_0=0)
    1

    >>> calc_val_1_0(val_1='Y', val_0='N', app_val_1=False)
    'N'

    >>> calc_val_1_0(val_1=1, val_0=0)
    1
    """

    if app_val_1 is None:
        if app_val_1_default is None:
            app_val_1_default = True
        app_val_1 = app_val_1_default

    if app_val_1:
        return val_1
    else:
        return val_0
