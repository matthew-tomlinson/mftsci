"""
Core module for meta utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable


# External packages
import numpy as np

# Internal packages


# =====================================
# FUNCTIONS
# =====================================


def return_args(
                *args,
                **kwargs,
                ):
    return (*args, *kwargs.values())


def return_args_(
                _,
                *args,
                **kwargs,
                ):
    args = return_args(
            *args,
            **kwargs,
            )
    if len(args) == 0:
        return _
    else:
        return (_, *args)


def return_kwargs(
                **kwargs,
                ):
    return kwargs


def return_x(
                x: Optional[object] = None,
                ) -> object:
    """
    Returns the input object x.

    Parameters
    ----------
    x : object, optional
        The object to be returned. If not provided, defaults to None.

    Returns
    -------
    object
        The input object x.

    Examples
    --------
    >>> return_x()
    None

    >>> return_x(10)
    10

    >>> return_x('hello')
    'hello'
    """

    return x


def set_default(
                val: Optional[object] = None,
                default: Optional[object] = None,
                default_call: Optional[Callable] = None,
                ) -> object:
    """
    Sets a value to the default if it is None.

    Parameters
    ----------
    val : object, optional
        The value to check. If it is None, the default value is returned. Defaults to
        None.
    default : object, optional
        The default value to set if `val` is None and `default_call` is None. Defaults
        to None.
    default_call : Callable, optional
        Returns the default value to set if `val` is None. Defaults to None.

    Returns
    -------
    object
        The value if it is not None, otherwise the default value.

    Examples
    --------
    >>> set_default(None, 10, None)
    10

    >>> set_default(None, 10, lambda: 20)
    20

    >>> set_default(30, 10, lambda: 20)
    30

    >>> set_default(None, None, None)
    None
    """

    if val is None:
        if default_call is not None:
            val = default_call()
        elif default is not None:
            val = default
    return val


def astype(
            val: object,
            type: Optional[Union[type, str,]] = None,
            ) -> object:

    if type is not None:
        if np.isscalar(val):
            val = type(val)
        elif isinstance(val, list):
            val = list(map(lambda x: type(x), val))
        elif isinstance(val, tuple):
            val = tuple(map(lambda x: type(x), val))
        elif isinstance(val, set):
            val = set(map(lambda x: type(x), val))
        elif isinstance(val, dict):
            val = dict(zip(val.keys(), map(lambda x: type(x), val.values())))
        else:
            val = val.astype(type)

    return val
