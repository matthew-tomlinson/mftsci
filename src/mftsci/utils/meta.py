"""
Module for meta utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Tuple

# External packages


# Internal packages


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================

def default_args(
                    args: Optional[tuple] = None,
                    ) -> tuple:
    """
    Returns default positional arguments if None are provided.

    Parameters
    ----------
    args : tuple, optional
        The input positional arguments. Default is ().

    Returns
    -------
    tuple
        The default or provided positional arguments.

    Examples
    --------
    >>> default_args()
    ()

    >>> default_args((1, 2))
    (1, 2)
    """

    if args is None:
        args = ()
    return args


def default_kwargs(
                    kwargs: Optional[dict] = None,
                    ) -> dict:
    """
    Returns default keyword arguments if None are provided.

    Parameters
    ----------
    kwargs : dict, optional
        The input keyword arguments. Default is {}.

    Returns
    -------
    dict
        The default or provided keyword arguments.

    Examples
    --------
    >>> default_kwargs()
    {}

    >>> default_kwargs({"key": "value"})
    {"key": "value"}
    """

    if kwargs is None:
        kwargs = {}
    return kwargs


def default_args_kwargs(
                        args: Optional[tuple] = None,
                        kwargs: Optional[dict] = None,
                        ) -> Tuple[tuple, dict]:
    """
    Returns default positional and keyword arguments if None are provided.

    Parameters
    ----------
    args : tuple, optional
        The input positional arguments. Default is ().
    kwargs : dict, optional
        The input keyword arguments. Default is {}.

    Returns
    -------
    Tuple[tuple, dict]
        The default or provided positional and keyword arguments.

    Examples
    --------
    >>> default_args_kwargs()
    ((), {})

    >>> default_args_kwargs((1, 2), {"key": "value"})
    ((1, 2), {"key": "value"})
    """

    return default_args(args=args), default_kwargs(kwargs=kwargs)


def default_reset(
                    reset: Optional[bool] = None,
                ):
    """
    Returns default reset value if None is provided.

    Parameters
    ----------
    reset : bool, optional
        The input reset value. Default is False.

    Returns
    -------
    bool
        The default or provided reset value.

    Examples
    --------
    >>> default_reset()
    False

    >>> default_reset(True)
    True
    """

    if reset is None:
        reset = False
    return reset


def attr(
            obj: object,
            attr: Optional[str] = None,
            ):
    """
    Returns the specified attribute of the object if provided, otherwise returns the
    object.

    Parameters
    ----------
    obj : object
        The input object.
    attr : str, optional
        The attribute to retrieve.

    Returns
    -------
    object
        The specified attribute of the object or the object itself.

    Examples
    --------
    >>> obj = {"key": "value"}
    >>> _attr(obj, attr="keys")
    <built-in method keys of dict object at 0x...>

    >>> _attr(obj)
    {"key": "value"}
    """

    if attr is not None:
        return getattr(obj, attr)
    else:
        return obj


def fattr(
            obj: object,
            attr: Optional[str] = None,
            args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            ):
    """
    Runs the specified attribute function with arguments.

    Parameters
    ----------
    obj : object
        The input object.
    attr : str, optional
        The attribute function to apply.
    args : tuple, optional
        The positional arguments to pass to the attribute function.
    kwargs : dict, optional
        The keyword arguments to pass to the attribute function.

    Returns
    -------
    object
        The result of running the attribute function or the object itself.

    Examples
    --------
    >>> obj = [1, 2, 3]
    >>> _fattr(obj, attr="append", args=(4,))
    None
    """

    if attr is not None:
        args, kwargs = default_args_kwargs(args=args, kwargs=kwargs)
        return getattr(obj, attr)(*args, **kwargs)
    else:
        return obj


def get_magic_method_names(
                            key: Optional[str] = None,
                            ) -> Union[dict, list,]:
    """
    Retrieves a dictionary of magic method names or a list of names based on the
    provided key.

    Parameters
    ----------
    key : str, optional
        The key to retrieve specific magic method names. If None, the entire dictionary
        is returned.

    Returns
    -------
    Union[dict, list,]
        A dictionary of magic method names or a list of names based on the provided key.

    Examples
    --------
    >>> get_magic_method_names()
    {"noval": ["__str__", "__repr__"],
     "val": ["__add__", "__sub__", "__mul__", "__truediv__", "__mod__", "__pow__"],
     "val_bool": ["__eq__", "__lt__", "__le__", "__gt__", "__ge__"]}

    >>> get_magic_method_names("val")
    ["__add__", "__sub__", "__mul__", "__truediv__", "__mod__", "__pow__"]
    """

    names = {
                "noval": ["__str__", "__repr__"],
                "val": ["__add__", "__sub__", "__mul__",
                        "__truediv__", "__mod__", "__pow__"],
                "val_bool": ["__eq__", "__lt__", "__le__", "__gt__", "__ge__"],
            }

    if key is None:
        return names
    else:
        return names[key]
