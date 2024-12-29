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

    if args is None:
        args = ()
    return args


def default_kwargs(
                    kwargs: Optional[dict] = None,
                    ) -> dict:

    if kwargs is None:
        kwargs = {}
    return kwargs


def default_args_kwargs(
                        args: Optional[tuple] = None,
                        kwargs: Optional[dict] = None,
                        ) -> Tuple[tuple, dict]:

    return default_args(args=args), default_kwargs(kwargs=kwargs)


def default_reset(
                    reset: Optional[bool] = None,
                ):

    if reset is None:
        reset = False
    return reset


def attr(
            obj: object,
            attr: Optional[str] = None,
            ):

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

    if attr is not None:
        args, kwargs = default_args_kwargs(args=args, kwargs=kwargs)
        return getattr(obj, attr)(*args, **kwargs)
    else:
        return obj


def get_magic_method_names(
                            key: Optional[str] = None,
                            ) -> Union[dict, list,]:

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
