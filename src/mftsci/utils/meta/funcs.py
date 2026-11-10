"""
Functions module for meta utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import (
                    Optional, Union, Tuple, Callable,
                    TypeVar, Generic, overload, Type,
                    )
from typing_extensions import ParamSpec
from functools import wraps
from inspect import ismethod, signature
import shutil
import os
import importlib

# External packages
import numpy as np
import pandas as pd


# Internal packages
from mftsci.utils.meta import defaults  # noqa: F401, F403
from mftsci.utils.meta.core import (    # noqa: F401
                            return_args_, return_args, return_kwargs,
                            return_x, set_default, astype,
                            )


# =====================================
# TYPING
# =====================================

_T = TypeVar("T")
_R = TypeVar("R")
_P = ParamSpec("P")


# =====================================
# CLASSES
# =====================================

class hybridmethod(Generic[_T, _P, _R]):
    def __init__(self, func: Callable[..., _R]):
        self.func = func

    @overload
    def __get__(
            self,
            obj: None,
            cls: Type[_T]
            ) -> Callable[_P, _R]: ...

    @overload
    def __get__(
            self,
            obj: _T,
            cls: Type[_T]
            ) -> Callable[_P, _R]: ...

    def __get__(self, obj, cls=None):
        if obj is None:
            def _func(*args, **kwargs):
                return self.func(cls, *args, **kwargs)
        else:
            def _func(*args, **kwargs):
                return self.func(obj, *args, **kwargs)

        return _func


# =====================================
# FUNCTIONS
# =====================================

def is_importable(
            module,
            name,
            **kwargs
            ):

    try:
        _module = importlib.import_module(module)
        found = hasattr(_module, name)
    except Exception:     # noqa: E722
        found = False

    return found


def is_modified(cls):

    try:
        module = importlib.import_module(cls.__module__)
        original = getattr(module, cls.__name__)
        return cls is not original
    except Exception:
        return True


def init(
        obj,
        *args,
        **kwargs,
        ):

    if isinstance(obj, type):
        return obj(*args, **kwargs)
    else:
        return type(obj)(*args, **kwargs)


def require_optional(
        module: str,
        attr: Optional[str] = None,
        install: Optional[str] = None,
        ):

    try:
        mod = importlib.import_module(module)
    except ImportError:
        strs = [f"'{module}' is required for this feature."]
        if install is not None:
            strs.append(f"Install with:\n`pip install {install}`")
        raise ImportError(" ".join(strs))

    if attr is None:
        return mod
    elif isinstance(attr, str):
        return getattr(mod, attr)
    else:
        return tuple([getattr(mod, a) for a in attr])


def clear_cache_0(
                path: Optional[str] = None,
                ):
    """
    Removes generic `__pycache__` .

    Parameters
    ----------
    path : str
        Parent directory.
    """
    if path is None:
        path = "./"
    try:
        for all in os.listdir(path):
            if os.path.isdir(path + all):
                if all == '__pycache__':
                    shutil.rmtree(path + all, ignore_errors=False)
    except Exception as e:
        print(e)
        pass


def clear_cache(
                path: Optional[str] = None,
                ):
    """
    Removes generic `__pycache__` .

    Parameters
    ----------
    path : str
        Parent directory.
    """
    if path is None:
        path = "./"

    for walk in os.walk(path):
        if walk[0].endswith("\\__pycache__"):
            shutil.rmtree(walk[0], ignore_errors=False)


def smap(
            f: Callable,
            ) -> object:

    return f()


def pack_args_kwargs(
                        *args, **kwargs,
                        ) -> Tuple[tuple, dict]:
    """
    Packs positional and keyword arguments into a tuple and a dictionary.

    Parameters
    ----------
    *args
        Positional arguments.
    **kwargs
        Keyword arguments.

    Returns
    -------
    Tuple[tuple, dict]
        A tuple containing the positional arguments and a dictionary containing the
        keyword arguments.

    Examples
    --------
    >>> pack_args_kwargs(1, 2, a=3, b=4)
    ((1, 2), {'a': 3, 'b': 4})
    """

    return args, kwargs


def pack_args_kwargs_kw(
                            *args, **kwargs,
                            ) -> dict:
    """
    Packs positional and keyword arguments into a dictionary.

    Parameters
    ----------
    *args
        Positional arguments.
    **kwargs
        Keyword arguments.

    Returns
    -------
    dict
        A dictionary containing the positional arguments under the key "args" and the
        keyword arguments under the key "kwargs".

    Examples
    --------
    >>> pack_args_kwargs_kw(1, 2, a=3, b=4)
    {'args': (1, 2), 'kwargs': {'a': 3, 'b': 4}}
    """

    return {"args": args, "kwargs": kwargs}


def default_args(
                    args: Optional[tuple] = None,
                    ) -> tuple:
    """
    Returns default positional arguments if None are provided.

    Parameters
    ----------
    args : tuple, optional
        The tuple of positional arguments. Defaults to None.

    Returns
    -------
    tuple
        The default or provided positional arguments.

    Examples
    --------
    >>> default_args((1, 2))
    (1, 2)

    >>> default_args(None)
    ()
    """

    return set_default(default=(), val=args)


def default_kwargs(
                    kwargs: Optional[dict] = None,
                    **xkwargs,
                    ) -> dict:
    """
    Returns default keyword arguments if None are provided.

    Parameters
    ----------
    kwargs : dict, optional
        The dictionary of keyword arguments. Defaults to None.
    **xkwargs
        Additional keyword arguments to update the dictionary with.

    Returns
    -------
    dict
        The default or updated dictionary of keyword arguments.

    Examples
    --------
    >>> default_kwargs({'a': 1}, b=2)
    {'a': 1, 'b': 2}

    >>> default_kwargs(None, a=1)
    {'a': 1}
    """

    kwargs = set_default(default={}, val=kwargs)
    kwargs.update(xkwargs)

    return kwargs


def set_default_kwargs(
                    kwargs: Optional[dict] = None,
                    dkwargs: Optional[dict] = None,
                    ) -> dict:

    kwargs = set_default(default={}, val=kwargs)
    dkwargs = set_default(default={}, val=dkwargs)
    for k in dkwargs:
        kwargs.setdefault(k, dkwargs[k])

    return kwargs


def default_args_kwargs(
                        args: Optional[tuple] = None,
                        kwargs: Optional[dict] = None,
                        **xkwargs,
                        ) -> Tuple[tuple, dict]:
    """
    Returns default positional and keyword arguments if None are provided.

    Parameters
    ----------
    args : tuple, optional
        The tuple of positional arguments. Defaults to None.
    kwargs : dict, optional
        The dictionary of keyword arguments. Defaults to None.
    **xkwargs
        Additional keyword arguments to update the dictionary with.

    Returns
    -------
    Tuple[tuple, dict]
        The tuple of positional arguments and the updated dictionary of keyword
        arguments.

    Examples
    --------
    >>> default_args_kwargs((1,), {'a': 2}, b=3)
    ((1,), {'a': 2, 'b': 3})

    >>> default_args_kwargs(None, None, a=1)
    ((), {'a': 1})
    """

    return default_args(args=args), default_kwargs(kwargs=kwargs, **xkwargs)


def get_ovarnames(
                    func: Callable,
                    ) -> tuple:
    """
    Retrieves the variable names of a function.

    Parameters
    ----------
    func : Callable
        The function to retrieve variable names from.

    Returns
    -------
    tuple
        A tuple containing the variable names of the function.

    Examples
    --------
    >>> def example_func(a, b, c=3): pass
    >>> get_ovarnames(example_func)
    ('a', 'b', 'c')
    """

    if "__ovarnames__" in dir(func):
        ovarnames = func.__ovarnames__
    else:
        ovarnames = tuple(signature(func).parameters.keys())

    if ismethod(func):
        return ovarnames[1:]
    else:
        return ovarnames


def get_okdefaults(
                    func: Callable,
                    ) -> dict:
    """
    Retrieves the keyword-only argument defaults of a function.

    Parameters
    ----------
    func : Callable
        The function to retrieve keyword-only argument defaults from.

    Returns
    -------
    dict
        A dictionary containing the keyword-only argument defaults of the function.

    Examples
    --------
    >>> def example_func(a, b, *, c=3, d=4): pass
    >>> get_okdefaults(example_func)
    {'c': 3, 'd': 4}
    """

    if "__okwdefaults__" in dir(func):
        okwdefaults = func.__okwdefaults__
    else:
        okwdefaults = func.__kwdefaults__

    return okwdefaults


def merge_args_kwargs(
                        _func: Optional[Callable] = None,
                        args: Optional[tuple] = None,
                        kwargs: Optional[dict] = None,
                        **xkwargs,
                        ) -> dict:
    """
    Merges positional and keyword arguments with the parameters of a function.

    Parameters
    ----------
    _func : Callable, optional
        The function whose parameters are to be merged with the arguments. Defaults to
        None.
    args : tuple, optional
        The tuple of positional arguments. Defaults to None.
    kwargs : dict, optional
        The dictionary of keyword arguments. Defaults to None.
    **xkwargs
        Additional keyword arguments.

    Returns
    -------
    dict
        The merged dictionary of function parameters and arguments.

    Examples
    --------
    >>> def example_func(a, b): pass
    >>> merge_args_kwargs(_func=example_func, args=(1,), kwargs={'b': 2})
    {'a': 1, 'b': 2}
    """

    args, kwargs = default_args_kwargs(args, kwargs, **xkwargs)

    if _func is not None:
        N_args = len(args)

        pos_kwargs = get_ovarnames(_func)[:N_args]

        for k, kw in enumerate(pos_kwargs):
            kwargs[kw] = args[k]

    return kwargs


def set_defaults(
                    _defaults: Optional[dict] = None,
                    _defaults_call: Optional[dict] = None,
                    _no_append: Optional[bool] = None,
                    _func: Optional[Callable] = None,
                    args: Optional[tuple] = None,
                    kwargs: Optional[dict] = None,
                    **xkwargs,
                    ) -> dict:
    """
    Sets default values for a dictionary of keyword arguments, with options to merge
    with function parameters and avoid appending new keys.

    Parameters
    ----------
    _defaults : dict, optional
        The default values for the keyword arguments. Defaults to None.
    _defaults_call : dict, optional
        The default values for the keyword arguments that are callable. Defaults to
        None.
    _no_append : bool, optional
        Whether to avoid appending new keys that are not in the defaults. Defaults to
        None.
    _func : Callable, optional
        The function whose parameters are to be merged with the arguments. Defaults to
        None.
    args : tuple, optional
        The tuple of positional arguments. Defaults to None.
    kwargs : dict, optional
        The dictionary of keyword arguments. Defaults to None.
    **xkwargs
        Additional keyword arguments.

    Returns
    -------
    dict
        The updated dictionary of keyword arguments with defaults applied.

    Examples
    --------
    >>> set_defaults(
                        _defaults={'a': 1},
                        _defaults_call={'b': lambda: 2},
                        kwargs={'c': 3},
                        )
    {'a': 1, 'b': 2, 'c': 3}
    """

    _no_append = defaults._no_append(_no_append)

    _defaults = set_default(default={}, val=_defaults)
    _defaults_call = set_default(default={}, val=_defaults_call)

    kwargs = merge_args_kwargs(_func=_func, args=args, kwargs=kwargs, **xkwargs)

    s_dflt = set(_defaults)

    s_dflt_c = set(_defaults_call)
    s_dflt_nc = s_dflt - s_dflt_c

    if _no_append:
        s_kw = set(kwargs)
        s_dflt_nc.intersection_update(s_kw)
        s_dflt_c.intersection_update(s_kw)

    s_dflt_iter = s_dflt_c.union(s_dflt_nc)

    for kw in s_dflt_iter:
        kwargs[kw] = set_default(
                                    val=kwargs.get(kw, None),
                                    default=_defaults.get(kw, None),
                                    default_call=_defaults_call.get(kw, None),
                                    )

    return kwargs


def attr(
            obj: object,
            attr: Optional[str] = None,
            ):
    """
    Retrieves an attribute from an object.

    Parameters
    ----------
    obj : object
        The object to retrieve the attribute from.
    attr : str, optional
        The name of the attribute to retrieve. If None, the object itself is returned.
        Defaults to None.

    Returns
    -------
    object
        The attribute of the object, or the object itself if the attribute is None.

    Examples
    --------
    >>> class Example:
    ...     def __init__(self):
    ...         self.value = 42
    ...
    >>> example = Example()
    >>> attr(example, 'value')
    42

    >>> attr(example)
    <__main__.Example object at 0x...>
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
    Retrieves an attribute from an object and calls it with the provided arguments.

    Parameters
    ----------
    obj : object
        The object to retrieve the attribute from.
    attr : str, optional
        The name of the attribute to retrieve and call. If None, the object itself is
        returned. Defaults to None.
    args : tuple, optional
        The tuple of positional arguments to pass to the attribute. Defaults to None.
    kwargs : dict, optional
        The dictionary of keyword arguments to pass to the attribute. Defaults to None.

    Returns
    -------
    object
        The result of calling the attribute with the provided arguments, or the object
        itself if the attribute is None.

    Examples
    --------
    >>> class Example:
    ...     def __init__(self):
    ...         self.value = 42
    ...     def add(self, x):
    ...         return self.value + x
    ...
    >>> example = Example()
    >>> fattr(example, 'add', args=(8,))
    50

    >>> fattr(example)
    <__main__.Example object at 0x...>
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
    Retrieves magic method names based on a key.

    Parameters
    ----------
    key : str, optional
        The key to retrieve the magic method names for. If None, all magic method names
        are returned. Defaults to None.

    Returns
    -------
    Union[dict, list]
        A dictionary of magic method names if key is None, otherwise a list of magic
        method names for the specified key.

    Examples
    --------
    >>> get_magic_method_names()
    {'noval': ['__str__', '__repr__'],
    'val': ['__add__', '__sub__', '__mul__', '__truediv__', '__mod__', '__pow__'],
    'val_bool': ['__eq__', '__lt__', '__le__', '__gt__', '__ge__']}

    >>> get_magic_method_names('val')
    ['__add__', '__sub__', '__mul__', '__truediv__', '__mod__', '__pow__']
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


# -------------------------------------
# DECORATORS
# -------------------------------------

def wraps_ovar(
                func: Optional[Callable] = None,
                ) -> Callable:
    """
    Generates a decorator to wrap a function, preserving its original variable names.

    Parameters
    ----------
    func : Callable
        The function to be wrapped.

    Returns
    -------
    Callable
        The decorator that wraps a function with preserved variable names.

    Examples
    --------
    >>> @wraps_ovar()
    ... def example_func(a, b): pass
    ...
    >>> example_func.__ovarnames__
    ('a', 'b')
    """

    def decorator(_func):

        nonlocal func

        func = set_default(default=_func, val=func)

        @wraps(func)
        def wrapper(
                    *args, **kwargs,
                    ) -> object:
            return _func(*args, **kwargs)
        wrapper.__ovarnames__ = get_ovarnames(func)

        return wrapper

    return decorator


def defaults_kwargs(
                    _defaults: Optional[dict] = None,
                    _defaults_call: Optional[dict] = None,
                    ) -> Callable:
    """
    Generates a decorator to set default keyword arguments for a function.

    Parameters
    ----------
    _defaults : dict, optional
        The default values for the keyword arguments. Defaults to None.
    _defaults_call : dict, optional
        The default values for the keyword arguments that are callable. Defaults to
        None.

    Returns
    -------
    Callable
        The decorator that applies default keyword arguments to a function.

    Examples
    --------
    >>> @defaults_kwargs(_defaults={'a': 1}, _defaults_call={'b': lambda: 2})
    ... def example_func(a, b):
    ...     return a + b
    ...
    >>> example_func()
    3
    """

    def decorator(
                    _func: Callable,
                    ) -> Callable:

        @wraps_ovar(_func)
        def wrapper(
                    *args, **kwargs,
                    ) -> object:

            kwargs = set_defaults(_defaults=_defaults,
                                  _defaults_call=_defaults_call,
                                  _func=_func,
                                  args=args, kwargs=kwargs)

            return _func(**kwargs)

        return wrapper

    return decorator


def metafunc_decorator(
                        metafunc: Optional[Callable] = None,
                        ) -> Callable:
    """
    Generates a decorator to apply a metafunction to the output of a function.

    Parameters
    ----------
    metafunc : Callable, optional
        The metafunction to apply. Defaults to `return_x`.

    Returns
    -------
    Callable
        The decorator that applies the metafunction to the output of a function.

    Examples
    --------
    >>> @metafunc_decorator(lambda x: x * 2)
    ... def example_func(a, b):
    ...     return a + b
    ...
    >>> example_func(1, 2)
    6
    """

    metafunc = set_default(default=return_x, val=metafunc)

    def decorator(
                    func: Optional[Callable] = None,
                    ) -> Callable:

        func = set_default(default=return_x, val=func)

        @wraps_ovar(func)
        def wrapper(
                    *args, **kwargs,
                    ) -> object:

            return metafunc(func(*args, **kwargs))

        return wrapper

    return decorator


def vectorize(
                        *args, **kwargs,
                        ) -> Callable:

    def decorator(func):
        wrapper = np.vectorize(func, *args, **kwargs)
        wrapper.__ovarnames__ = get_ovarnames(func)
        return wrapper

    return decorator


def vectorize_pd(
                        *vargs, **vkwargs,
                        ) -> Callable:

    def decorator(func):

        vfunc = vectorize(*vargs, **vkwargs)(func)

        @wraps_ovar(vfunc)
        def wrapper(
                    *args, **kwargs,
                    ) -> object:
            kwargs = merge_args_kwargs(_func=func, args=args, kwargs=kwargs)
            kw_0 = next(iter(kwargs.keys()), None)
            if isinstance(kwargs.get(kw_0, None), pd.core.series.Series):
                srs = kwargs.pop(kw_0)
                return srs.apply(func, **kwargs)
            elif isinstance(kwargs.get(kw_0, None), pd.core.frame.DataFrame):
                df = kwargs.pop(kw_0)
                return df.map(func, **kwargs)
            else:
                return vfunc(**kwargs)

        return wrapper

    return decorator


def vectorize_pd_alt(
                        *vargs, **vkwargs,
                        ) -> Callable:

    def decorator(func):

        vfunc = vectorize(*vargs, **vkwargs)(func)

        @wraps_ovar(vfunc)
        def wrapper(
                    *args, **kwargs,
                    ) -> object:
            kwargs = merge_args_kwargs(_func=func, args=args, kwargs=kwargs)
            kw_0 = next(iter(kwargs.keys()), None)
            if isinstance(kwargs.get(kw_0, None), pd.core.series.Series):
                srs = kwargs.pop(kw_0)
                return pd.Series(
                                    vfunc(**{**{kw_0: srs.values}, **kwargs}),
                                    index=srs.index,
                                    )
            elif isinstance(kwargs.get(kw_0, None), pd.core.frame.DataFrame):
                df = kwargs.pop(kw_0)
                return pd.DataFrame(
                                    vfunc(**{**{kw_0: df.values}, **kwargs}),
                                    index=df.index, columns=df.columns,
                                    )
            else:
                return vfunc(**kwargs)

        return wrapper

    return decorator
