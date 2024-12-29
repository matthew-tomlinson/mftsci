"""
Module for exception utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Callable, Tuple

# External packages


# Internal packages


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def try_f(
            f: Callable,
            *args, **kwargs
        ) -> Tuple[bool, object]:
    """
    Attempts to call a function with provided arguments and returns the success status
    and output.

    Parameters
    ----------
    f : Callable
        The function to be called.
    *args : tuple
        The positional arguments to pass to the function.
    **kwargs : dict
        The keyword arguments to pass to the function.

    Returns
    -------
    Tuple[bool, object]
        A tuple containing a boolean indicating success or failure, and the function
        output.

    Examples
    --------
    >>> def sample_function(x): return x * 2
    >>> try_f(sample_function, 3)
    (True, 6)

    >>> try_f(sample_function, 'a')
    (True, 'aa')

    >>> def raise_exception(x): raise ValueError("An error occurred.")
    >>> try_f(raise_exception, 3)
    An error occurred.
    (False, None)
    """

    try:
        out = f(*args, **kwargs)
    except Exception as e:
        print(e.__repr__())
        return False, None
    else:
        return True, out
