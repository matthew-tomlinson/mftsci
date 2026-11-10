"""
Defaults module for meta utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Callable

# External packages


# Internal packages
from mftsci.utils.meta.core import return_x, set_default, astype    # noqa: F401


# =====================================
# DEFAULTS
# =====================================


def int(
            val: Optional[int] = None,
            ) -> int:
    """
    Returns the default integer if None is given.

    Parameters
    ----------
    val : int, optional
        The value to check. If it is None, the default value is returned. Defaults to
        None.

    Returns
    -------
    int
        The value if it is not None, otherwise the default value.

    Examples
    --------
    >>> int()
    0

    >>> int(1)
    1
    """
    return set_default(val=val, default=0)


def float(
            val: Optional[float] = None,
            ) -> float:
    """
    Returns the default float if None is given.

    Parameters
    ----------
    val : float, optional
        The value to check. If it is None, the default value is returned. Defaults to
        None.

    Returns
    -------
    float
        The value if it is not None, otherwise the default value.

    Examples
    --------
    >>> float()
    0.0

    >>> float(1.0)
    1.0
    """

    return set_default(val=val, default=0.0)


def call(
            val: Optional[Callable] = None,
            ) -> Callable:
    """
    Returns the default Callable if None is given.

    Parameters
    ----------
    val : Callable, optional
        The value to check. If it is None, the default value is returned. Defaults to
        None.

    Returns
    -------
    Callable
        The value if it is not None, otherwise the default value.

    Examples
    --------
    >>> Callable()(1)
    1

    >>> Callable(lambda x: 2*x)(1)
    2
    """

    return set_default(val=val, default=return_x)


def reset(
                reset: Optional[bool] = None,
            ) -> bool:
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
    >>> reset(True)
    True

    >>> reset(None)
    False
    """

    return set_default(val=reset, default=False)


def _no_append(
                _no_append: Optional[bool] = None,
                ) -> bool:

    return set_default(val=_no_append, default=False)
