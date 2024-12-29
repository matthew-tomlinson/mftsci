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

    try:
        out = f(*args, **kwargs)
    except Exception as e:
        print(e.__repr__())
        return False, None
    else:
        return True, out
