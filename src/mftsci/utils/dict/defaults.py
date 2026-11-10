"""
Defaults module for dictionary utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable


# External packages
import numpy as np

# Internal packages
from ..meta import set_default, return_x
from ..arr import iterarr


# =====================================
# DEFAULTS
# =====================================


def d(
        d: Optional[dict] = None,
        ) -> dict:

    return set_default(val=d, default={})


def array(
        array: Optional[bool] = None,
        ) -> bool:

    return set_default(val=array, default=True)


def ret_dict(
        ret_dict: Optional[bool] = None,
        ) -> bool:

    return set_default(val=ret_dict, default=True)


def None_rm(
        None_rm: Optional[bool] = None,
        ) -> bool:

    return set_default(val=None_rm, default=True)


def subdictkeys(
        subdictkeys: Optional[Union[np.ndarray, list, object,]] = None,
        ) -> np.ndarray:

    return iterarr(set_default(val=subdictkeys, default=[]))


def get_store(
        get_store: Optional[Callable] = None,
        ) -> Callable:

    return set_default(val=get_store, default=return_x)


def store_type(
        store_type: Optional[type] = None,
        ) -> type:

    return set_default(val=store_type, default=dict)


def is_node(
        is_node: Optional[type] = None,
        ) -> type:

    if is_node is None:
        def is_node(node):
            return True

    return is_node
