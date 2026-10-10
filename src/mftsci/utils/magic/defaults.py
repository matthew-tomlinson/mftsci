"""
Defaults module for magic method utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional


# External packages


# Internal packages
from mftsci.utils.meta.core import set_default


# =====================================
# DEFAULTS
# =====================================


def ret_class(
        ret_class: Optional[bool] = None,
        ) -> bool:

    return set_default(val=ret_class, default=False)


def ret_class_val(
        ret_class_val: Optional[bool] = None,
        ) -> bool:

    return set_default(val=ret_class_val, default=True)


def ret_class_val_bool(
        ret_class_val_bool: Optional[bool] = None,
        ) -> bool:

    return set_default(val=ret_class_val_bool, default=False)
