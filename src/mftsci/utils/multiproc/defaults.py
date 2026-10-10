"""
Defaults module for multiprocessing utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional
import multiprocessing

# External packages


# Internal packages
from mftsci.utils.meta import set_default


# =====================================
# DEFAULTS
# =====================================


def print_mp_time(
        print_mp_time: Optional[bool] = None,
        ) -> bool:

    return set_default(val=print_mp_time, default=True)


def write_mp_time(
        write_mp_time: Optional[bool] = None,
        ) -> bool:

    return set_default(val=write_mp_time, default=False)


def write_mp_fn(
        write_mp_fn: Optional[str] = None,
        ) -> str:

    return set_default(val=write_mp_fn, default="write_mp_timer.txt")


def max_factor(
        max_factor: Optional[int] = None,
        ) -> int:

    return set_default(val=max_factor, default_call=multiprocessing.cpu_count)
