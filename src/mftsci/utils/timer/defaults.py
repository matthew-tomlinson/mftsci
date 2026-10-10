"""
Defaults module for timer utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Literal


# External packages


# Internal packages
from mftsci.utils.meta import set_default


# =====================================
# DEFAULTS
# =====================================


def reset(
        reset: Optional[bool] = None,
        ) -> bool:

    return set_default(val=reset, default=False)


def reset_start(
        reset_start: Optional[bool] = None,
        ) -> bool:

    return set_default(val=reset_start, default=False)


def val(
        val: Optional[Literal["utc", "cpu"]] = None,
        ) -> Union[Literal["utc", "cpu"]]:

    return set_default(val=val, default="cpu")


def print_timer(
        print_timer: Optional[bool] = None,
        ) -> bool:

    return set_default(val=print_timer, default=True)


def print_tab(
        print_tab: Optional[str] = None,
        ) -> str:

    return set_default(val=print_tab, default="")


def print_head(
        print_head: Optional[str] = None,
        ) -> str:

    return set_default(val=print_head, default="")


def write_timer(
        write_timer: Optional[bool] = None,
        ) -> bool:

    return set_default(val=write_timer, default=False)


def write_dir(
        write_dir: Optional[str] = None,
        ) -> str:

    return set_default(val=write_dir, default="./")


def write_fn(
        write_fn: Optional[str] = None,
        ) -> str:

    return set_default(val=write_fn, default="timer.txt")


def pref(
        pref: Optional[str] = None,
        ) -> str:

    return set_default(val=pref, default="")


def ret_Timer(
        ret_Timer: Optional[bool] = None,
        ) -> bool:

    return set_default(val=ret_Timer, default=False)
