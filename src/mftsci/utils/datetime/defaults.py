"""
Defaults module for datetime utilities.
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


def year(
            year: Optional[int] = None,
            ) -> int:

    return set_default(val=year, default=1970)


def month(
            month: Optional[int] = None,
            ) -> int:

    return set_default(val=month, default=1)


def day(
            day: Optional[int] = None,
            ) -> int:

    return set_default(val=day, default=1, )


def day_0(
            day_0: Optional[Union[int, str,]] = None,
            ) -> int:

    return set_default(val=day_0, default=1, )


def season_kind(
                    kind: Optional[Literal["all", "ast", "met", "quarter"]] = None,
                    ) -> Literal["all", "ast", "met", "quarter"]:

    return set_default(val=kind, default="ast")


def leapday(
                leapday: Optional[Literal[
                                            "unique", "splice",
                                            "bmerge", "fmerge",
                                            ]] = None,
                ) -> bool:

    return set_default(val=leapday, default="unique")


def iterarr(
            iterarr: Optional[int] = None,
            ) -> int:

    return set_default(val=iterarr, default=False)
