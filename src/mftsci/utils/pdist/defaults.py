"""
Defaults module for scipy distribution utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional,  Literal


# External packages


# Internal packages
from mftsci.utils.meta import set_default


# =====================================
# DEFAULTS
# =====================================


def opt(
        opt: Optional[bool] = None,
        ) -> bool:

    return set_default(val=opt, default=False)


def dist(
        dist: Optional[Literal[
                                "norm", "t", "nct",
                                "expon", "genpareto", "genextreme",
                                "gamma", "beta",
                                ]] = None,
        ) -> Literal[
                        "norm", "t", "nct",
                        "expon", "genpareto", "genextreme",
                        "gamma", "beta",
                        ]:

    return set_default(val=dist, default="norm")


def nan_rm(
        nan_rm: Optional[bool] = None,
        ) -> bool:

    return set_default(val=nan_rm, default=True)


def moment_order(
        moment_order: Optional[bool] = None,
        ) -> bool:

    return set_default(val=moment_order, default=False)


def as_array(
        as_array: Optional[bool] = None,
        ) -> bool:

    return set_default(val=as_array, default=True)


def dist_only(
        dist_only: Optional[bool] = None,
        ) -> bool:

    return set_default(val=dist_only, default=True)


def add_None(
        add_None: Optional[bool] = None,
        ) -> bool:

    return set_default(val=add_None, default=False)


def specific(
        specific: Optional[bool] = None,
        ) -> bool:

    return set_default(val=specific, default=True)
