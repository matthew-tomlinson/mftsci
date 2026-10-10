"""
Defaults module for pandas utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional,  Union


# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.arr import iterarr


# =====================================
# DEFAULTS
# =====================================

def srs(
        srs: Optional[pd.core.series.Series] = None,
        ) -> pd.core.series.Series:

    return set_default(val=srs, default=pd.Series())


def df(
        df: Optional[pd.core.frame.DataFrame] = None,
        ) -> pd.core.frame.DataFrame:

    return set_default(val=df, default=pd.DataFrame())


def index(
        index: Optional[pd.core.indexes.base.Index] = None,
        ) -> pd.core.indexes.base.Index:

    return set_default(val=index, default=srs().index)


def name(
        name: Optional[Union[str, int,]] = None,
        ) -> Union[str, int,]:

    return set_default(val=name, default=0)


def reindex(
        reindex: Optional[bool] = None,
        ) -> bool:

    return set_default(val=reindex, default=True)


def incl_index(
        incl_index: Optional[bool] = None,
        ) -> bool:

    return set_default(val=incl_index, default=False)


def incl_index_ret(
        incl_index_ret: Optional[bool] = None,
        ) -> bool:

    return set_default(val=incl_index_ret, default=False)


def cv(
        cv: Optional[dict] = None,
        ) -> dict:

    return set_default(val=cv, default={})


def conserve_index(
        conserve_index: Optional[bool] = None,
        ) -> bool:

    return set_default(val=conserve_index, default=True)


def group_source(
        group_source: Optional[int] = None,
        ) -> int:

    return set_default(val=group_source, default=0)


def group_destination(
        group_destination: Optional[int] = None,
        ) -> int:

    return set_default(val=group_destination, default=0)


def overwrite_level_names(
        overwrite_level_names: Optional[bool] = None,
        ) -> bool:

    return set_default(val=overwrite_level_names, default=False)


def appear_order(
        appear_order: Optional[bool] = None,
        ) -> bool:

    return set_default(val=appear_order, default=True)


def col_levels(
        col_levels: Optional[Union[np.ndarray, list, int,]] = None,
        ) -> np.ndarray:

    return iterarr(set_default(val=col_levels, default=[]))
