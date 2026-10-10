
"""
Functions module for sklearn.neighbors.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import numpy as np
import pandas as pd
import sklearn


# Internal packages
from ...meta import set_default_kwargs
from .. import preprocessing, pipeline
from ..neighbors import defaults


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================

def make_kde_pipeline(
        data: Optional[np.typing.ArrayLike] = None,
        kde_kwargs: Optional[dict] = None,
        scaler: Optional[Union[Callable, str]] = None,
        ):

    kde_kwargs = set_default_kwargs(
        kwargs=kde_kwargs,
        dkwargs={"kernel": "epanechnikov", "bandwidth": "silverman"},
        )

    if scaler is None:
        scaler = preprocessing.StandardScaler

    if not scaler:
        _scaler = []
    else:
        if isinstance(scaler, str):
            scaler = getattr(sklearn.preprocessing, scaler)
        if callable(scaler):
            scaler = scaler()
        _scaler = [("scale", scaler)]

    kde_pipeline = pipeline.Pipeline([
                    ("reshape", preprocessing.Reshepe1Dto2D()),
                    *_scaler,
                    ("kde", sklearn.neighbors.KernelDensity(**kde_kwargs))
                    ])

    if data is not None:
        kde_pipeline = kde_pipeline.fit(data)

    return kde_pipeline


def calc_density(
        data: Union[np.ndarray, pd.DataFrame, pd.Series,],
        kde_kwargs: Optional[dict] = None,
        scaler: Optional[Union[Callable, str]] = None,
        ret_pipeline: Optional[bool] = None,
        ):

    ret_pipeline = defaults.set_default(ret_pipeline, True)

    kde_pipeline = make_kde_pipeline(kde_kwargs=kde_kwargs, scaler=scaler, data=data)
    density = np.exp(kde_pipeline.score_samples(data))

    if ret_pipeline:
        return density, kde_pipeline
    else:
        return density,
