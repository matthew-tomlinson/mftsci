"""
Core module for point processes in continuous time.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Union, Callable

# External packages
import numpy as np


# Internal packages


# =====================================
# FUNCTIONS
# =====================================


def gen_lambda_const(
                        a_lambda: Union[float, int,],
                        ) -> Callable:

    def f(t=None):
        return a_lambda * np.ones_like(t)

    return f
