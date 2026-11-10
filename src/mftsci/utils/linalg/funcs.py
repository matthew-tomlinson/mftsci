"""
Functions module for linalg utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import (     # noqa: F401
                        Optional, Union, Literal,
                    )

# External packages
import numpy as np
import pandas as pd
import scipy


# Internal packages
from mftsci.utils.pd import inv  # noqa: F401


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def min_invertible_submatrix(
        A,
        idx=None,
        ):

    if idx is None:
        return A

    if isinstance(A, pd.DataFrame):
        index = A.index
        columns = A.columns
        try:
            A_prime = A.loc[idx]
            if isinstance(A_prime, pd.Series):
                A_prime = A_prime.to_frame()
            A_prime = A_prime.values
        except:     # noqa: E722
            A_prime = None
        A = A.values
    else:
        index = None
        columns = None
        A_prime = None

    if A_prime is None:
        idx = np.atleast_1d(idx)
        A_prime = A[idx]

    if A_prime.shape[0] == 0:
        i_idx = []
        i_col = []
    else:
        i_col = np.where(np.any(A_prime.astype(bool), axis=0))[0]
        B = A[:, i_col].astype(bool)
        Q, R, piv = scipy.linalg.qr(B.T, pivoting=True)
        rank = np.linalg.matrix_rank(B)
        i_idx = piv[:rank]

    A = A[i_idx, :][:, i_col]

    if index is not None:
        A = pd.DataFrame(A, index=index[i_idx], columns=columns[i_col])

    return A
