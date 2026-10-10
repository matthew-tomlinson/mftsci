
"""
Functions module for sklearn.pipeline.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional

# External packages
import numpy as np
from sklearn.pipeline import (
                        Pipeline as _Pipeline
                        )
from sklearn.utils.validation import check_is_fitted

# Internal packages
from . import defaults    # noqa: F401


# =====================================
# CLASSES
# =====================================

class Pipeline(_Pipeline):

    def calc_logdet(
            self,
            X: Optional[np.typing.ArrayLike] = None,
            ) -> float:
        """
        Return per-sample log|det J_T(x)| for y = T(x) at inputs X.
        Default raises; subclasses must implement or call super() to fall back.
        """
        check_is_fitted(self)

        logdet = sum([
                    getattr(s[1], "calc_logdet", _null_calc_logdet)(X)
                    for s in self.steps
                    ])

        return logdet

    def score_samples_logdet(self, X):
        return self.score_samples(X) + self.calc_logdet(X)


# =====================================
# FUNCTIONS
# =====================================

def _null_calc_logdet(
        X: Optional[np.typing.ArrayLike] = None,
        ) -> int:
    return 0
