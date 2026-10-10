
"""
Functions module for utils.preprocessing.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional

# External packages
import numpy as np
import sklearn
import pandas as pd
from sklearn.utils.validation import check_is_fitted, check_array

from sklearn.base import (
                            BaseEstimator as _BaseEstimator,
                            TransformerMixin as _TransformerMixin,
                            )

from sklearn.preprocessing import (
                            StandardScaler as _StandardScaler,
                            MinMaxScaler as _MinMaxScaler,
                            RobustScaler as _RobustScaler,
                            MaxAbsScaler as _MaxAbsScaler,
                            )

from sklearn.preprocessing import (
                            PowerTransformer as _PowerTransformer,
                            QuantileTransformer as _QuantileTransformer,
                            )
from sklearn.decomposition import (
                            PCA as _PCA
                            )

# Internal packages
from . import defaults   # noqa: F401
from ..base import BaseLogDetMixin, TransformerMixin


# =====================================
# CLASSES
# =====================================


class Reshepe1Dto2D(_BaseEstimator, _TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                return X.reshape(-1, 1)
            else:
                return X
        else:
            return X


class Reshepe2Dto1D(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, np.ndarray):
            if len(X.shape) == 2:
                return X[:, 0]
            else:
                return X
        else:
            return X


class LinearLogDetMixin(BaseLogDetMixin):
    """
    Adds `calc_logdet(X)` to a transformer.
    Must be combined with a fitted sklearn transformer.
    """

    def calc_scale(
            self,
            X: Optional[np.typing.ArrayLike] = None,
            ) -> np.typing.ArrayLike:
        """
        Returns scale
        """
        check_is_fitted(self)
        scale = np.asarray(self.scale_, dtype=float)
        return scale

    def calc_logdet(
            self,
            X: Optional[np.typing.ArrayLike] = None,
            ) -> float:
        """
        Return per-sample log|det J_T(x)| for y = T(x) at inputs X.
        Default raises; subclasses must implement or call super() to fall back.
        """
        check_is_fitted(self)
        logdet = -np.sum(np.log(self.calc_scale()))
        if X is not None:
            if False:
                X = check_array(X, accept_sparse=False)
                logdet = np.full(X.shape[0], logdet)

        return logdet


class NonLinearLogDetMixin(BaseLogDetMixin):
    """
    Adds `calc_logdet(X)` to a transformer.
    Must be combined with a fitted sklearn transformer.
    """
    def calc_logdet(
            self,
            X: np.typing.ArrayLike,
            ) -> np.typing.ArrayLike:
        """
        Return per-sample log|det J_T(x)| for y = T(x) at inputs X.
        Default raises; subclasses must implement or call super() to fall back.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement calc_logdet()."
        )


# ----------------- Linear / affine scalers (constant logdet) -----------------

class _UnitaryScaler(_StandardScaler):

    def fit(self, X, y=None):
        super().fit(X=X, y=y)
        self.mean_ *= 0.0
        self.scale_ /= self.scale_
        return self


class UnitaryScaler(LinearLogDetMixin, TransformerMixin, _UnitaryScaler):
    pass


class StandardScaler(LinearLogDetMixin, TransformerMixin, _StandardScaler):
    # y = (x - mean) / scale => dy/dx = diag(1/scale)
    pass
    # y = (x - mean) / scale => dy/dx = diag(1/scale)
    pass


class MinMaxScaler(LinearLogDetMixin, TransformerMixin, _MinMaxScaler):
    # y = (x - min) * scale + a => dy/dx = diag(scale)
    def calc_scale(
            self,
            X: Optional[np.typing.ArrayLike] = None,
            ) -> np.typing.ArrayLike:
        return 1/super().calc_scale(X=X)


class RobustScaler(LinearLogDetMixin, TransformerMixin, _RobustScaler,):
    pass


class MaxAbsScaler(LinearLogDetMixin, TransformerMixin, _MaxAbsScaler):
    # y = x / max_abs_ => dy/dx = diag(1/max_abs_)
    pass

# ----------------- Feature-wise nonlinear scalers (sample-dependent) -----------------


class PowerTransformer(
                NonLinearLogDetMixin,
                TransformerMixin,
                _PowerTransformer,
                ):
    """
    Exact feature-wise Jacobian for Box-Cox and Yeo-Johnson.
    """
    def calc_logdet(
            self,
            X: np.typing.ArrayLike,
            ) -> np.typing.ArrayLike:
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        lam = np.asarray(self.lambdas_, dtype=float)  # (d,)
        n, d = X.shape
        out = np.zeros(n, dtype=float)

        if self.method == "box-cox":
            if np.any(X <= 0):
                raise ValueError("Box-Cox requires strictly positive inputs.")
            # dy/dx = x^{λ-1} (λ≠0), 1/x (λ=0)
            for j in range(d):
                lj = lam[j]
                xj = X[:, j]
                if lj == 0.0:
                    out += -_safe_log(xj)
                else:
                    out += (lj - 1.0) * _safe_log(xj)

        elif self.method == "yeo-johnson":
            # pos: dy/dx = (x+1)^{λ-1} (λ≠0) or 1/(x+1) (λ=0)
            # neg: dy/dx = (1-x)^{1-λ} (λ≠2) or 1/(1-x) (λ=2)
            for j in range(d):
                lj = lam[j]
                xj = X[:, j]
                pos = xj >= 0
                neg = ~pos

                if lj == 0.0:
                    out[pos] += -_safe_log(1.0 + xj[pos])
                    out[neg] += _safe_log(1.0 - xj[neg])
                elif lj == 2.0:
                    out[pos] += _safe_log(1.0 + xj[pos])
                    out[neg] += -_safe_log(1.0 - xj[neg])
                else:
                    out[pos] += (lj - 1.0) * _safe_log(1.0 + xj[pos])
                    out[neg] += (1.0 - lj) * _safe_log(1.0 - xj[neg])
        else:
            raise NotImplementedError(f"Unknown method: {self.method}")

        return out


class QuantileTransformer(
            NonLinearLogDetMixin,
            TransformerMixin,
            _QuantileTransformer,
            ):
    """
    Exact per-sample Jacobian using stored quantiles_ and references_.
    """
    def calc_logdet(
            self,
            X: np.typing.ArrayLike,
            ) -> np.typing.ArrayLike:
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        q = np.asarray(self.quantiles_)     # (m, d)
        r = np.asarray(self.references_)    # (m,)
        n, d = X.shape

        # We need z = transform(X) for the normal case (phi(z))
        Z = super().transform(X)
        if hasattr(Z, "toarray"):
            Z = Z.toarray()

        out = np.zeros(n, dtype=float)
        for j in range(d):
            slope_u = _quantile_slope_1d(q[:, j], r, X[:, j])  # du/dx
            if self.output_distribution == "uniform":
                out += _safe_log(slope_u)
            elif self.output_distribution == "normal":
                zj = Z[:, j]
                phi = _normal_pdf(zj)
                out += _safe_log(slope_u) - _safe_log(phi)
            else:
                raise NotImplementedError(self.output_distribution)
        return out


# ----------------- PCA (square only) -----------------


class PCA(NonLinearLogDetMixin, TransformerMixin, _PCA):
    def calc_logdet(
            self,
            X: np.typing.ArrayLike,
            ) -> np.typing.ArrayLike:
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        d_in = self.n_features_in_
        d_out = self.components_.shape[0]
        if d_out != d_in:
            raise NotImplementedError(
                    "PCA with dimensionality change is non-invertible."
                    )
        if getattr(self, "whiten", False):
            ev = np.asarray(self.explained_variance_, dtype=float)
            return np.full(X.shape[0], -0.5 * np.sum(np.log(ev)))
        else:
            return np.zeros(X.shape[0])


# =====================================
# FUNCTIONS
# =====================================


def _safe_log(x, eps=1e-300):
    return np.log(np.clip(x, eps, None))


def _normal_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _quantile_slope_1d(q, r, x):
    """
    Piecewise-linear slope du/dx for QuantileTransformer (uniform stage).
    q: (m,) quantile values; r: (m,) refs in [0,1]; x: (n,)
    """
    idx = np.searchsorted(q, x, side="right") - 1
    idx = np.clip(idx, 0, len(r) - 2)
    dx = q[idx + 1] - q[idx]
    dr = r[idx + 1] - r[idx]
    slope = dr / np.where(dx == 0.0, np.finfo(float).tiny, dx)
    return slope
