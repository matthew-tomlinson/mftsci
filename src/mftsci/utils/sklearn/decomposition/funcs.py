
"""
Functions module for utils.decomposition.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import numpy as np
import pandas as pd
import scipy
from sklearn.utils.validation import (
                            check_is_fitted, check_array,
                            )
from sklearn.decomposition import PCA as _PCA
from sklearn.base import (
                            BaseEstimator as _BaseEstimator,
                            TransformerMixin as _TransformerMixin,
                            )
# Internal packages
from . import defaults   # noqa: F401
from ..preprocessing import (
                                NonLinearLogDetMixin, TransformerMixin
                                )
from ...meta import set_default_kwargs
from ..base import ClassNamePrefixFeaturesOutMixin

# =====================================
# CLASSES
# =====================================


class _AfflineBasisLogdet(NonLinearLogDetMixin, TransformerMixin):
    pass


class PCA(
        NonLinearLogDetMixin,
        TransformerMixin, _PCA
        ):

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

    def get_feature_names_out(
                self,
                input_features=None
                ):
        return ClassNamePrefixFeaturesOutMixin.get_feature_names_out(
                    self, input_features,
                    )


class _LinBasis(_BaseEstimator, _TransformerMixin):

    def __init__(
            self,
            pca_kwargs=None,
            min_det=None,
            minimize_kwargs=None,
            buffer=None,
            ):
        super().__init__()
        if min_det is None:
            min_det = True

        self._pca_kwargs = pca_kwargs
        self._min_det = min_det
        self._minimize_kwargs = minimize_kwargs
        self._buffer = buffer

    @property
    def _n_features_out(self):
        check_is_fitted(self, attributes="n_components_")
        return self.n_components_

    def fit(
            self,
            X,
            y=None,
            ):

        if isinstance(X, pd.DataFrame):
            X = X.values

        D = X.shape[1]
        self.loc_ = 0.0
        self.A_ = np.eye(D)
        self.V_ = np.eye(D)

        P = self._hull(X)

        self._pca_kwargs = set_default_kwargs(
                        self._pca_kwargs,
                        )
        self._pca_kwargs["n_components"] = D

        pca = _PCA(
                **self._pca_kwargs,
                )
        pca.fit(P)

        cattrs = [
                     "feature_names_in_"
                    ]

        attrs = [
                    "n_features_in_",
                    "n_components_",
                    "n_samples_",
                    *[a for a in cattrs if hasattr(pca, a)],
                    ]

        for k in attrs:
            setattr(self, k, getattr(pca, k, None))

        R = pca.components_.T
        Y = X @ R
        mins = Y.min(axis=0)
        widths = np.maximum(Y.max(axis=0) - mins, 1e-12)

        self.A_ = np.diag(1.0 / widths) @ R.T
        self.loc_ = R @ mins

        self._min_det = bool(self._min_det)
        if self._min_det:
            self._minimize_kwargs = set_default_kwargs(
                                self._minimize_kwargs,
                                dkwargs=dict(
                                            method="SLSQP",
                                            )
                                )
            self._minimize_kwargs["options"] = set_default_kwargs(
                                            self._minimize_kwargs.get("options", None),
                                            dkwargs=dict(
                                                        maxiter=5000,
                                                        ftol=1e-12,
                                                        disp=True,
                                                        )
                                            )
            constraints = [
                            {
                                "type": "ineq",
                                "fun":
                                    lambda p, d=D, x=P:
                                    self._det_min_com(
                                                p, D=D, X=x
                                                )
                            }
                            ]

            x0 = np.concat([self.A_.ravel(), self.loc_])
            self._minimize_result = scipy.optimize.minimize(
                        self._det_min_obj,
                        x0,
                        args=(D,),
                        constraints=constraints,
                        **self._minimize_kwargs,
                        )
            if self._minimize_result.success:
                self.A_, self.loc_ = self._unpack_params(
                                        self._minimize_result.x, D,
                                        )

        self.V_ = np.linalg.inv(self.A_)

        return self

    def _transform(
            self,
            X,
            A,
            loc,
            ):
        return (X - loc) @ A.T

    def _inverse_transform(
            self,
            X,
            V,
            loc,
            ):
        return loc + X @ V.T

    def transform(self, X):
        return self._transform(X=X, A=self.A_, loc=self.loc_)

    def inverse_transform(self, X):
        return self._inverse_transform(X=X, V=self.V_, loc=self.loc_)

    def _hull(self, X):

        Y = self._transform(X=X, A=self.A_, loc=self.loc_)
        if Y.shape[1] == 1:
            uv = [np.argmin(Y), np.argmax(Y)]
        else:
            uv = np.unique(scipy.spatial.ConvexHull(Y).vertices)
        return X[uv]

    def _unpack_params(
            self,
            params,
            D,
            ):
        A = params[:-D].reshape(D, D)
        loc = params[-D:]
        return A, loc

    def _det_min_obj(
            self,
            params,
            D,
            ):
        A = self._unpack_params(params, D)[0]
        sign, logdet = np.linalg.slogdet(A)
        return -logdet

    def _det_min_com(
            self,
            params,
            D,
            X,
            ):

        A, loc = self._unpack_params(params, D)
        Y = self._transform(X=X, A=A, loc=loc)
        lower = Y.ravel()
        upper = (1.0 - Y).ravel()
        return np.concat([lower, upper])
    pass


class LinBasis(
            ClassNamePrefixFeaturesOutMixin,
            _AfflineBasisLogdet, _LinBasis):
    pass
