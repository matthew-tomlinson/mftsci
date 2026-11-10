"""
Functions module for linalg.regr
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import numpy as np
import pandas as pd
import scipy


# Internal packages
from mftsci import utils
from mftsci.utils.regr import clean_X, clean_y, clean_W, capp_bias, prep_y_X_W, \
                                calc_df as _calc_df, calc_N_data_params     # noqa: F401
from mftsci.linalg.regr import defaults  # noqa: F401
from mftsci.htest.confidence_interval import ConfInterval


# =====================================
# CLASSES
# =====================================

class LinearRegression:

    def __init__(
            self,
            prep_X: Optional[Callable] = None,
            ):

        self.set_prep_X(prep_X=prep_X, reset=True)
        self._app_bias = False
        self._bias_name = None
        self._beta_fixed = None

    def fit(
            self,
            y: Union[np.ndarray, pd.Series],
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            as_pd: Optional[bool] = None,
            save_ic: Optional[bool] = None,
            as_matrix: Optional[bool] = None,
            app_bias: Optional[bool] = None,
            check_bias: Optional[bool] = None,
            bias_name: Optional[str] = None,
            beta_fixed: Optional[Union[np.ndarray, pd.Series,]] = None,
            prep_X: Optional[Callable] = None,
            ):

        self.set_prep_X(prep_X=prep_X)
        X = self._prep_X(X)
        if X.ndim < 2:
            _N_params = 1
        else:
            _N_params = X.shape[1]
        (
            y, X, W,
            ) = prep_y_X_W(
                    y=y, X=X, W=W,
                    as_pd=as_pd, save_ic=save_ic,
                    as_matrix=as_matrix,
                    app_bias=app_bias, check_bias=check_bias,
                    bias_name=bias_name,
                    valid_only=True,
                    ret_all=False,
                    )
        self._N_params = X.shape[1]
        self._app_bias = self._N_params == (_N_params + 1)
        is_X_pd = hasattr(X, "values")

        if is_X_pd:
            self._beta_fixed = pd.Series(np.nan, index=X.columns)
        else:
            self._beta_fixed = np.repeat(np.nan, self._N_params)

        if beta_fixed is not None:
            is_bf_pd = hasattr(beta_fixed, "values")
            if not is_bf_pd:
                beta_fixed = np.atleast_1d(beta_fixed).reshape(-1)[:self._N_params]

            if is_X_pd:
                if is_bf_pd:
                    self._beta_fixed = beta_fixed.combine_first(
                                            self._beta_fixed
                                            ).loc[self._beta_fixed.index]
                else:
                    self._beta_fixed.iloc[:beta_fixed.size] = beta_fixed
            else:
                if is_bf_pd:
                    beta_fixed = beta_fixed.values
                self._beta_fixed[:beta_fixed.size] = beta_fixed

        is_fixed = ~np.isnan(self._beta_fixed)
        any_fixed = np.any(is_fixed)
        if any_fixed:
            if is_X_pd:
                X_fixed = X.loc[:, is_fixed]
                X = X.loc[:, ~is_fixed]
            else:
                X_fixed = X[:, is_fixed]
                X = X[:, ~is_fixed]
            y -= calc_y_pred_mean(
                        X=X_fixed,
                        beta=self._beta_fixed[is_fixed],
                        )

        self.N_data, self.N_params = calc_N_data_params(X=X)
        beta_rcov = calc_beta_rcov(X=X, W=W)
        beta = calc_beta(X=X, y=y, W=W, beta_rcov=beta_rcov)

        if any_fixed:
            if is_X_pd:
                self.beta_rcov = beta_rcov.combine_first(
                                    pd.DataFrame(
                                        np.nan,
                                        index=self._beta_fixed.index,
                                        columns=self._beta_fixed.index,
                                        )
                                    ).loc[
                                        self._beta_fixed.index,
                                        self._beta_fixed.index,
                                        ]
                self.beta = beta.combine_first(
                                    self._beta_fixed,
                                    ).loc[self._beta_fixed.index]
            else:
                self.beta_rcov = np.full((self._N_params, self._N_params), np.nan)
                self.beta = self._beta_fixed.copy()
                self.beta_rcov[~is_fixed, ~is_fixed] = beta_rcov
                self.beta[~is_fixed] = beta
        else:
            self.beta_rcov = beta_rcov
            self.beta = beta

        self.sigma2 = calc_sigma2(y=y, X=X, W=W, beta=beta)
        self.log_det_W = calc_log_det_W(W=W)
        self.logL = _calc_logL(
                        N_data=self.N_data, N_params=self.N_params,
                        sigma2=self.sigma2, log_det_W=self.log_det_W,
                        )

        if self._app_bias:
            self._app_bias = True
            if hasattr(self.beta, "values"):
                self._bias_name = self.beta.index[0]
            else:
                self._bias_name = None
        else:
            self._bias_name = None

        return self

    def set_prep_X(
                self,
                prep_X: Optional[Callable] = None,
                reset: Optional[bool] = None,
                ):

        if prep_X is not None:
            self._prep_X = prep_X
        elif defaults.reset(reset):
            self._prep_X = utils.meta.return_x

    @property
    def df(
            self,
            ) -> int:

        return _calc_df(N_data=self.N_data, N_params=self.N_params)

    @property
    def beta_se(
            self,
            ):

        return calc_beta_se(beta_rcov=self.beta_rcov, sigma2=self.sigma2)

    def beta_VIF(
            self,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            as_pd: Optional[bool] = None,
            app_prep: Optional[bool] = None,
            ):

        return calc_beta_VIF(
                    X=self.prep_X(X, as_pd=as_pd, app_prep=app_prep),
                    W=W,
                    )

    @property
    def beta_pvalue(
            self,
            ):

        return calc_beta_pvalue(
                    beta=self.beta,
                    beta_se=self.beta_se,
                    N_data=self.N_data,
                    N_params=self.N_params,
                    )

    def beta_power(
            self,
            alpha: Optional[float] = None,
            ratio: Optional[float] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            as_pd: Optional[bool] = None,
            app_prep: Optional[bool] = None,
            ):

        return calc_beta_power(
                    alpha=alpha,
                    ratio=ratio,
                    X=self.prep_X(X, as_pd=as_pd, app_prep=app_prep),
                    W=W,
                    N_params=self.N_params,
                    )

    def prep_X(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            as_pd: Optional[bool] = None,
            app_prep: Optional[bool] = None,
            ):

        if app_prep is None:
            app_prep = True

        if not app_prep:
            return X
        else:
            return capp_bias(
                        X=self._prep_X(X),
                        app_bias=self._app_bias,
                        check_bias=False,
                        bias_name=self._bias_name,
                        as_pd=as_pd,
                        )

    def y_sim(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            N_resamp: Optional[int] = None,
            as_pd: Optional[bool] = None,
            app_prep: Optional[bool] = None,
            ) -> Union[np.ndarray, pd.Series]:

        N_resamp = defaults.set_default(N_resamp, 1)
        X = self.prep_X(X=X, as_pd=as_pd, app_prep=app_prep)

        y_mean = self.y_pred_mean(X=X, app_prep=False).repeat(N_resamp)

        if self.sigma2 == 0:
            return y_mean
        else:
            y_se = self.y_pred_se(X=X, app_prep=False).repeat(N_resamp)
            return y_mean + np.random.randn(y_se.shape[0]) * y_se

    def y_pred_mean(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            as_pd: Optional[bool] = None,
            app_prep: Optional[bool] = None,
            ) -> Union[np.ndarray, pd.Series]:

        return calc_y_pred_mean(
                    X=self.prep_X(X=X, as_pd=as_pd, app_prep=app_prep),
                    beta=self.beta,
                    )

    def y_pred_se(
                self,
                X: Union[np.ndarray, pd.DataFrame],
                as_pd: Optional[bool] = None,
                of_mean: Optional[bool] = None,
                app_prep: Optional[bool] = None,
                ) -> Union[np.ndarray, pd.Series]:

        X = self.prep_X(X=X, as_pd=as_pd, app_prep=app_prep)
        beta_rcov = self.beta_rcov
        is_free = np.isnan(self._beta_fixed)
        if np.any(~is_free):
            if hasattr(X, "values"):
                X = X.loc[:, is_free]
            else:
                X = X[:, is_free]

            if hasattr(beta_rcov, "values"):
                beta_rcov = beta_rcov.loc[is_free, is_free]
            else:
                beta_rcov = beta_rcov[is_free, is_free]

        return calc_y_pred_se(
                X=X,
                beta_rcov=beta_rcov,
                sigma2=self.sigma2,
                of_mean=of_mean,
                )

    def y_pred_ppf(
                self,
                X: Union[np.ndarray, pd.DataFrame],
                q: Union[float, np.ndarray, pd.Series],
                of_mean: Optional[bool] = None,
                as_pd: Optional[bool] = None,
                app_prep: Optional[bool] = None,
                ) -> Union[np.ndarray, pd.Series]:

        # f = (1 - np.outer(1 - CI.alpha, np.array([-1, 1])))/2

        X = self.prep_X(X=X, as_pd=as_pd, app_prep=app_prep)

        y_pred_mean = self.y_pred_mean(X=X, as_pd=as_pd, app_prep=False)
        y_pred_se = self.y_pred_se(X=X, as_pd=as_pd, of_mean=of_mean, app_prep=False)
        was_pd = hasattr(y_pred_mean, "values")
        if hasattr(q, "values"):
            q_pd = True
        else:
            q_pd = False

        if as_pd is None:
            as_pd = was_pd or q_pd

        if was_pd:
            index = y_pred_mean.index
            y_pred_mean = y_pred_mean.values
            y_pred_se = y_pred_se.values
        else:
            index = None

        if q_pd:
            if not as_pd:
                q = q.values
            else:
                if isinstance(q, pd.DataFrame):
                    q = q.stack()
        else:
            if as_pd:
                q = utils.pd.arr_to_srs(q)

        q_index = None
        if hasattr(q, "values"):
            q_index = q.index

        if hasattr(y_pred_mean, "values"):
            y_pred_mean = y_pred_mean.values

        t = scipy.stats.t(df=self.df).ppf(q)
        if not as_pd:
            y_dev = np.expand_dims(t, axis=-1) * y_pred_se
        else:
            y_dev = pd.DataFrame(np.outer(y_pred_se, t), index=index, columns=q_index)

        y = y_pred_mean.reshape(-1, 1) + y_dev

        return y

    def y_pred_CI(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            CI: Optional[ConfInterval] = None,
            alpha: Optional[Union[float, list, np.ndarray]] = None,
            of_mean: Optional[bool] = None,
            as_pd: Optional[bool] = None,
            app_prep: Optional[bool] = None,
            ) -> Union[np.ndarray, pd.Series]:

        if CI is None:
            CI = ConfInterval(alpha=alpha)

        return self.y_pred_ppf(
                    X=X,
                    q=CI.q_srs(),
                    of_mean=of_mean,
                    as_pd=as_pd,
                    app_prep=app_prep,
                    )


# =====================================
# FUNCTIONS
# =====================================


def calc_df(
            N_data: Optional[int] = None,
            N_params: Optional[int] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y: Optional[Union[np.ndarray, pd.Series]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_resid: Optional[Union[np.ndarray, pd.Series]] = None,
            beta: Optional[Union[np.ndarray, pd.Series]] = None,
            ):
    """
    Calculate degrees of freedom.
    """
    return _calc_df(*calc_N_data_params(
                    N_data=N_data, N_params=N_params,
                    X=X,
                    y=y,
                    W=W,
                    y_resid=y_resid,
                    beta=beta,
                    ))


def calc_Gram(
                X: Union[np.ndarray, pd.DataFrame, pd.Series],
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                X_1: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                inv: Optional[bool] = None,
                ):
    """
    Calculate [inverse] Gram (information) matrix.
    """

    if X_1 is None:
        X_1 = X

    if W is None:
        G = X.T @ X_1
    elif W.ndim == 2:
        G = X.T @ W @ X_1
    else:
        G = (X.T * W) @ X_1

    if defaults.set_default(inv, False):
        G_inv = np.linalg.inv(G)
        if hasattr(G, "values"):
            G_inv = pd.DataFrame(G_inv, index=G.index, columns=G.columns)
        return G_inv
    else:
        return G


def calc_D_opt_slow(
                X: Union[np.ndarray, pd.DataFrame, pd.Series],
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                ):
    """
    Calculate D-optimality criterion
    """
    return np.linalg.det(calc_Gram(X=X, W=W, inv=False))


def calc_hlog_D_opt(
                X: Union[np.ndarray, pd.DataFrame, pd.Series],
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                ):
    """
    Calculate half log D-optimality criterion
    """
    if W is not None:
        if W.ndim == 2:
            X = np.linalg.cholesky(W).T @ X
        elif W.ndim == 1:
            if hasattr(W, "values"):
                W = W.values
            X *= X * W[:, None]

    return np.sum(np.log(np.linalg.svd(X, compute_uv=False)))


def calc_D_opt(
                X: Union[np.ndarray, pd.DataFrame, pd.Series],
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                ):
    """
    Calculate D-optimality criterion
    """
    return np.exp(2*calc_hlog_D_opt(X=X, W=W))


def calc_alias(
        X_0: Union[np.ndarray, pd.DataFrame, pd.Series],
        X_1: Union[np.ndarray, pd.DataFrame, pd.Series],
        W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
        ):
    """
    Calculate alias matrix
    """

    return calc_Gram(X=X_0, W=W, inv=True) @ calc_Gram(X=X_0, W=W, X_1=X_1, inv=False)


def calc_beta_rcov(
                X: Union[np.ndarray, pd.DataFrame, pd.Series],
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                ):
    """
    Calculate relative correlation of beta
    """
    return calc_Gram(X=X, W=W, inv=True)


def calc_beta(
                X: Union[np.ndarray, pd.DataFrame, pd.Series],
                y: Union[np.ndarray, pd.Series],
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
                ):
    """
    Calculate linear regression coefficients (beta).
    """

    if beta_rcov is None:
        beta_rcov = calc_beta_rcov(X=X, W=W)

    if W is not None:
        if W.ndim == 2:
            y = W @ y
        else:
            y = W * y

    return beta_rcov @ X.T @ y


def calc_y_pred_mean(
                X: Union[np.ndarray, pd.DataFrame, pd.Series],
                beta: Optional[Union[np.ndarray, pd.Series,]] = None,
                y: Optional[Union[np.ndarray, pd.Series]] = None,
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
                ):
    """
    Calculate y predicted mean
    """
    if beta is None:
        beta = calc_beta(X=X, y=y, W=W, beta_rcov=beta_rcov)

    return beta @ X.T


def calc_y_resid(
                y: Union[np.ndarray, pd.Series],
                y_pred_mean: Optional[Union[np.ndarray, pd.Series]] = None,
                X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
                beta: Optional[Union[np.ndarray, pd.Series,]] = None,
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
                ):
    """
    Calculate residuals.
    """

    if y_pred_mean is None:
        y_pred_mean = calc_y_pred_mean(X=X, beta=beta, y=y, W=W, beta_rcov=beta_rcov)

    return y_pred_mean - y


def calc_sigma2(
            y_resid: Optional[Union[np.ndarray, pd.Series]] = None,
            y: Union[np.ndarray, pd.Series] = None,
            y_pred_mean: Optional[Union[np.ndarray, pd.Series]] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            beta: Optional[Union[np.ndarray, pd.Series,]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
            ):
    """
    Calculate residual variance.
    """
    df = calc_df(
            X=X,
            y=y,
            W=W,
            y_resid=y_resid,
            beta=beta,
            )

    if y_resid is None:
        y_resid = calc_y_resid(
                            y=y, y_pred_mean=y_pred_mean,
                            X=X, beta=beta,
                            W=W, beta_rcov=beta_rcov,
                            )

    if W is None:
        sigma2_sum = y_resid @ y_resid
    elif W.ndim == 2:
        sigma2_sum = y_resid @ W @ y_resid
    else:
        sigma2_sum = y_resid @ (W * y_resid)

    return sigma2_sum / df


def calc_beta_se(
            sigma2: Optional[float] = None,
            beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
            y_resid: Optional[Union[np.ndarray, pd.Series]] = None,
            y: Union[np.ndarray, pd.Series] = None,
            y_pred_mean: Optional[Union[np.ndarray, pd.Series]] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            beta: Optional[Union[np.ndarray, pd.Series,]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            ):
    """
    Calculate linear regression coefficients standard error (beta_se).
    """
    if beta_rcov is None:
        beta_rcov = calc_beta_rcov(X=X, W=W)

    if sigma2 is None:
        sigma2 = calc_sigma2(
                    y_resid=y_resid,
                    y=y,
                    y_pred_mean=y_pred_mean,
                    X=X,
                    beta=beta,
                    W=W,
                    beta_rcov=beta_rcov,
                    )

    beta_se = np.sqrt(sigma2 * np.diag(beta_rcov))

    if hasattr(beta_rcov, "values"):
        beta_se = pd.Series(beta_se, index=beta_rcov.index)

    return beta_se


def calc_beta_VIF(
                beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
                N_data: Optional[int] = None,
                X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                ):
    """
    Calculate Variance Inflation Factor

    """
    if beta_rcov is None:
        beta_rcov = calc_beta_rcov(X=X, W=W)

    if N_data is None:
        N_data = X.shape[0]

    VIF = N_data * np.diag(beta_rcov)

    if hasattr(beta_rcov, "values"):
        VIF = pd.Series(VIF, index=beta_rcov.index)

    return VIF


def calc_beta_pvalue(
            beta: Optional[Union[np.ndarray, pd.Series,]] = None,
            beta_se: Optional[Union[np.ndarray, pd.Series,]] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
            y: Union[np.ndarray, pd.Series] = None,
            y_pred_mean: Optional[Union[np.ndarray, pd.Series]] = None,
            y_resid: Optional[Union[np.ndarray, pd.Series]] = None,
            sigma2: Optional[float] = None,
            N_data: Optional[int] = None,
            N_params: Optional[int] = None,
            ):

    if beta is None and beta_se is None:
        if beta_rcov is None:
            beta_rcov = calc_beta_rcov(X=X, W=W)

    if beta is None:
        beta = calc_beta(
                X=X,
                y=y,
                W=W,
                beta_rcov=beta_rcov,
                )
    if beta_se is None:
        beta_se = calc_beta_se(
                    sigma2=sigma2,
                    beta_rcov=beta_rcov,
                    y_resid=y_resid,
                    y=y,
                    y_pred_mean=y_pred_mean,
                    X=X,
                    beta=beta,
                    W=W,
                    )

    df = calc_df(
            N_data=N_data,
            N_params=N_params,
            X=X,
            y=y,
            W=W,
            y_resid=y_resid,
            beta=beta,
            )

    t = beta / beta_se
    p = 2*(1 - scipy.stats.t(df=df).cdf(np.abs(t)))
    if hasattr(t, "values"):
        p = pd.Series(p, index=t.index)

    return p


def calc_beta_power(
                alpha: Optional[float] = None,
                ratio: Optional[float] = None,
                beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
                X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
                W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
                N_params: Optional[int] = None,
                ):
    """
    power is probability that the target delta will result in a rejection of the
    null hypothesis: beta == 0

    """
    alpha = defaults.set_default(alpha, 0.05)
    ratio = defaults.set_default(ratio, 1.0)

    if beta_rcov is None:
        beta_rcov = calc_beta_rcov(X=X, W=W)

    df = calc_df(
                N_params=N_params,
                X=X,
                W=W,
                )

    t_crit = scipy.stats.t(df=df).ppf(1 - alpha/2)

    delta = ratio / np.sqrt(np.diag(beta_rcov))
    power = 1 - np.diff(
                    scipy.stats.nct(df=df, nc=delta).cdf(
                            t_crit*np.array([-1, 1]).reshape(-1, 1)
                            ),
                    axis=0,
                    )[0]

    if hasattr(beta_rcov, "values"):
        power = pd.Series(power, index=beta_rcov.index)

    return power


def calc_y_pred_se(
            sigma2: Optional[float] = None,
            beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
            y_resid: Optional[Union[np.ndarray, pd.Series]] = None,
            y: Union[np.ndarray, pd.Series] = None,
            y_pred_mean: Optional[Union[np.ndarray, pd.Series]] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            beta: Optional[Union[np.ndarray, pd.Series,]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            of_mean: Optional[bool] = None,
            ):
    """
    Calculate standard error of predicted y.
    """
    of_mean = defaults.set_default(of_mean, False)

    if beta_rcov is None:
        beta_rcov = calc_beta_rcov(X=X, W=W)

    if sigma2 is None:
        sigma2 = calc_sigma2(
                    y_resid=y_resid,
                    y=y,
                    y_pred_mean=y_pred_mean,
                    X=X,
                    beta=beta,
                    W=W,
                    beta_rcov=beta_rcov,
                    )

    return np.sqrt(
            sigma2 * (
                int(not of_mean) + (X.T * (beta_rcov @ X.T))
                ).sum(axis=0)
            )


def calc_log_det_W(
        W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
        ) -> float:

    if W is None:
        log_det_W = 0.0
    elif W.ndim == 2:
        sign, log_det_W = np.linalg.slogdet(W)
    else:
        log_det_W = np.sum(np.log(np.abs(W)))

    return log_det_W


def _calc_logL(
        N_data: int,
        N_params: int,
        sigma2: float,
        log_det_W: float,
        ):
    """
    Calculate GLS log-likelihood.
    """

    df = _calc_df(N_data=N_data, N_params=N_params)
    return -0.5 * (
                    N_data * np.log(2 * np.pi)
                    - log_det_W
                    + df * sigma2
                    )


def calc_logL(
            N_data: Optional[int] = None,
            N_params: Optional[int] = None,
            sigma2: Optional[float] = None,
            log_det_W: Optional[float] = None,
            X: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            W: Optional[Union[np.ndarray, pd.DataFrame, pd.Series,]] = None,
            y: Union[np.ndarray, pd.Series] = None,
            beta_rcov: Optional[Union[np.ndarray, pd.DataFrame,]] = None,
            beta: Optional[Union[np.ndarray, pd.Series,]] = None,
            y_resid: Optional[Union[np.ndarray, pd.Series]] = None,
            y_pred_mean: Optional[Union[np.ndarray, pd.Series]] = None,
            ):
    """
    Calculate GLS log-likelihood.
    """

    N_data, N_params = calc_N_data_params(
                N_data=N_data, N_params=N_params,
                X=X,
                y=y,
                W=W,
                y_resid=y_resid,
                beta=beta,
                )

    if sigma2 is None:
        sigma2 = calc_sigma2(
                    y_resid=y_resid,
                    y=y,
                    y_pred_mean=y_pred_mean,
                    X=X,
                    beta=beta,
                    W=W,
                    beta_rcov=beta_rcov,
                    )

    if log_det_W is None:
        log_det_W = calc_log_det_W(W=W)

    return _calc_logL(
                N_data=N_data, N_params=N_params,
                sigma2=sigma2, log_det_W=log_det_W,
                )
