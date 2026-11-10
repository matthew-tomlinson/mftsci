"""
Functions module for tdist.truncnorm
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union

# External packages
import numpy as np
import scipy
import pandas as pd
import mftsci as msci

# Local packages

# Local modules
from mftsci.tdist.truncnorm import defaults


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================

def limits_to_uls(limits):

    df = pd.concat([
            limits.iloc[:, 0].rename("loc"),
            limits.diff(axis=1).iloc[:, 1].rename("scale"),
            ], axis=1)

    df.columns = df.columns.set_names("theta")

    return df


def limits_to_mls(limits, rescale=None):
    if rescale is None:
        rescale = 1.0

    df = pd.concat([
            limits.mean(axis=1).rename("loc"),
            rescale * limits.diff(axis=1).iloc[:, 1].rename("scale")/2,
            ], axis=1)

    df.columns = df.columns.set_names("theta")

    return df


def calc_kwargs_old(
            lim,
            is_sym=None,
            ):
    """
    Calculates stochastic keyword arguments for scipy.stats.truncnorm
    """

    kwargs = pd.Series(np.nan, index=["a", "b", "loc", "scale"])
    if is_sym is None:
        is_sym = True

    kwargs["loc"] = np.mean(lim)
    if is_sym:
        frac_lims = [0.25, 0.32]
    else:
        kwargs["loc"] *= (1 - (1/300))
        frac_lims = [0.3, 0.35]
    kwargs["scale"] = np.diff(lim)[0] * np.random.uniform(*frac_lims)

    # convert limits to truncnorm parameters
    kwargs["a"], kwargs["b"] = (lim - kwargs["loc"]) / kwargs["scale"]

    return kwargs


def calc_kwargs(
            lim,
            is_sym=None,
            ):
    """
    Calculates stochastic keyword arguments for scipy.stats.truncnorm
    """

    kwargs = pd.Series(np.nan, index=["a", "b", "loc", "scale"])
    if is_sym is None:
        is_sym = True

    kwargs["loc"] = np.mean(lim)
    if is_sym:
        frac_lims = [0.25, 0.32]
    else:
        frac_lims = [0.3, 0.35]
    kwargs["scale"] = np.diff(lim)[0] * np.random.uniform(*frac_lims)

    # convert limits to truncnorm parameters
    kwargs["a"], kwargs["b"] = (lim - kwargs["loc"]) / kwargs["scale"]

    if not is_sym:
        kwargs["loc"] += scipy.stats.truncnorm(
                                        a=kwargs["a"], b=kwargs["b"],
                                        scale=kwargs["scale"]/2, loc=0).rvs()
        kwargs["a"], kwargs["b"] = (lim - kwargs["loc"]) / kwargs["scale"]

    return kwargs


def make_theta(
                a: Optional[float] = None,
                b: Optional[float] = None,
                loc: Optional[float] = None,
                scale: Optional[float] = None,
                xa: Optional[float] = None,
                xb: Optional[float] = None,
                buffer: Optional[float] = None,
                as_pd: Optional[bool] = None
                ):

    theta = np.array([a, b, loc, scale], dtype=float)
    xtheta = np.array([xa, xb], dtype=float)
    is_xtheta = ~np.isnan(xtheta)
    if np.any(is_xtheta):
        ab = (xtheta - theta[2]) / theta[3]
        theta[:2] = np.where(is_xtheta, ab, theta[:2])

    buffer = defaults.set_default(buffer, 1)
    theta[:2] *= buffer

    if defaults.set_default(as_pd, False):
        theta = pd.Series(theta, index=["a", "b", "loc", "scale"])

    return theta


def calc_xtheta(
                a, b,
                loc, scale,
                as_pd: Optional[bool] = None
                ):

    theta = np.array([a, b, loc, scale], dtype=float)
    xtheta = theta[2] + theta[3] * theta[:2]

    if defaults.set_default(as_pd, False):
        xtheta = pd.Series(xtheta, index=["xa", "xb"])

    return xtheta


def fit(
        data,
        buffer: Optional[float] = None,
        res_kwargs: Optional[dict] = None,
        **kwargs,
        ):

    _params = pd.Index(msci.utils.pdist.list_parameters(dist="truncnorm"))
    N_lim = 2

    theta_init = _params.to_series().apply(
                        lambda v: kwargs.get(f"f{v}", np.nan)
                        ).astype(float)
    theta_free = np.isnan(theta_init)
    if theta_free["loc"]:
        theta_init["loc"] = np.mean(data)
    if theta_free["scale"]:
        if False:
            theta_init["scale"] = np.std(data)
        else:
            theta_init["scale"] = np.diff(np.quantile(
                                            data, scipy.stats.norm.cdf([-0.5, 0.5])
                                            ))[0]

    xtheta_fixed = ("x" + _params[:N_lim]).to_series().apply(
                            lambda v: kwargs.get(f"f{v}", np.nan)
                            ).astype(float)
    xtheta_data = pd.Series(
                    [np.min(data), np.max(data)],
                    index=xtheta_fixed.index
                    )
    xtheta_free = np.isnan(xtheta_fixed) & theta_free.values[:N_lim]
    xtheta_0 = xtheta_fixed.combine_first(xtheta_data[xtheta_free])

    buffer = np.atleast_1d(defaults.set_default(buffer, 1)).reshape(-1)[:N_lim]
    buffer = np.where(xtheta_free, buffer, 1)

    opt_theta_free = theta_free[N_lim:]
    bounds = msci.utils.pdist.get_scipy_dist_param_bounds(dist="truncnorm")[N_lim:]
    bounds[0] = xtheta_data.values
    bounds[1, 0] = 1E-6

    def _make_theta(theta):

        return make_theta(
                        *theta_init.values[:2],
                        *theta,
                        *xtheta_0,
                        buffer=buffer,
                        as_pd=False,
                        )

    def log_L(theta):

        _theta = _make_theta(theta=theta)
        ll = -scipy.stats.truncnorm.nnlf(theta=_theta, x=data)
        return ll

    opt_theta_init = theta_init.values[N_lim:]

    res_kwargs = msci.utils.meta.default_kwargs(res_kwargs)

    res = msci.utils.likelihood.MaxLogLRes(
            log_L=log_L,
            theta_init=opt_theta_init,
            theta_free=opt_theta_free,
            bounds=bounds,
            N_data=np.size(data),
            **res_kwargs,
            )

    res.tnorm_theta_init = theta_init
    res.tnorm_xtheta_0 = xtheta_0
    res.tnorm_buffer = buffer

    res.tnorm_theta = make_theta(*_make_theta(theta=res.res.x), as_pd=True)
    res.tnorm_xtheta = calc_xtheta(
                            **res.tnorm_theta,
                            as_pd=True,
                            )

    return res


def infer_loc(
        mean: float,
        scale: float,
        xa: float,
        xb: float,
        opt_kwargs: Optional[dict] = None,
        ):

    def misfit(x):

        return (
                mean - scipy.stats.truncnorm.mean(
                            *make_theta(
                                loc=x[0], scale=scale,
                                xa=xa, xb=xb,
                                )
                            )
                )**2

    bounds = scipy.optimize.Bounds(lb=xa, ub=xb)
    x0 = np.atleast_1d(mean).reshape(-1)[:1]

    opt_kwargs = msci.utils.meta.default_kwargs(opt_kwargs)

    res = scipy.optimize.minimize(fun=misfit, x0=x0, bounds=bounds, **opt_kwargs)

    res.tnorm_theta = make_theta(
                                loc=res.x[0], scale=scale,
                                xa=xa, xb=xb,
                                as_pd=True,
                                )

    return res


def infer_locscale_xab_alpha(
        xab_alpha: Union[pd.DataFrame, np.ndarray],
        alpha: Union[pd.DataFrame, pd.Series, np.ndarray, float],
        as_pd: Optional[bool] = None,
        ):

    was_pd = False
    columns = getattr(xab_alpha, "columns", None)
    if hasattr(xab_alpha, "values"):
        was_pd = True
        xab_alpha = xab_alpha.values

    if hasattr(alpha, "values"):
        was_pd = True
        if hasattr(alpha, "columns"):
            if columns is not None:
                alpha = alpha[columns]
            else:
                columns = alpha.columns
        elif alpha.shape[0] != 2:
            if columns is not None:
                alpha = alpha.loc[columns]
            else:
                columns = alpha.index

        alpha = alpha.values

    as_pd = defaults.set_default(as_pd, was_pd)

    alpha = np.atleast_1d(alpha)
    if alpha.ndim == 1:
        N_alpha = alpha.shape[0]
        if N_alpha != 2:
            alpha = np.repeat(alpha/2, 2, axis=0).reshape(N_alpha, 2).T

    Fab = alpha.copy()
    Fab[1] = 1 - Fab[1]

    zab = scipy.stats.norm().ppf(Fab)

    locscale = np.array([
                    np.diff(zab * xab_alpha[::-1], axis=0)[0] / np.diff(zab, axis=0)[0],
                    np.diff(xab_alpha, axis=0)[0] / np.diff(zab, axis=0)[0],
                    ])
    if as_pd:
        locscale = pd.DataFrame(locscale.T, columns=["loc", "scale"], index=columns)

    return locscale


def infer_kwargs_xab_alpha(
        xab: Union[pd.DataFrame, np.ndarray],
        xab_alpha: Union[pd.DataFrame, np.ndarray],
        alpha: Union[pd.DataFrame, pd.Series, np.ndarray, float],
        as_pd: Optional[bool] = None,
        ):

    was_pd = False
    columns = getattr(xab, "columns", None)
    if hasattr(xab, "values"):
        was_pd = True
        xab = xab.values

    if hasattr(xab_alpha, "values"):
        was_pd = True
        if columns is not None:
            xab_alpha = xab_alpha[columns]
        else:
            columns = getattr(xab_alpha, "columns", None)
        xab_alpha = xab_alpha.values

    if hasattr(alpha, "values"):
        was_pd = True
        if hasattr(alpha, "columns"):
            if columns is not None:
                alpha = alpha[columns]
            else:
                columns = alpha.columns
        elif alpha.shape[0] != 2:
            if columns is not None:
                alpha = alpha.loc[columns]
            else:
                columns = alpha.index

        alpha = alpha.values

    as_pd = defaults.set_default(as_pd, was_pd)

    locscale = infer_locscale_xab_alpha(xab_alpha=xab_alpha, alpha=alpha, as_pd=False)

    kwargs = np.concat(
                [
                    (xab - locscale[0])/locscale[1],
                    locscale,
                    ],
                axis=0
                )

    if as_pd:
        kwargs = pd.DataFrame(
                    kwargs.T, columns=["a", "b", "loc", "scale"],
                    index=columns,
                    )

    return kwargs


def infer_locscale_xab_old(
        xab: Union[pd.DataFrame, np.ndarray],
        xab_soft: Union[pd.DataFrame, np.ndarray],
        alpha_soft: Union[pd.DataFrame, pd.Series, np.ndarray, float],
        opt_kwargs: Optional[dict] = None,
        ):

    def _make_theta(locscale, xab):

        N = int(locscale.size/2)
        loc = locscale[:N]
        scale = locscale[N:]

        a = (xab[0] - loc)/scale
        b = (xab[1] - loc)/scale

        return {"a": a, "b": b, "loc": loc, "scale": scale}

    def misfit(locscale, xab, xab_soft, alpha_soft):

        soft_cdf = scipy.stats.truncnorm.cdf(
                                xab_soft,
                                **_make_theta(locscale, xab)
                            )

        return np.sum(np.array([
                            alpha_soft[0] - soft_cdf[0],
                            alpha_soft[1] - (1 - soft_cdf[1])
                            ])**2)

    columns = getattr(xab, "columns", None)
    if hasattr(xab, "values"):
        xab = xab.values

    if hasattr(xab_soft, "values"):
        if columns is not None:
            xab_soft = xab_soft[columns]
        else:
            columns = getattr(xab_soft, "columns", None)
        xab_soft = xab_soft.values

    if hasattr(alpha_soft, "values"):
        if hasattr(alpha_soft, "columns"):
            if columns is not None:
                alpha_soft = alpha_soft[columns]
            else:
                columns = alpha_soft.columns
        elif alpha_soft.shape[0] != 2:
            if columns is not None:
                alpha_soft = alpha_soft.loc[columns]
            else:
                columns = alpha_soft.index

        alpha_soft = alpha_soft.values

    alpha_soft = np.atleast_1d(alpha_soft)
    if alpha_soft.ndim == 1:
        N_alpha = alpha_soft.shape[0]
        if N_alpha != 2:
            alpha_soft = np.repeat(alpha_soft/2, 2, axis=0).reshape(N_alpha, 2).T

    loc_0 = np.mean(xab_soft, axis=0)
    scale_0 = np.diff(xab_soft, axis=0)[0] / 2

    N = loc_0.size
    locscale_0 = np.concat([loc_0, scale_0])
    locscale_lb = 1.0 * locscale_0
    locscale_ub = 1.0 * locscale_0

    locscale_0[N:] *= 0.75
    locscale_lb[N:] *= 0.5

    locscale_lb[:N] = xab_soft[0]
    locscale_ub[:N] = xab_soft[1]

    opt_kwargs = msci.utils.meta.default_kwargs(opt_kwargs)

    res = scipy.optimize.minimize(
                        misfit, x0=locscale_0,
                        args=(xab, xab_soft, alpha_soft),
                        bounds=scipy.optimize.Bounds(lb=locscale_lb, ub=locscale_ub),
                        )
    if alpha_soft.shape[1] == 1:
        alpha_soft = np.repeat(alpha_soft, N).reshape(2, N)
    res.alpha_soft = pd.DataFrame(alpha_soft, columns=columns)

    res.tnorm_theta = pd.DataFrame(
                            _make_theta(
                                locscale=res.x,
                                xab=xab,
                                ),
                            index=columns,
                            )
    return res


def calc_p_i_old(
        alpha: float,
        N: int,
        k: Optional[int] = None,
        ):

    if np.isscalar(alpha):
        if k is None:
            k = 2
        p_i = np.repeat((1 - alpha**(1/N)) / k, k)
    else:
        alpha = np.atleast_1d(alpha).reshape(-1)
        if k is not None:
            alpha = alpha[:k]
        p_i = (1 - alpha**(1/N))

    return p_i


def expand_ab_old(
        ab: np.ndarray,
        alpha: float,
        N: int,
        ):

    p_i = calc_p_i_old(alpha=alpha, N=N, k=2)
    F_ab = scipy.stats.norm.cdf(ab)
    F_ab_new = 0.5 * (1 - (1 - 2*F_ab)/(1 - 2*p_i))
    F_ab_new[F_ab_new < 0] = 0
    F_ab_new[F_ab_new > 1] = 1
    ab_new = scipy.stats.norm.ppf(F_ab_new)

    return ab_new


def calc_p_i(
        alpha: float,
        N: int,
        k: Optional[int] = None,
        ):

    N_shape = np.shape(N)
    N = np.atleast_1d(N).reshape(-1)

    if np.isscalar(alpha):
        if k is None:
            k = 2
        p_i = np.repeat((1 - alpha**(1/N)) / k, k).reshape(*[*N_shape, k])
    else:
        alpha_shape = np.shape(alpha)
        alpha = np.atleast_1d(alpha).reshape(-1)

        p_i = (1 - np.power.outer(alpha, 1/N).T).reshape(*[*N_shape, *alpha_shape])

    return p_i


def expand_ab(
        ab: np.ndarray,
        alpha: float,
        N: int,
        ):

    p_i = calc_p_i(alpha=alpha, N=N, k=2)
    F_ab = scipy.stats.norm.cdf(ab)
    F_ab_new = 0.5 * (1 - (1 - 2*F_ab)/(1 - 2*p_i))
    F_ab_new[F_ab_new < 0] = 0
    F_ab_new[F_ab_new > 1] = 1
    ab_new = scipy.stats.norm.ppf(F_ab_new)

    return ab_new
