"""
Functions module for likelihood utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import numpy as np
import scipy
import pandas as pd
import numdifftools

# Internal packages
from mftsci.utils.meta import default_args, default_kwargs, \
                                default_args_kwargs     # noqa: F401
from mftsci.utils.xcpt import try_f
from mftsci.utils.arr import iterarr
from mftsci.utils.dict import filter_keys
from mftsci.utils.timer import Timer, ftimer
from mftsci.utils.likelihood import defaults


# =====================================
# CLASSES
# =====================================

class MaxLogLRes():

    def __init__(
                    self,
                    log_L: Callable,
                    theta_init: np.ndarray,
                    theta_free: Optional[np.ndarray] = None,
                    bounds: Optional[np.ndarray] = None,
                    opt: Optional[bool] = None,
                    opt_kwargs: Optional[dict] = None,
                    step_args: Optional[tuple] = None,
                    step_kwargs: Optional[dict] = None,
                    names: Optional[Union[np.ndarray, list, str, int]] = None,
                    free_only: Optional[bool] = None,
                    N_data: Optional[Union[np.ndarray, int]] = None,
                    index: Optional[Union[
                                        pd.core.indexes.base.Index,
                                        np.ndarray,
                                        list,
                                        ]] = None,
                    *args, **kwargs,
                    ):

        self.set_est(
                        log_L=log_L,
                        theta_init=theta_init,
                        theta_free=theta_free,
                        bounds=bounds,
                        opt=opt,
                        opt_kwargs=opt_kwargs,
                        *args, **kwargs,
                    )
        self.set_theta(
                        log_L=log_L,
                        theta_init=theta_init,
                        theta_free=theta_free,
                        bounds=bounds,
                        step_args=step_args,
                        step_kwargs=step_kwargs,
                        names=names,
                        free_only=free_only,
                        *args, **kwargs,
                    )
        self.set_eval(
                        log_L=log_L,
                        N_data=N_data,
                        index=index,
                    )

    def __repr__(self):

        if self.opt_timer is None:
            timer = ""
        else:
            # timer = f"\n{self.opt_timer.__repr__()}"
            timer = ""

        return f"{self.theta.__repr__()}\n\n{self.eval.__repr__()}{timer}"

    def set_log_L(
                    self,
                    log_L: Callable,
                    theta_init: Optional[np.ndarray] = None,
                    theta_free: Optional[np.ndarray] = None,
                    bounds: Optional[np.ndarray] = None,
                    *args, **kwargs,
                    ):

        self.args = args
        self.kwargs = kwargs

    def set_est(
                    self,
                    log_L: Callable,
                    theta_init: Optional[np.ndarray] = None,
                    theta_free: Optional[np.ndarray] = None,
                    bounds: Optional[np.ndarray] = None,
                    opt: Optional[bool] = None,
                    opt_kwargs: Optional[dict] = None,
                    *args, **kwargs,
                ):

        self.est, self.res, self.opt_timer = max_log_L_est(
                                            log_L=log_L,
                                            theta_init=theta_init,
                                            theta_free=theta_free,
                                            bounds=bounds,
                                            opt=opt,
                                            opt_kwargs=opt_kwargs,
                                            *args, **kwargs,
                                            )

    def set_theta(
                    self,
                    log_L: Callable,
                    theta_init: Optional[np.ndarray] = None,
                    theta_free: Optional[np.ndarray] = None,
                    bounds: Optional[np.ndarray] = None,
                    step_args: Optional[tuple] = None,
                    step_kwargs: Optional[dict] = None,
                    names: Optional[Union[np.ndarray, list, str, int]] = None,
                    free_only: Optional[bool] = None,
                    *args, **kwargs,
                    ):

        self.theta = calc_log_L_theta(
                                        log_L=log_L,
                                        est=self.est,
                                        theta_init=theta_init,
                                        theta_free=theta_free,
                                        bounds=bounds,
                                        step_args=step_args,
                                        step_kwargs=step_kwargs,
                                        names=names,
                                        free_only=free_only,
                                        *args, **kwargs,
                                        )

    def set_eval(
                    self,
                    log_L: Callable,
                    N_data: Optional[Union[np.ndarray, int]] = None,
                    index: Optional[Union[
                                        pd.core.indexes.base.Index,
                                        np.ndarray,
                                        list,
                                        ]] = None,
                    *args, **kwargs,
                ):

        _log_L = log_L(self.est, *args, **kwargs)
        N_params = np.sum(self.theta["free"])

        self.eval = calc_log_L_eval(
                                    log_L=_log_L, N_params=N_params, N_data=N_data,
                                    index=index,
                                    )


# =====================================
# FUNCTIONS
# =====================================


def max_log_L_est(
                        log_L: Callable,
                        theta_init: np.ndarray,
                        opt: Optional[bool] = None,
                        theta_free: Optional[np.ndarray] = None,
                        bounds: Optional[np.ndarray] = None,
                        opt_kwargs: Optional[dict] = None,
                        *args, **kwargs,
                        ):

    theta_init = iterarr(theta_init)

    opt = defaults.opt(opt)

    if opt:
        theta_shape = theta_init.shape

        if theta_free is None:
            theta_free = np.full(theta_shape, True)

        # Reduced parameter set excluding the fixed parameters
        theta_init_red = theta_init[theta_free]

        # Define a wrapper for the log likelihood that only considers free parameters
        theta_full = np.copy(theta_init)

        def minus_log_L_red(theta_red):
            theta_full[theta_free] = theta_red
            return -log_L(theta_full, *args, **kwargs)

        # Bounds
        if bounds is not None:
            bounds = scipy.optimize.Bounds(
                                            lb=bounds[theta_free, 0],
                                            ub=bounds[theta_free, 1]
                                            )

        # Optimisation
        opt_kwargs = default_kwargs(opt_kwargs)
        res, opt_timer = ftimer(ret_Timer=True)(lambda: scipy.optimize.minimize(
                                                            minus_log_L_red,
                                                            x0=theta_init_red,
                                                            bounds=bounds,
                                                            **opt_kwargs,
                                                            ))()

        if False:
            opt_timer = Timer()
            res = scipy.optimize.minimize(
                                            minus_log_L_red, x0=theta_init_red,
                                            bounds=bounds,
                                            **opt_kwargs,
                                            )
            opt_timer.stop()

        theta_est = np.copy(theta_init)
        theta_est[theta_free] = res.x
    else:
        theta_est = np.copy(theta_init)
        res = None
        opt_timer = None

    return theta_est, res, opt_timer


def calc_log_L_theta(
                    log_L: Callable,
                    est: Union[np.ndarray, list, float] = None,
                    theta_free: Optional[np.ndarray] = None,
                    theta_init: Optional[np.ndarray] = None,
                    bounds: Optional[np.ndarray] = None,
                    step_args: Optional[tuple] = None,
                    step_kwargs: Optional[dict] = None,
                    names: Optional[Union[np.ndarray, list, str, int]] = None,
                    free_only: Optional[bool] = None,
                    *args, **kwargs,
                    ):

    free_only = defaults.free_only(free_only)

    if theta_free is None:
        theta_free = np.full(np.shape(est), True)

    se = standard_error(log_L=log_L, theta=est, theta_free=theta_free, bounds=bounds,
                        step_args=step_args, step_kwargs=step_kwargs)

    if free_only:
        include = theta_free
    else:
        include = np.full(np.size(est), True)

    ret_dict = {
                    "est": est[include],
                    "se": se[include],
                    "free": theta_free[include],
                }
    if theta_init is not None:
        theta_init = iterarr(theta_init)
        ret_dict["init"] = theta_init[include]

    if names is None:
        names = np.arange(np.size(est))
    names = iterarr(names)

    return pd.DataFrame(ret_dict, index=names[include])


def standard_error(
                    log_L: Callable,
                    theta: np.ndarray,
                    theta_free: Optional[np.ndarray] = None,
                    bounds: Optional[np.ndarray] = None,
                    step_args: Optional[tuple] = None,
                    step_kwargs: Optional[dict] = None,
                    *args, **kwargs,
                    ) -> np.ndarray:

    theta_shape = theta.shape

    if theta_free is None:
        theta_free = np.full(theta_shape, True)

    # Initialise standard errors array with NaNs
    se_theta = np.full(theta_shape, np.nan)

    # Reduced parameter set excluding the fixed parameters
    theta_red = theta[theta_free]

    # Define a wrapper for the log-likelihood that only considers free parameters
    theta_full = np.copy(theta)

    def minus_log_L_red(theta_red):
        theta_full[theta_free] = theta_red
        return -log_L(theta_full, *args, **kwargs)

    # Compute the Hessian matrix at theta using numdifftools
    step_args, step_kwargs = default_args_kwargs(step_args, step_kwargs)
    hessian_fun = numdifftools.Hessian(
                                        minus_log_L_red,
                                        method="central",
                                        step=Hessian_step(theta=theta, bounds=bounds,
                                                          *step_args, **step_kwargs)
                                        )
    _success_0, hessian_matrix = try_f(lambda: hessian_fun(theta_red))

    if _success_0:
        # Invert the Hessian to get the covariance matrix
        _success_1, covariance_matrix = try_f(lambda: np.linalg.inv(hessian_matrix))

        if _success_1:
            # Standard errors are the square roots of the diagonal elements of the
            # covariance matrix
            se_theta_red = np.sqrt(np.abs(np.diag(covariance_matrix)))

            # Insert the calculated standard errors for free parameters
            se_theta[theta_free] = se_theta_red

    return se_theta


def Hessian_step(
                    method: Optional[str] = None,
                    theta: Optional[np.ndarray] = None,
                    frac: Optional[Union[np.ndarray, float,]] = None,
                    min_step: Optional[float] = None,
                    bounds: Optional[np.ndarray] = None,
                    ) -> Union[np.ndarray, float, None,]:

    if method is None:
        if theta is not None and frac is not None:
            method = "theta_frac"

    if method == "theta_frac":
        frac = defaults.frac(frac)
        min_step = defaults.frac(min_step)

        step = frac * np.abs(theta)
        if min_step is not None and not np.isnan(min_step):
            step[step < min_step] = min_step

        if bounds is not None:
            theta_edge = np.expand_dims(theta, axis=1) + np.outer(step,
                                                                  np.array([-1, 1]))
            clip = np.max((theta_edge - bounds) * np.array([-1, 1]), axis=1)
            clip[clip < 0] = 0.0
            step -= clip

    else:
        step = None

    return step


def calc_log_L_eval(
                        log_L: Union[np.ndarray, float],
                        N_params: Union[np.ndarray, int],
                        N_data: Optional[Union[np.ndarray, int]] = None,
                        index: Optional[Union[
                                            pd.core.indexes.base.Index,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                    ) -> Union[pd.core.series.Series, pd.core.frame.DataFrame]:

    N_data = defaults.N_data(N_data)

    log_L = iterarr(log_L)
    N_params = iterarr(N_params, dtype=int)
    N_data = iterarr(N_data, dtype=int)

    deviance = calc_deviance(log_L=log_L)
    AIC = calc_AIC(log_L=log_L, N_params=N_params)
    BIC = calc_BIC(log_L=log_L, N_params=N_params, N_data=N_data)

    _locals = filter_keys(locals(), keys_rm=["index"])

    return pd.DataFrame(_locals, index=index)


def calc_deviance(
                    log_L: Union[np.ndarray, list, float],
                    ) -> Union[np.ndarray, list, float]:

    return -2*np.array(log_L)


def calc_AIC(
                log_L: Union[np.ndarray, list, float],
                N_params: Union[np.ndarray, list, int],
                ) -> Union[np.ndarray, list, float]:

    return 2*(np.array(N_params) - np.array(log_L))


def calc_BIC(
                log_L: Union[np.ndarray, list, float],
                N_params: Union[np.ndarray, list, int],
                N_data: Union[np.ndarray, list, int],
                ) -> Union[np.ndarray, float]:

    return np.log(N_data)*np.array(N_params) - 2*np.array(log_L)
