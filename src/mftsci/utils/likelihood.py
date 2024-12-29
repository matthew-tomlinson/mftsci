"""
Module for likelihood utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Callable

# External packages
import numpy as np
import numdifftools

# Internal packages
from mftsci.utils.meta import default_args_kwargs
from mftsci.utils.xcpt import try_f


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def standard_error(
                    logL: Callable,
                    theta: np.ndarray,
                    theta_free: Optional[np.ndarray] = None,
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

    def minus_logL_red(theta_red):
        theta_full[theta_free] = theta_red
        return logL(theta_full, *args, **kwargs)

    # Compute the Hessian matrix at theta using numdifftools
    step_args, step_kwargs = default_args_kwargs(step_args, step_kwargs)
    hessian_fun = numdifftools.Hessian(
                                        minus_logL_red,
                                        method="central",
                                        step=Hessian_step(theta=theta,
                                                          *step_args, **step_kwargs)
                                        )
    _success, hessian_matrix = try_f(lambda: hessian_fun(theta_red))

    if _success:
        # Invert the Hessian to get the covariance matrix
        covariance_matrix = np.linalg.inv(hessian_matrix)

        # Standard errors are the square roots of the diagonal elements of the
        # covariance matrix
        se_theta_red = np.sqrt(np.diag(covariance_matrix))

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
        if frac is None:
            frac = 1E-6
        if min_step is None:
            min_step = 1E-14

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
