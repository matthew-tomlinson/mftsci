"""
Functions module for scipy distribution utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
import sys
from typing import Optional, Union

# External packages
import numpy as np
import scipy
import pandas as pd

# Internal packages
from mftsci.utils import arr
from mftsci.utils.dict import filter_keys
from mftsci.utils.unit import Unit
from mftsci.utils.likelihood import MaxLogLRes as _MaxLogLRes
from mftsci.utils.pdist import defaults


# =====================================
# CLASSES
# =====================================

class MaxLogLRes(_MaxLogLRes):

    def __init__(
                    self,
                    sample: np.ndarray,
                    unit: Optional[Unit] = None,
                    dedim: Optional[bool] = None,
                    dist: Optional[str] = None,
                    nan_rm=None,
                    theta_init: Optional[np.ndarray] = None,
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
                    **fparams,
                    ):

        opt = defaults.opt(opt)
        nan_rm = defaults.nan_rm(nan_rm)
        self._dist = defaults.dist(dist)

        fparams = get_fparams(
                                dist=dist, dist_only=True, add_None=False,
                                **fparams,
                                )

        if dedim is None:
            if unit is not None:
                dedim = True
            else:
                dedim = False
        if unit is None:
            unit = Unit()
        self.unit = unit
        sample = self.unit.dedim(sample, do=dedim)

        if theta_init is None:
            theta_init = calc_scipy_dist_est(
                                            sample=sample, dist=dist,
                                            nan_rm=False,
                                            **fparams,
                                        )
        else:
            theta_init = arr.iterarr(theta_init)

        theta_free = ~get_scipy_dist_param_fparams_bool(dist=dist, fparams=fparams)

        bounds = get_scipy_dist_param_bounds(dist=dist, rm=None, as_array=True)

        N_data = np.size(sample)

        names = get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None)

        def log_L(theta):
            return log_L_scipy_dist(sample=sample, dist=dist, params=theta)

        super().__init__(
                            log_L=log_L,
                            theta_init=theta_init,
                            theta_free=theta_free,
                            bounds=bounds,
                            opt=opt,
                            opt_kwargs=opt_kwargs,
                            step_args=step_args,
                            step_kwargs=step_kwargs,
                            names=names,
                            free_only=free_only,
                            N_data=N_data,
                            index=index,
                        )

    def calc_est_dist(self):

        return getattr(scipy.stats, self._dist)(*self.est)


# =====================================
# FUNCTIONS
# =====================================

def list_parameters(
                        dist,
                        moment_order=None,
                        ):
    """List parameters for scipy.stats.dist.
    # Arguments
        dist: a string or scipy.stats distribution object.
    # Returns
        A list of distribution parameter strings.
    """
    moment_order = defaults.moment_order(moment_order)

    if isinstance(dist, str):
        dist = getattr(scipy.stats, dist)
    if dist.shapes:
        parameters = [name.strip() for name in dist.shapes.split(',')]
    else:
        parameters = []

    loc_scale = []

    if dist.name in scipy.stats._discrete_distns._distn_names:
        loc_scale = ["loc"]
    elif dist.name in scipy.stats._continuous_distns._distn_names:
        loc_scale = ["loc", "scale"]
    else:
        sys.exit("Distribution name not found in discrete or continuous lists.")

    if defaults.moment_order(moment_order):
        parameters = loc_scale + parameters
    else:
        parameters = parameters + loc_scale

    return parameters


def get_scipy_dist_param_list(
                                dist=None,
                                rm=None,
                                moment_order=None,
                                ):

    parameters = arr.filter(list_parameters(dist=dist,
                                            moment_order=moment_order),
                            a_rm=rm,
                            )

    return parameters


def get_scipy_dist_param_list_index(
                                    dist=None,
                                    params=None,
                                    rm=None,
                                    source_rm=None,
                                    source_moment_order=None,
                                    as_array=None,
                                    ):

    as_array = defaults.as_array(as_array)

    _scipy_dist_param_list = get_scipy_dist_param_list(
                                dist=dist,
                                rm=source_rm,
                                moment_order=source_moment_order,
                                )

    if params is not None:
        _scipy_dist_param_list_key = arr.iterarr(params)
    else:
        _scipy_dist_param_list_key = get_scipy_dist_param_list(
                                                        dist=dist,
                                                        rm=rm,
                                                        moment_order=True,
                                                        )

    _scipy_dist_param_list_index = {p: None for p in _scipy_dist_param_list_key}
    for p in range(_scipy_dist_param_list_key.size):
        param = _scipy_dist_param_list_key[p]
        where = np.where(param == _scipy_dist_param_list)[0]
        if np.size(where) > 0:
            _scipy_dist_param_list_index[param] = where[0]

    if as_array:
        _scipy_dist_param_list_index = np.array([
                                            _scipy_dist_param_list_index[p]
                                            for p in _scipy_dist_param_list_index
                                            ])

    return _scipy_dist_param_list_index


def get_scipy_dist_param_fparams_bool(
                                        dist,
                                        fparams,
                                        ):

    names = get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None)

    fparams_index = get_scipy_dist_param_list_index(dist=dist,
                                                    params=[f"{fparam}"[1:]
                                                            for fparam in fparams],
                                                    as_array=True)
    fparams_bool = np.full(np.size(names), False)
    for k in fparams_index:
        if k is not None:
            fparams_bool[k] = True

    return fparams_bool


def get_scipy_dist_param_bounds(
                                param=None,
                                dist=None,
                                rm=None,
                                as_array=None,
                                ):

    as_array = defaults.as_array(as_array)

    _scipy_dist_param_bounds = {
                                "a": np.array([0, np.inf]),  # Exclusive lower
                                "b": np.array([0, np.inf]),  # Exclusive lower
                                "c": np.array([-1, np.inf]),
                                "df": np.array([2, np.inf]),
                                "nc": np.array([-np.inf, np.inf]),
                                "loc": np.array([-np.inf, np.inf]),
                                "scale": np.array([0, np.inf]),  # Exclusive lower
                                }

    if param is not None:
        return _scipy_dist_param_bounds[param]
    else:
        if dist is None:
            _scipy_dist_param_bounds_dict = _scipy_dist_param_bounds
        else:
            _scipy_dist_param_bounds_dict = {p: _scipy_dist_param_bounds.get(
                                                        p,
                                                        np.inf*np.array([-1, 1])
                                                        )
                                             for p in get_scipy_dist_param_list(
                                                            dist=dist, rm=rm,
                                                            )}

        if as_array:
            return np.array([_scipy_dist_param_bounds_dict[p]
                             for p in _scipy_dist_param_bounds_dict])


def get_fparams(
                dist=None,
                dist_only=None,
                # add_None=None,
                **fparams,
                ):

    dist_only = defaults.dist_only(dist_only)
    # add_None = defaults.add_None(add_None)

    if dist_only:
        dist = defaults.dist(dist)

        fparams = filter_keys(
                    fparams,
                    keys_in="f" + np.array(get_scipy_dist_param_list(
                                                                    dist=dist,
                                                                    rm=None,
                                                                    moment_order=None,
                                                                    )),
                    )

    return fparams


def get_scipy_dist_param_symb(
                                param=None,
                                dist=None,
                                specific=None,
                                ):

    _scipy_dist_param_symb = {
                                "a": "a",
                                "b": "b",
                                "c": "\\xi",
                                "df": "\\nu",
                                "nc": "\\eta",
                                "loc": "\\mu",
                                "scale": "\\sigma",
                                }

    if dist is not None:
        specific = defaults.specific(specific)

        if specific:
            if np.isin(dist, ["expon", "genpareto"]):
                _scipy_dist_param_symb["loc"] == "u"
                _scipy_dist_param_symb["scale"] == "\\varsigma"

    if param is None:
        return _scipy_dist_param_symb
    else:
        return _scipy_dist_param_symb.get(param, param)


def get_scipy_dist_fparam(
                            param=None,
                            ):

    _scipy_dist_fparam = {f"f{param}": param for param in get_scipy_dist_param_symb()}

    if param is None:
        return _scipy_dist_fparam
    else:
        return _scipy_dist_fparam.get(param, f"f{param}")


def calc_scipy_dist_est(sample,
                        dist=None,
                        nan_rm=None,
                        **fparams,
                        # train_t=None, train_T=None, train_N=None, train_N_T=None,
                        ):

    dist = defaults.dist(dist)
    nan_rm = defaults.nan_rm(nan_rm)

    fparams = get_fparams(
                        dist=dist, dist_only=True, add_None=False,
                        **fparams,
                        )
    kwargs = {
                **{"data": sample},
                **fparams
            }

    fparams_list_no_f = [param[1:] for param in fparams]
    params_list = get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None)

    if all(ele in fparams_list_no_f for ele in params_list):
        fit = np.array([fparams[f"f{param}"] for param in params_list], dtype=float)
    else:
        if np.size(sample) == 0:
            kwargs["data"] = [0]
            fit = np.array(getattr(scipy.stats, dist).fit(**kwargs))
            fit[
                ~np.isin(
                            np.arange(fit.size),
                            get_scipy_dist_param_list_index(
                                dist=dist,
                                params=[param for param in fparams_list_no_f],
                                as_array=True,
                                )
                            )] = np.nan
        else:
            fit = np.array(getattr(scipy.stats, dist).fit(**kwargs))

    return fit


def log_L_scipy_dist(
                        sample,
                        dist=None,
                        params=None,
                        nan_rm=None,
                        **fparams,
                        # train_t=None, train_T=None, train_N=None, train_N_T=None,
                        ):

    dist = defaults.dist(dist)
    if params is None:
        params = calc_scipy_dist_est(sample=sample, dist=dist,
                                     nan_rm=nan_rm,
                                     **fparams,
                                     # train_t=train_t, train_T=train_T,
                                     # train_N=train_N, train_N_T=train_N_T,
                                     )

    return np.sum(np.log(getattr(scipy.stats, dist)(*np.array(params)).pdf(sample)))
