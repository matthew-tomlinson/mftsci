"""
Core module for `dist_tests` subpackage.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from __future__ import annotations
from typing import Optional, Union, Callable

# External packages
import numpy as np
import scipy

# Internal packages
from mftsci import utils    # temp
from mftsci.utils.unit import Unit
from mftsci.htest.htests.core import *  # noqa: F401, F403
from mftsci.htest.htests.core import TestWrapper, TestRes, alpha_to_q    # temp
from mftsci.htest.htests.dist_tests import defaults


# =====================================
# CLASSES
# =====================================

class Dist():

    def __init__(
                    self,
                    dist: Optional[Union[
                                scipy.stats._distn_infrastructure.rv_continuous_frozen,
                                Dist,
                                DistN,
                                ]] = None,
                    cdf: Optional[Callable] = None,
                    ppf: Optional[Callable] = None,
                    pdf: Optional[Callable] = None,
                    ):

        self.set_all(self, dist=dist, cdf=cdf, ppf=ppf, pdf=pdf)

    def set_all(
                self,
                dist: Optional[Union[
                            scipy.stats._distn_infrastructure.rv_continuous_frozen,
                            Dist,
                            DistN,
                            ]] = None,
                cdf: Optional[Callable] = None,
                ppf: Optional[Callable] = None,
                pdf: Optional[Callable] = None,
                ):

        if dist is not None:
            self.cdf = dist.cdf
            self.ppf = dist.ppf
            self.pdf = dist.pdf
        else:
            self.cdf = cdf
            self.ppf = ppf
            self.pdf = pdf


class DistN(Dist):

    def __init__(
                    self,
                    a_lambda: Optional[Union[float, int,]] = None,
                    N_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    N_0: Optional[int] = None,
                    N_1: Optional[int] = None,
                    Delta_N: Optional[int] = None,
                    t_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    t_0: Optional[Union[float, int,]] = None,
                    t_1: Optional[Union[float, int,]] = None,
                    Delta_t: Optional[Union[float, int,]] = None,
                    t_unit: Optional[Unit] = None,
                    ):

        self.set_all(a_lambda=a_lambda,
                     N_bounds=N_bounds, N_0=N_0, N_1=N_1, Delta_N=Delta_N,
                     t_bounds=t_bounds,
                     t_0=t_0, t_1=t_1, Delta_t=Delta_t,
                     t_unit=t_unit,
                     reset=True)

    def set_a_lambda(
                        self,
                        a_lambda: Optional[Union[float, int,]] = None,
                        reset: Optional[bool] = None,
                        ):

        if a_lambda is not None:
            self.a_lambda = a_lambda
        elif utils.meta.defaults.reset(reset=reset):
            self.a_lambda = defaults.a_lambda()

    def set_N_bounds(
                        self,
                        N_bounds: Optional[Union[
                                                    utils.bounds.Bounds,
                                                    np.ndarray,
                                                    list,
                                                    ]] = None,
                        N_0: Optional[int] = None,
                        N_1: Optional[int] = None,
                        Delta_N: Optional[int] = None,
                        reset: Optional[bool] = None,
                        ):

        if utils.meta.defaults.reset(reset=reset):
            self.N_bounds = utils.bounds.Bounds(bounds=N_bounds,
                                                a=N_0,
                                                b=N_1,
                                                Delta=Delta_N)
        else:
            self.N_bounds.set_bounds(bounds=N_bounds, a=N_0, b=N_1, Delta=Delta_N)

    def set_t_bounds(
                        self,
                        t_bounds: Optional[Union[
                                                    utils.bounds.Bounds,
                                                    np.ndarray,
                                                    list,
                                                    ]] = None,
                        t_0: Optional[Union[float, int,]] = None,
                        t_1: Optional[Union[float, int,]] = None,
                        Delta_t: Optional[Union[float, int,]] = None,
                        t_unit: Optional[Unit] = None,
                        reset: Optional[bool] = None,
                        ):

        if utils.meta.defaults.reset(reset=reset):
            self.t_bounds = utils.bounds.Bounds(bounds=t_bounds,
                                                a=t_0, b=t_1,
                                                Delta=Delta_t,
                                                unit=t_unit,
                                                )
        else:
            self.t_bounds.set_bounds(bounds=t_bounds,
                                     a=t_0, b=t_1, Delta=Delta_t,
                                     unit=t_unit,
                                     )

    def calc_Delta_N_eff(
                            self,
                            test: Optional[bool] = None,
                            ) -> Union[np.ndarray, float,]:

        if test is None:
            test = False

        if not test or np.isnan(self.N_bounds.Delta()):
            # return self.a_lambda * self.t_bounds.Delta()
            return self.a_lambda * self.t_bounds.Delta(dedim=True)
        else:
            return self.N_bounds.Delta()

    def set_all(
                self,
                a_lambda: Optional[Union[float, int,]] = None,
                N_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                N_0: Optional[int] = None,
                N_1: Optional[int] = None,
                Delta_N: Optional[int] = None,
                t_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                t_0: Optional[Union[float, int,]] = None,
                t_1: Optional[Union[float, int,]] = None,
                Delta_t: Optional[Union[float, int,]] = None,
                t_unit: Optional[Unit] = None,
                reset: Optional[bool] = None,
                ):

        self.set_a_lambda(a_lambda=a_lambda, reset=reset)
        self.set_N_bounds(N_bounds=N_bounds,
                          N_0=N_0, N_1=N_1, Delta_N=Delta_N,
                          reset=reset)
        self.set_t_bounds(t_bounds=t_bounds,
                          t_0=t_0, t_1=t_1,
                          Delta_t=Delta_t,
                          t_unit=t_unit,
                          reset=reset)

    def cdf(
            self,
            t: Union[np.ndarray, float, int,],
            test: Optional[bool] = None,
            ) -> Union[np.ndarray, float,]:

        if test is None:
            test = False

        Delta_N = self.calc_Delta_N_eff(test=test)

        # F = (self.a_lambda/Delta_N) * (t - self.t_bounds.bounds[0])

        F = (self.a_lambda/Delta_N) *\
            (self.t_bounds.unit.dedim(t) - self.t_bounds.get_bounds(dedim=True)[0])

        return F

    def ppf(
            self,
            q: Union[np.ndarray, float,],
            test: Optional[bool] = None,
            ) -> Union[np.ndarray, float,]:

        if test is None:
            test = False

        Delta_N = self.calc_Delta_N_eff(test=test)

        # t = self.t_bounds.bounds[0] + (Delta_N/self.a_lambda) * q

        t = self.t_bounds.get_bounds(dedim=False)[0] +\
            self.t_bounds.unit.redim((Delta_N/self.a_lambda) * q, skip_loc=True)

        return t

    def pdf(
            self,
            t: Union[np.ndarray, float, int,],
            test: Optional[bool] = None,
            ) -> Union[np.ndarray, float,]:

        if test is None:
            test = False

        Delta_N = self.calc_Delta_N_eff(test=test)

        f = (self.a_lambda/Delta_N) * np.ones_like(t)

        return f


class DistTest(TestWrapper):

    def __init__(
                    self,
                    sample: Optional[Union[np.ndarray, list,]] = None,
                    D_0: Optional[Union[
                                scipy.stats._distn_infrastructure.rv_continuous_frozen,
                                Dist,
                                DistN,
                                ]] = None,
                    D_T: Optional[Union[
                                scipy.stats._distn_infrastructure.rv_continuous_frozen,
                                Dist,
                                DistN,
                                ]] = None,
                    app_PIT: Optional[bool] = None,
                    inf_alpha: Optional[float] = None,
                    *args, **kwargs,
                 ):

        super().__init__(*args, **kwargs)
        self.set_inf_alpha(inf_alpha=inf_alpha, reset=True)
        self.set_D_0(D_0=D_0, reset=True)
        self.set_D_T(D_T=D_T, reset=True)
        self.set_sample(sample=sample, app_PIT=app_PIT, reset=True)

    def clean_sample(
                        self,
                        sample: Optional[Union[np.ndarray, list,]] = None,
                        ) -> np.ndarray:

        return sample

    def set_sample(
                    self,
                    sample: Optional[Union[np.ndarray, list,]] = None,
                    set_res: Optional[bool] = None,
                    app_PIT: Optional[bool] = None,
                    reset: Optional[bool] = None,
                    ):

        if sample is not None:
            self.sample = self.clean_sample(sample)
        elif utils.meta.defaults.reset(reset=reset):
            self.sample = None

        if self.sample is not None:
            self.set_inf_alpha(np.min([self.inf_alpha, 1/np.size(self.sample)]))

        if set_res is None:
            set_res = True

        if set_res:
            if self.sample is not None:
                self.set_res(app_PIT=app_PIT)

    def set_res(
                self,
                app_PIT: Optional[bool] = None,
                ):
        super().set_res()

    def set_inf_alpha(
                        self,
                        inf_alpha: Optional[float] = None,
                        reset: Optional[bool] = None,
                        ):

        if inf_alpha is not None:
            self.inf_alpha = np.max([inf_alpha, 1E-16])
        elif utils.meta.defaults.reset(reset=reset):
            self.inf_alpha = defaults.inf_alpha()

    def set_D_0(
                self,
                D_0: Optional[Union[
                            scipy.stats._distn_infrastructure.rv_continuous_frozen,
                            Dist,
                            DistN,
                            ]] = None,
                reset: Optional[bool] = None,
                ):

        if D_0 is not None:
            self.D_0 = D_0
        elif utils.meta.defaults.reset(reset=reset):
            self.D_0 = scipy.stats.norm()

    def set_D_T(
                self,
                D_T: Optional[Union[
                                scipy.stats._distn_infrastructure.rv_continuous_frozen,
                                Dist,
                                DistN,
                                ]] = None,
                reset: Optional[bool] = None,
                ):

        if D_T is not None:
            self.D_T = D_T
        elif utils.meta.defaults.reset(reset=reset):
            self.D_T = scipy.stats.norm()

    def PIT(
            self,
            x: Union[np.ndarray, list, float, int,],
            inv: Optional[bool] = None,
            inf_alpha: Optional[float] = None,
            make_inf_finite: Optional[bool] = None,
            ) -> Union[np.ndarray, float,]:

        if defaults.inv(inv):
            cdf = self.D_T.cdf
            ppf = self.D_0.ppf
        else:
            cdf = self.D_0.cdf
            ppf = self.D_T.ppf

        F = cdf(x)
        y = ppf(F)

        self.set_inf_alpha(inf_alpha=inf_alpha)

        if make_inf_finite is None:
            make_inf_finite = self.inf_alpha > 0

        if make_inf_finite:
            if self.inf_alpha > 0:
                if np.isscalar(x):
                    xtype_float = isinstance(x, float)
                    if xtype_float:
                        x_is_inf = np.isinf(x)
                    else:
                        x_is_inf = False
                else:
                    xtype_float = x.dtype == float
                    if xtype_float:
                        x_is_inf = np.isinf(x)
                    else:
                        x_is_inf = np.full(x.shape, False)

                if np.isscalar(y):
                    ytype_float = isinstance(y, float)
                    if ytype_float:
                        y_is_inf = np.isinf(y)
                    else:
                        y_is_inf = False
                else:
                    ytype_float = y.dtype == float
                    if ytype_float:
                        y_is_inf = np.isinf(y)
                    else:
                        y_is_inf = np.full(y.shape, False)

                F_0 = self.inf_alpha
                F_1 = 1 - self.inf_alpha

                y_0 = ppf(F_0)
                y_1 = ppf(F_1)

                if np.isscalar(x_is_inf):
                    if not x_is_inf and y_is_inf:
                        if F < F_0:
                            y = y_0
                        elif F > F_1:
                            y = y_1
                else:
                    to_inf = np.logical_and(~x_is_inf, y_is_inf)
                    to_inf_F_0 = np.logical_and(to_inf, F < F_0)
                    to_inf_F_1 = np.logical_and(to_inf, F > F_1)

                    y[to_inf_F_0] = y_0
                    y[to_inf_F_1] = y_1

        return y

    def get_sample(
                    self,
                    app_PIT: Optional[bool] = None,
                    ) -> np.ndarray:

        if defaults.app_PIT(app_PIT=app_PIT):
            return self.PIT(self.sample)
        else:
            return self.sample

    def get_D_0(
                self,
                app_PIT: Optional[bool] = None,
                ) -> Union[
                            scipy.stats._distn_infrastructure.rv_continuous_frozen,
                            Dist,
                            DistN,
                            ]:

        if defaults.app_PIT(app_PIT=app_PIT):
            return self.D_T
        else:
            return self.D_0

    def get_cdf_0(
                    self,
                    app_PIT: Optional[bool] = None,
                    ) -> Callable:

        return self.get_D_0(app_PIT=app_PIT).cdf

    def get_ppf_0(
                    self,
                    app_PIT: Optional[bool] = None,
                    ) -> Callable:

        return self.get_D_0(app_PIT=app_PIT).ppf

    def calc_N_sample(
                        self,
                        N: Optional[int] = None,
                        ) -> Union[int, None]:

        if N is None:
            if self.sample is not None:
                N = np.size(self.sample)

        return N

    def calc_N_sample_nan(
                            self,
                            N: Optional[int] = None,
                            ) -> Union[int, float]:

        N = self.calc_N_sample(N=N)
        if N is None:
            N = np.nan

        return N

    def calc_sample_bounds(
                            self,
                            app_PIT: Optional[bool] = None,
                            excl_inf: Optional[bool] = None,
                            ) -> Union[utils.bounds.Bounds, None]:

        sample = self.get_sample(app_PIT=app_PIT)

        if sample is None:
            return None
        else:
            if defaults.excl_inf(excl_inf):
                if np.issubdtype(sample.dtype, np.floating):
                    sample = sample[~np.isinf(sample)]
            return utils.bounds.Bounds(a=np.min(sample), b=np.max(sample))

    def calc_q_bounds(
                        self,
                        q_bounds: Optional[Union[
                                                    utils.bounds.Bounds,
                                                    np.ndarray,
                                                    list,
                                                    ]] = None,
                        alpha: Optional[Union[np.ndarray, list, float,]] = None,
                        app_PIT: Optional[bool] = None,
                        ) -> utils.bounds.Bounds:

        if q_bounds is None:
            if alpha is None:
                alpha = 2*1E-4

            q_bounds = self.get_ppf_0(app_PIT=app_PIT)(alpha_to_q(alpha=alpha))

        return utils.bounds.Bounds(bounds=q_bounds)

    def calc_x_bounds(
                        self,
                        x_bounds: Optional[Union[
                                                    utils.bounds.Bounds,
                                                    np.ndarray,
                                                    list,
                                                    ]] = None,
                        alpha: Optional[Union[np.ndarray, list, float,]] = None,
                        app_PIT: Optional[bool] = None,
                        ) -> utils.bounds.Bounds:

        app_PIT = defaults.app_PIT(app_PIT=app_PIT)

        x_bounds = self.calc_q_bounds(q_bounds=x_bounds, alpha=alpha, app_PIT=app_PIT)

        bounds_list = [x_bounds]
        sample_bounds = self.calc_sample_bounds(app_PIT=app_PIT, excl_inf=True)
        if sample_bounds is not None:
            bounds_list.append(sample_bounds)

        return utils.bounds.outer_merge(bounds_list=bounds_list)

    def calc_x(
                self,
                x: Optional[Union[np.ndarray, list, float, int,]] = None,
                x_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                alpha: Optional[Union[np.ndarray, list, float,]] = None,
                app_PIT: Optional[bool] = None,
                xs_num: Optional[int] = None,
                xs_delta: Optional[Union[float, int,]] = None,
                ) -> np.ndarray:

        if x is None:
            x_bounds = self.calc_x_bounds(x_bounds=x_bounds,
                                          alpha=alpha,
                                          app_PIT=app_PIT)
            x = x_bounds.xs(num=xs_num, delta=xs_delta)

        return x


class DistTestRes(TestRes):

    def __init__(
                    self,
                    test_long_name: str,
                    test_short_name: str,
                    app_PIT: Optional[bool] = None,
                    ):

        TestRes.__init__(self,
                         test_long_name=test_long_name,
                         test_short_name=test_short_name)

        self.app_PIT = app_PIT


# =====================================
# FUNCTIONS
# =====================================


def ret_Dist(
                dist: Optional[Union[
                            scipy.stats._distn_infrastructure.rv_continuous_frozen,
                            Dist,
                            DistN,
                            ]] = None,
                cdf: Optional[Callable] = None,
                ppf: Optional[Callable] = None,
                pdf: Optional[Callable] = None,
                ) -> Union[
                            scipy.stats._distn_infrastructure.rv_continuous_frozen,
                            Dist,
                            DistN,
                            ]:

    if dist is not None:
        return dist
    else:
        return Dist(cdf=cdf, ppf=ppf, pdf=pdf)
