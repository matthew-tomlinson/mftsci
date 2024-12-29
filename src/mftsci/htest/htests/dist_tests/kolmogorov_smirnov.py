"""
Kolmogorov-Smirnov test module.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from __future__ import annotations
from typing import Optional, Union, Callable, Tuple

# External packages
import numpy as np  # temp
import scipy  # temp
import matplotlib  # temp
import matplotlib.pyplot as plt     # temp

# Internal packages
from mftsci.htest.htests.dist_tests.core import *   # noqa: F401, F403
from mftsci.htest.htests.dist_tests.core import ConfInterval, \
                                                TestWrapper, TestMeta, TestRes, \
                                                DistTest, Dist, DistN, \
                                                default_app_PIT, default_N, \
                                                default_alpha, alpha_to_q, \
                                                q_to_salpha   # temp
from mftsci import utils    # temp


# =====================================
# CLASSES
# =====================================


class KolmogorovSmirnovMeta(TestMeta):

    def __init__(
                    self,
                    ):

        TestMeta.__init__(self,
                          test_long_name="Kolmogorov--Smirnov",
                          test_short_name="KS")


class KolmogorovSmirnov(DistTest, KolmogorovSmirnovMeta):

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
                    ):

        KolmogorovSmirnovMeta.__init__(self)
        DistTest.__init__(self, sample=sample, D_0=D_0, D_T=D_T, app_PIT=app_PIT)
        self._point_proc = False

    def clean_sample(
                        self,
                        sample: Optional[Union[np.ndarray, list,]] = None,
                        ) -> np.ndarray:

        if sample is None:
            return sample
        else:
            return np.sort(sample)

    def set_res(
                self,
                app_PIT: Optional[bool] = None,
                ) -> KolmogorovSmirnovResult:

        self.res = KolmogorovSmirnovResult(sample=self.get_sample(app_PIT=app_PIT),
                                           cdf_0=self.get_cdf_0(app_PIT=app_PIT))

        return self.res

    def calc_C_ppf(
                    self,
                    x: Optional[Union[np.ndarray, list, float, int,]] = None,
                    q: Union[np.ndarray, float,] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    N: Optional[int] = None,
                    app_PIT: Optional[bool] = None,
                    x_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    x_bounds_alpha: Optional[float] = None,
                    xs_num: Optional[int] = None,
                    xs_delta: Optional[Union[float, int,]] = None,
                    ) -> np.ndarray:

        x = self.calc_x(x=x, x_bounds=x_bounds, alpha=x_bounds_alpha, app_PIT=app_PIT,
                        xs_num=xs_num, xs_delta=xs_delta)

        return calc_KS_C_ppf(x=x, q=q, alpha=alpha, N=self.calc_N_sample(N=N),
                             cdf=self.get_cdf_0(app_PIT=app_PIT))

    def plot(
                self,
                ax: Optional[matplotlib.axes._axes.Axes] = None,
                app_PIT: Optional[bool] = None,
                x_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                x_bounds_alpha: Optional[float] = None,
                xs_num: Optional[int] = None,
                xs_delta: Optional[Union[float, int,]] = None,
                plot_emp_args: Optional[tuple] = None,
                plot_emp_kwargs: Optional[dict] = None,
                CI_args: Optional[tuple] = None,
                CI_kwargs: Optional[dict] = None,
                F_0: Optional[float] = None,
                norm_F: Optional[bool] = None,
                ):

        if ax is None:
            fig, ax = plt.subplots()

        app_PIT = default_app_PIT(app_PIT=app_PIT)

        if norm_F is None:
            if self._point_proc:
                norm_F = False
            else:
                norm_F = True
        if F_0 is None:
            if self._point_proc and not norm_F:
                F_0 = self.get_D_0(app_PIT=app_PIT).N_bounds.bounds[0]
            else:
                F_0 = 0.0

        # Sample
        sample_x = np.sort(self.get_sample(app_PIT=app_PIT))

        # Scaling
        if self._point_proc:
            N_eff = self.get_D_0(app_PIT=app_PIT).calc_Delta_N_eff(test=False)
        else:
            N_eff = sample_x.size
        if norm_F:
            ylabel = "CDF"
            F_scale = 1/N_eff
        else:
            ylabel = "$N$"
            F_scale = 1.0
        F_max = F_0 + F_scale*np.max([N_eff, sample_x.size])

        # x-axis
        if self._point_proc:
            if x_bounds is None:
                x_bounds = self.get_D_0(app_PIT=app_PIT).t_bounds

        x_bounds = self.calc_x_bounds(app_PIT=app_PIT,
                                      x_bounds=x_bounds,
                                      alpha=x_bounds_alpha)
        xs = x_bounds.xs(num=xs_num, delta=xs_delta)

        # Plot 95% and 99% confidence intervals
        CI_args, CI_kwargs = utils.meta.default_args_kwargs(CI_args, CI_kwargs)
        CI = ConfInterval(*CI_args, **CI_kwargs)
        if CI.N > 0:
            CI_x = xs
            CI_y = F_0 + N_eff*F_scale*self.calc_C_ppf(app_PIT=app_PIT,
                                                       x=CI_x,
                                                       alpha=CI.alpha,
                                                       N=self.sample.size)
            for k in range(CI.N):
                ax.fill_between(CI_x,
                                y1=CI_y[k, 0, :], y2=CI_y[k, 1, :],
                                color=CI.color[k])
            # Add CI legend
            ax.legend(ax.get_children()[:CI.N], CI.calc_CI_strs(), loc="lower right")

        # Plot Empirical CDF
        plot_emp_args, plot_emp_kwargs = utils.meta.default_args_kwargs(plot_emp_args,
                                                                        plot_emp_kwargs)
        ax.plot(*calc_x_y_step(sample_x, y_0=F_0, y_scale=F_scale),
                *plot_emp_args, **plot_emp_kwargs)

        # Print KS test p-value on axis
        self.set_res(app_PIT=app_PIT)
        p_KS = self.res.p
        p_str = CI.p_scif_reject(p=p_KS, brac_kwargs={"plot": True})
        ax.annotate(
                    f"$p_{{\\mathrm{{KS}}}} = {p_str}$",
                    xycoords="axes fraction", xy=[0.02, 0.98],
                    horizontalalignment="left",
                    verticalalignment="top",
                    )

        # Labels and limits
        if not norm_F:
            utils.plot.integer_ticks(ax=ax, xys="y",
                                     maj_mult=None, min_mult=None,
                                     app_minor=None,
                                     maj_offset=None, min_offset=None)
        ax.set_ylabel(ylabel)
        # ax.set_xlabel(f"")
        ylim = F_0 + F_max*np.array([0, 1])
        ax.set_ylim(ylim)
        ax.set_xlim(x_bounds.bounds)


class KolmogorovSmirnovResult(TestRes, KolmogorovSmirnovMeta):

    def __init__(
                    self,
                    sample: Union[np.ndarray, list,],
                    cdf_0: Optional[Callable] = None,
                    ):

        KolmogorovSmirnovMeta.__init__(self)
        self.run(sample=sample, cdf_0=cdf_0)

    def __defaultkeys__(
                        self,
                        ) -> list:

        return ["stat", "p", "stat_loc", "stat_sign"]

    def run(
            self,
            sample: Union[np.ndarray, list, None,],
            cdf_0: Optional[Callable] = None,
            ):

        if sample is not None:
            self.res = scipy.stats.kstest(sample, cdf_0)

            self.stat = self.res.statistic
            self.p = self.res.pvalue
            self.stat_loc = self.res.statistic_location
            self.stat_sign = self.res.statistic_sign
        else:
            for attr in ["res", "stat", "p", "stat_loc", "stat_sign"]:
                setattr(self, attr, None)


class KolmogorovSmirnovPoisson(KolmogorovSmirnov):

    def __init__(
                    self,
                    sample: Optional[Union[np.ndarray, list,]] = None,
                    a_lambda: Optional[Union[float, int,]] = None,
                    N_0: Optional[int] = None,
                    t_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    t_0: Optional[Union[float, int,]] = None,
                    t_1: Optional[Union[float, int,]] = None,
                    Delta_t: Optional[Union[float, int,]] = None,
                    app_PIT: Optional[bool] = None,
                    ):

        KolmogorovSmirnovMeta.__init__(self)
        TestWrapper.__init__(self)
        self._point_proc = True

        self.set_sample(sample=sample, a_lambda=a_lambda,
                        N_0=N_0,
                        t_bounds=t_bounds, t_0=t_0, t_1=t_1, Delta_t=Delta_t,
                        app_PIT=app_PIT,
                        reset=True)

    def set_sample(
                    self,
                    sample: Optional[Union[np.ndarray, list,]] = None,
                    a_lambda: Optional[Union[float, int,]] = None,
                    N_0: Optional[int] = None,
                    t_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    t_0: Optional[Union[float, int,]] = None,
                    t_1: Optional[Union[float, int,]] = None,
                    Delta_t: Optional[Union[float, int,]] = None,
                    set_res: Optional[bool] = None,
                    app_PIT: Optional[bool] = None,
                    reset: Optional[bool] = None,
                    ):

        super().set_sample(sample=sample, set_res=False, reset=reset)
        self.set_D(a_lambda=a_lambda,
                   N_0=N_0,
                   t_bounds=t_bounds, t_0=t_0, t_1=t_1, Delta_t=Delta_t,
                   reset=reset)

        if set_res is None:
            set_res = True

        if set_res:
            if self.sample is not None:
                self.set_res(app_PIT=app_PIT)

    def set_D(
                self,
                a_lambda: Optional[Union[float, int,]] = None,
                N_0: Optional[int] = None,
                Delta_N: Optional[int] = None,
                t_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                t_0: Optional[Union[float, int,]] = None,
                t_1: Optional[Union[float, int,]] = None,
                Delta_t: Optional[Union[float, int,]] = None,
                reset: Optional[bool] = None,
                ):

        if N_0 is None:
            N_0 = 0

        if Delta_N is None:
            Delta_N = self.calc_N_sample_nan()

        self.set_D_0(a_lambda=a_lambda,
                     N_0=N_0, Delta_N=Delta_N,
                     t_bounds=t_bounds, t_0=t_0, t_1=t_1, Delta_t=Delta_t,
                     reset=reset)
        self.set_D_T(reset=reset)

    def set_D_0(
                self,
                a_lambda: Optional[Union[float, int,]] = None,
                N_0: Optional[int] = None,
                Delta_N: Optional[int] = None,
                t_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                t_0: Optional[Union[float, int,]] = None,
                t_1: Optional[Union[float, int,]] = None,
                Delta_t: Optional[Union[float, int,]] = None,
                reset: Optional[bool] = None,
                ):

        if N_0 is None:
            N_0 = 0

        if Delta_N is None:
            Delta_N = self.calc_N_sample_nan()

        if utils.meta.default_reset(reset=reset):
            self.D_0 = DistN(
                                a_lambda=a_lambda,
                                N_0=N_0, Delta_N=Delta_N,
                                t_bounds=t_bounds, t_0=t_0, t_1=t_1, Delta_t=Delta_t,
                                )
        else:
            self.D_0.set_all(
                                a_lambda=a_lambda,
                                N_0=N_0, Delta_N=Delta_N,
                                t_bounds=t_bounds, t_0=t_0, t_1=t_1, Delta_t=Delta_t,
                                )

    def set_D_T(
                self,
                a_lambda: Optional[Union[float, int,]] = None,
                Delta_N: Optional[int] = None,
                t_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                a_lambda_T: Optional[Union[float, int,]] = None,
                N_0_T: Optional[int] = None,
                t_0_T: Optional[Union[float, int,]] = None,
                Delta_t_T: Optional[Union[float, int,]] = None,
                reset: Optional[bool] = None,
                ):

        if a_lambda is None:
            a_lambda = self.D_0.a_lambda

        if Delta_N is None:
            Delta_N = self.D_0.N_bounds.Delta()

        if t_bounds is None:
            t_bounds = self.D_0.t_bounds

        if a_lambda_T is None:
            a_lambda_T = 1.0
        if N_0_T is None:
            N_0_T = self.D_0.N_bounds.bounds[0]
        if t_0_T is None:
            t_0_T = 0.0
        Delta_t_T = t_bounds.Delta() * (a_lambda/a_lambda_T)

        if utils.meta.default_reset(reset=reset):
            self.D_T = DistN(a_lambda=a_lambda_T,
                             N_0=N_0_T, Delta_N=Delta_N,
                             t_0=t_0_T, Delta_t=Delta_t_T,
                             )
        else:
            self.D_T.set_all(a_lambda=a_lambda_T,
                             N_0=N_0_T, Delta_N=Delta_N,
                             t_0=t_0_T, Delta_t=Delta_t_T,
                             )

    def set_res(
                self,
                app_PIT: Optional[bool] = None,
                ) -> KolmogorovSmirnovResult:

        self.res = KolmogorovSmirnovResult(
                        sample=self.get_sample(app_PIT=app_PIT),
                        cdf_0=lambda x: self.get_cdf_0(app_PIT=app_PIT)(x, test=True),
                        )

        return self.res

    def calc_C_ppf(
                    self,
                    x: Optional[Union[np.ndarray, list, float, int,]] = None,
                    q: Union[np.ndarray, float,] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    N: Optional[int] = None,
                    app_PIT: Optional[bool] = None,
                    x_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    x_bounds_alpha: Optional[float] = None,
                    xs_num: Optional[int] = None,
                    xs_delta: Optional[Union[float, int,]] = None,
                    ) -> np.ndarray:

        x = self.calc_x(x=x, x_bounds=x_bounds, alpha=x_bounds_alpha, app_PIT=app_PIT,
                        xs_num=xs_num, xs_delta=xs_delta)

        return calc_KS_C_ppf(x=x, q=q, alpha=alpha, N=self.calc_N_sample(N=N),
                             cdf=self.get_cdf_0(app_PIT=app_PIT), point_proc=True)


# =====================================
# FUNCTIONS
# =====================================


def calc_KS_D_crit(
                    alpha: Union[np.ndarray, float,],
                    N: Union[np.ndarray, int,],
                    ) -> Union[np.ndarray, float,]:

    return np.sqrt(np.log(2/alpha)/(2*N))


def calc_KS_adj(
                N: Union[np.ndarray, int,],
                ) -> Union[np.ndarray, float,]:

    return (N+0.12+(0.11/N))/N


def calc_KS_C_alpha(
                    x: Optional[Union[np.ndarray, list, float, int,]] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    N: Optional[int] = None,
                    cdf: Optional[Callable] = None,
                    point_proc: Optional[bool] = None,
                    norm_point: Optional[bool] = None,
                    loc: Optional[Union[float, int,]] = None,
                    scale: Optional[Union[float, int,]] = None,
                    cap_0: Optional[bool] = None,
                    cap_1: Optional[bool] = None,
                    ) -> np.ndarray:

    if point_proc is None:
        point_proc = False

    if norm_point is None:
        norm_point = True

    if cap_0 is None:
        cap_0 = True
    if cap_1 is None:
        cap_1 = not point_proc

    N = default_N(N=N)

    if loc is None:
        loc = 0.0
    if scale is None:
        if point_proc and not norm_point:
            scale = N
        else:
            scale = 1.0

    if alpha is None:
        alpha = 0.05

    if cdf is None:
        cdf = scipy.stats.norm().cdf

    sign = np.array([-1, 1])

    F = cdf(x) + scale*np.multiply.outer(sign,
                                         calc_KS_D_crit(alpha=alpha, N=N)
                                         * calc_KS_adj(N=N))

    if cap_0:
        F[F < 0] = loc
    if cap_1:
        F[F > scale] = scale

    return F


def calc_KS_C_ppf(
                    x: Optional[Union[np.ndarray, list, float, int,]] = None,
                    q: Union[np.ndarray, float,] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    N: Optional[int] = None,
                    cdf: Optional[Callable] = None,
                    point_proc: Optional[bool] = None,
                    norm_point: Optional[bool] = None,
                    loc: Optional[Union[float, int,]] = None,
                    scale: Optional[Union[float, int,]] = None,
                    cap_0: Optional[bool] = None,
                    cap_1: Optional[bool] = None,
                    ) -> np.ndarray:

    tail = "both"

    if point_proc is None:
        point_proc = False

    if norm_point is None:
        norm_point = True

    if cap_0 is None:
        cap_0 = not point_proc
    if cap_1 is None:
        cap_1 = not point_proc

    N = default_N(N=N)

    if loc is None:
        loc = 0.0
    if scale is None:
        if point_proc and not norm_point:
            scale = N
        else:
            scale = 1.0

    if q is None:
        if alpha is None:
            alpha = default_alpha()
        q = alpha_to_q(alpha=alpha, tail=tail)
    salpha = q_to_salpha(q=q, tail=tail)

    if cdf is None:
        if point_proc:
            cdf = calc_point_proc_N_cdf(norm_point=norm_point, N=N)
        else:
            cdf = scipy.stats.norm().cdf

    F = np.add.outer(scale * np.sign(salpha) * calc_KS_D_crit(alpha=np.abs(salpha), N=N)
                     * calc_KS_adj(N=N), cdf(x))

    if cap_0:
        F[F < loc] = loc
    if cap_1:
        F[F > scale] = scale

    return F


def KS_test_Poisson_scipy(
                            tilde_t_k, N: Optional[int] = None,
                            method: Optional[str] = None,
                            ) -> scipy.stats._stats_py.KstestResult:

    if method is None:
        method = "auto"

    size = np.size(tilde_t_k)
    if N is None:
        N = size
    if size < 1:
        tilde_t_k = [np.nan]

    args = {
            "rvs": tilde_t_k,
            "cdf": calc_point_proc_N_cdf(norm_point=True, N=N),
            # "method": method,
            }

    return scipy.stats.kstest(**args)


def calc_point_proc_N_cdf(
                            norm_point: Optional[bool] = None,
                            N: Optional[int] = None,
                            ) -> Callable:

    if norm_point is None:
        norm_point = True

    N = default_N(N=N)

    if norm_point:
        def cdf(x): return x/N
    else:
        def cdf(x): return x

    return cdf


def calc_N_x(
                sample: Union[np.ndarray, list,],
                x: Optional[Union[np.ndarray, list, float, int,]] = None,
                x_0: Optional[Union[float, int,]] = None,
                x_bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                x_min: Optional[Union[float, int]] = None,
                x_max: Optional[Union[float, int]] = None,
                num: Optional[int] = None,
                delta: Optional[float] = None,
                ) -> np.ndarray:
    """
    Calculate empirical sample count.
    """

    if x is None:
        x_bounds = utils.bounds.Bounds(bounds=x_bounds, a=x_min, b=x_max)
        x = x_bounds.xs(num=num, delta=delta)
    if np.isscalar(x):
        x = [x]

    if x_0 is None:
        N_0 = 0
    else:
        N_0 = np.sum(sample < x_0)

    bins = np.concatenate([[-np.inf], x])
    N_x = np.histogram(sample, bins)[0].cumsum() - N_0

    return N_x


def calc_x_y_step(
                    x: Optional[Union[np.ndarray, list,]] = None,
                    y: Optional[Union[np.ndarray, list,]] = None,
                    x_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    y_0: Optional[Union[float, int,]] = None,
                    y_scale: Optional[Union[float, int,]] = None,
                    y_norm: Optional[bool] = None,
                    ) -> Tuple[np.ndarray, np.ndarray,]:

    N = np.size(x)

    if y_norm is None:
        y_norm = False

    if y_scale is None:
        if y_norm:
            y_scale = 1/N
        else:
            y_scale = 1.0

    if x_bounds is None:
        x_bounds = x[[0, -1]]

    if y_0 is None:
        y_0 = 0
    if y is None:
        y = y_0 + y_scale*(1 + np.arange(N))

    x_step = np.concatenate([[x_bounds[0]], np.repeat(x, 2), [x_bounds[-1]]])
    y_step = np.repeat(np.concatenate([[y_0], y]), 2)

    return x_step, y_step


# Defaults
# -------------------------------------


def default_N_x(
                N_x: Optional[int] = None,
                ) -> int:

    if N_x is None:
        N_x = 401
    return N_x
