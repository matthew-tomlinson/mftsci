"""
Functions module for point processes in continuous time.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable, Tuple


# External packages
import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Internal packages
from mftsci import utils
from mftsci.utils.unit import Unit
from mftsci.point_proc.cont import defaults
from mftsci.point_proc.cont.core import gen_lambda_const as _gen_lambda_const


# =====================================
# CLASSES
# =====================================

class PointProcess():
    """
    Point process in local time bounds.
    """

    def __init__(
                    self,
                    name: Optional[str] = None,
                    t_bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                    t_0: Optional[Union[float, int,]] = None,
                    t_1: Optional[Union[float, int,]] = None,
                    unit: Optional[Unit] = None,
                    lambda_fn: Optional[Callable] = None,
                    ):

        self.set_name(name=name)
        self.set_t_bounds(t_bounds=t_bounds, t_0=t_0, t_1=t_1, unit=unit)
        self.set_lambda_fn(lambda_fn=lambda_fn)
        self.theta = None
        self.theta_bounds = None

    def set_name(
                    self,
                    name: str,
                    ):
        """
        Set name of point process.
        """

        self.name = name

    def set_t_bounds(
                        self,
                        t_bounds: Optional[Union[
                                                    utils.bounds.Bounds,
                                                    np.ndarray,
                                                    list,
                                                    ]] = None,
                        t_0: Optional[Union[float, int,]] = None,
                        t_1: Optional[Union[float, int,]] = None,
                        unit: Optional[Unit] = None,
                        ):
        """
        Set local bounds in time.
        """

        self.t_bounds = utils.bounds.Bounds(bounds=t_bounds, a=t_0, b=t_1, unit=unit)

    def set_lambda_fn(
                        self,
                        lambda_fn: Callable,
                        Delta_Lambda_fn: Optional[Callable] = None,
                        ):
        """
        Set intensity function (in dimensionless units).
        """
        self._lambda_fn = defaults.lambda_fn(lambda_fn)
        self.set_Delta_Lambda_fn(Delta_Lambda_fn=Delta_Lambda_fn)
        self.set_tilde_a_lambda()

    def set_Delta_Lambda_fn(
                            self,
                            Delta_Lambda_fn: Optional[Callable] = None,
                            ):
        """
        Set integrated intensity function
        """

        if Delta_Lambda_fn is None:
            def Delta_Lambda_fn(t_0, t_1):

                # Delta_redim = t_1 - t_0
                # Delta_dedim = self.t_bounds.unit.dedim(t_1) - self.t_bounds.unit.dedim(t_0)   # noqa: E501, W505
                # factor = Delta_dedim/Delta_redim

                return scipy.integrate.quad(self._lambda_fn, a=t_0, b=t_1)[0]

        self._Delta_Lambda_fn = utils.meta.vectorize()(Delta_Lambda_fn)

    def lambda_fn(
                    self,
                    t: Optional[Union[float, int,]] = None,
                    ) -> Union[np.ndarray, float]:
        f"""
        Intensity function (in dimensionless units).
        $$
        \\lambda{{(t)}}
        $$
        """     # noqa: F541

        t = self.t_bounds.unit.dedim(t)

        return self._lambda_fn(t)

    def Delta_Lambda_fn(
                        self,
                        t_0: Optional[Union[float, int,]] = None,
                        t_1: Optional[Union[float, int,]] = None,
                        ) -> Union[np.ndarray, float]:
        f"""
        Integrated intensity function (in dimensionless units) evaluated between t_0
        and t_1.
        $$
        \\Delta{{\\lambda}}{{(t_0, t_1)}}
        =
        \\int_{{t_0}}^{{t_1}}{{
        \\lambda{{(t')}}
        d{{t'}}
        }}
        $$
        """     # noqa: F541

        t_0 = self.t_bounds.unit.dedim(t_0)
        t_1 = self.t_bounds.unit.dedim(t_1)

        return self._Delta_Lambda_fn(t_0=t_0, t_1=t_1)

    def Lambda_fn(
                    self,
                    t: Optional[Union[float, int,]] = None,
                    ) -> Union[np.ndarray, float]:
        f"""
        Integrated intensity function (in dimensionless units) evaluated between 0 and
        t.
        $$
        \\Lambda{{(t)}}
        =
        \\int_{{0}}^{{t}}{{
        \\lambda{{(t')}}
        d{{t'}}
        }}
        $$
        """     # noqa: F541

        t = self.t_bounds.unit.dedim(t)

        return self._Delta_Lambda_fn(t_0=0.0, t_1=t)

    def calc_tilde_a_lambda(
                            self,
                            ) -> float:
        f"""
        Calculate expected arrival intensity (in dimensionless units) between local
        bounds t_0 and t_1.
        $$
        \\tilde{{a}}_{{\\lambda}}
        =
        \\frac{{
        \\int_{{t_0}}^{{t_1}}{{
        \\lambda{{(t')}}
        d{{t'}}
        }}
        }}{{
        t_1 - t_0
        }}
        $$
        """     # noqa: F541

        return self.Delta_Lambda_fn(
                        t_0=self.t_bounds.bounds[0],
                        t_1=self.t_bounds.bounds[1],
                        )/self.t_bounds.Delta(dedim=True)

    def _calc_log_L(
                    self,
                    t_bounds,
                    events_t
                    ):

        return -self._Delta_Lambda_fn(*t_bounds)\
                    + np.sum(np.log(self._lambda_fn(events_t)))

    def calc_log_L(
                    self,
                    t_bounds=None,
                    events_t=None,
                    dedim=None,
                    ):

        if dedim is None:
            dedim = True
        if events_t is None:
            events_t = self.events["t"]
        if t_bounds is None:
            t_bounds = self.t_bounds.get_bounds()

        if dedim:
            t_bounds = self.t_bounds.unit.dedim(t_bounds)
            events_t = self.t_bounds.unit.dedim(events_t)

        return self._calc_log_L(t_bounds, events_t)

    def calc_log_L_theta(
                            self,
                            theta=None,
                            t_bounds=None,
                            events_t=None,
                            dedim=None,
                            ):

        if theta is not None:
            self.theta = theta

        return self.calc_log_L(t_bounds=t_bounds, events_t=events_t, dedim=dedim)

    def set_tilde_a_lambda(
                            self,
                            ):
        """
        Set expected arrival intensity (in dimensionless units)
        """

        self.tilde_a_lambda = self.calc_tilde_a_lambda()

    def set_mark_dist(
                        self,
                        mark_dist: Optional[
                            scipy.stats._distn_infrastructure.rv_continuous_frozen
                            ] = None,
                        ):
        """
        Set mark distribution $\\mathcal{{D}}$.
        """

        mark_dist = defaults.mark_dist()

        self.mark_dist = mark_dist

    def sim_marks(self):
        """
        Simulate marks by sampling from mark distribution.
        """

        self.events["M"] = self.mark_dist.rvs(self.events["t"].size)

    def MC_sim_t_k(
                    self,
                    lambda_max_min: Optional[Union[float, int,]] = None,
                    ):
        """
        Simulate (via Monte Carlo) arrival times in local time bounds by subsampling
        uniform random points in a 2D area.
        """

        # Calculate expected arrival intensity
        self.set_tilde_a_lambda()

        # Simulate point process by generating uniform random points in a 2D area
        # where the number of random points is a random integer sampled from
        # Poisson(mu=AREA)
        self.MC_Pt2D = gen_points_2D(lambda_fn=self.lambda_fn,
                                     bounds=self.t_bounds,
                                     lambda_mm=lambda_max_min)

        # Filter arrival times from points_2D by selecting y < lambda_fn(x)
        self.events = MC_sim_point_process_t(points_2D=self.MC_Pt2D).to_frame()

    def calc_N_t(
                    self,
                    t: Optional[Union[float, int,]] = None,
                    num: Optional[int] = None,
                    delta: Optional[Union[float, int,]] = None,
                    ) -> Union[np.ndarray, float]:
        """
        Calculate event count time series.
        """

        if t is None:
            delta = defaults.delta(delta)
            t = self.t_bounds.xs(num=num, delta=delta)

        t = utils.arr.convert(t)

        bins = np.concatenate([[-np.inf], t])
        N_t = np.histogram(self.events["t"].values,
                           bins)[0].cumsum() - (self.events["t"].values < 0).sum()

        return N_t

    def plot_t_k(
                    self,
                    ax: matplotlib.axes._axes.Axes,
                    plot_scatter_args: Optional[tuple] = None,
                    plot_scatter_kwargs: Optional[dict] = None,
                    t_dedim: Optional[bool] = None,
                    ):
        """
        Plot arrival times.
        """
        t_dedim = defaults.t_dedim(t_dedim)

        def t_dedim_fn(t, **kwargs):
            return self.t_bounds.unit.dedim(x=t, do=t_dedim, **kwargs)

        ax = plot_t_k(ax=ax, t_k=t_dedim_fn(self.events["t"]),
                      plot_scatter_args=plot_scatter_args,
                      plot_scatter_kwargs=plot_scatter_kwargs)

        # Limits and labels
        ax.set_xlim(t_dedim_fn(self.t_bounds.bounds))
        ax.set_xlabel(f"$t{self.t_bounds.unit.get_symbol_unitsuff(app_unit_suff=t_dedim)}$")    # noqa: E501

    def MC_est_minmax(
                        self,
                        mm_N: Optional[int] = None,
                        mm_mode: Optional[str] = None,
                        ) -> np.ndarray:
        """
        Estimate (via Monte Carlo) maximum and minimum of intensity function in local
        time bounds.
        """

        return MC_est_f_minmax(
                                lambda_fn=self.set_lambda_fn,
                                bounds=self.t_bounds.bounds,
                                mm_N=mm_N, mm_mode=mm_mode
                                )

    def plot_MC_t_k(
                    self,
                    ax: matplotlib.axes._axes.Axes,
                    ts_num: Optional[int] = None,
                    plot_lambda_args: Optional[tuple] = None,
                    plot_lambda_kwargs: Optional[dict] = None,
                    plot_accept_args: Optional[tuple] = None,
                    plot_accept_kwargs: Optional[dict] = None,
                    plot_reject_args: Optional[tuple] = None,
                    plot_reject_kwargs: Optional[dict] = None,
                    t_dedim: Optional[bool] = None,
                    resid: Optional[bool] = None,
                    resid_lambda_fn: Optional[Callable] = None,
                    ):
        """
        Plot MC simulation of point process.
        """

        if ax is None:
            fig, ax = plt.subplots()

        if resid is None:
            if resid_lambda_fn is not None:
                resid = True
            else:
                resid = defaults.resid(resid)
        if resid:
            t_dedim = True
            if resid_lambda_fn is None:
                resid_lambda_fn = self._lambda_fn

            @utils.meta.vectorize()
            def resid_Lambda(t):
                return scipy.integrate.quad(resid_lambda_fn, a=0.0, b=t)[0]

            lambda_fn = gen_lambda_const(1.0)
            t_symb = "\\Lambda"
            t_unit = ""
        else:
            t_dedim = defaults.t_dedim(t_dedim)
            resid_lambda_fn = gen_lambda_const(1.0)
            resid_Lambda = utils.meta.return_x

            lambda_fn = self.lambda_fn
            t_symb = "t"
            t_unit = self.t_bounds.unit.get_symbol_unitsuff(app_unit_suff=t_dedim)

        def t_dedim_fn(t, **kwargs):
            return self.t_bounds.unit.dedim(x=t, do=t_dedim, **kwargs)

        # Set time linspace
        ts = self.t_bounds.xs(num=ts_num, dedim=False)

        # Scatter plot of random MC points
        # ...less than intensity function (accepted)
        plot_accept_kwargs = utils.meta.set_defaults(
                        _func=ax.scatter,
                        _defaults={
                                    "color": utils.colour.c_vadd_diff(v=1),
                                    # c
                                    "marker": ".",
                                    },
                        args=plot_accept_args,
                        kwargs=plot_accept_kwargs,
                        )
        ax.scatter(
            resid_Lambda(t_dedim_fn(self.MC_Pt2D["t"][self.MC_Pt2D["accept"]])),
            self.MC_Pt2D["lambda"][self.MC_Pt2D["accept"]]
            / resid_lambda_fn(t_dedim_fn(self.MC_Pt2D["t"][self.MC_Pt2D["accept"]])),
            **plot_accept_kwargs,
            )
        # ...greater than intensity function (rejected)
        plot_reject_kwargs = utils.meta.set_defaults(
                                _func=ax.scatter,
                                _defaults={
                                    "color": utils.colour.c_grad_cycle(N_c=1, N_grad=2),
                                    # c
                                    "marker": ".",
                                    },
                                args=plot_reject_args,
                                kwargs=plot_reject_kwargs,
                                )
        ax.scatter(
            resid_Lambda(t_dedim_fn(self.MC_Pt2D["t"][~self.MC_Pt2D["accept"]])),
            self.MC_Pt2D["lambda"][~self.MC_Pt2D["accept"]]
            / resid_lambda_fn(t_dedim_fn(self.MC_Pt2D["t"][~self.MC_Pt2D["accept"]])),
            **plot_reject_kwargs,
            )

        # Plot intensity function
        plot_lambda_kwargs = utils.meta.set_defaults(
                                            _func=ax.plot,
                                            args=plot_lambda_args,
                                            kwargs=plot_lambda_kwargs,
                                            )

        ax.plot(
                resid_Lambda(t_dedim_fn(ts)),
                lambda_fn(ts),
                **plot_lambda_kwargs,
                )

        # Legend
        ax.legend(
                    [ax.get_children()[k] for k in [2, 0]],
                    [f"$\\lambda{{\\left({t_symb}\\right)}}$",
                     f"$t_{{k}}$"],  # noqa: F541
                    loc="upper right",
                    )

        # Labels and Limits
        ax.set_ylabel(f"$\\lambda{{\\left({t_symb}\\right)}}{self.t_bounds.unit.get_symbol_unitsuff(inv=True)}$")    # noqa: E501
        ax.set_ylim(bottom=0)

        ax.set_xlim(resid_Lambda(self.t_bounds.get_bounds(dedim=t_dedim)))
        ax.set_xlabel(f"${t_symb}{t_unit}$")    # noqa: E501

    def plot_hist_Delta_t_Poisson(
                                    self,
                                    ax: matplotlib.axes._axes.Axes,
                                    x_bounds: Optional[Union[
                                                                utils.bounds.Bounds,
                                                                np.ndarray,
                                                                list,
                                                            ]] = None,
                                    ylog: Optional[bool] = None,
                                    plot_hist_args: Optional[tuple] = None,
                                    plot_hist_kwargs: Optional[dict] = None,
                                    plot_f_args: Optional[tuple] = None,
                                    plot_f_kwargs: Optional[dict] = None,
                                    t_dedim: Optional[bool] = None,
                                    resid: Optional[bool] = None,
                                    resid_lambda_fn: Optional[Callable] = None,
                                    ):
        """
        Plot histogram of interarrival times Delta_t_k with null: Poisson (time--
        homogeneous) point process.
        """

        if ax is None:
            fig, ax = plt.subplots()

        ylog = defaults.ylog(ylog)

        if resid is None:
            if resid_lambda_fn is not None:
                resid = True
            else:
                resid = defaults.resid(resid)
        if resid:
            t_dedim = True
            if resid_lambda_fn is None:
                resid_lambda_fn = self._lambda_fn

            @utils.meta.vectorize()
            def resid_Lambda(t):
                return scipy.integrate.quad(resid_lambda_fn, a=0.0, b=t)[0]

            _a_lambda = 1.0

            t_symb = "\\Lambda"
            t_unit = ""
            dist_args_symb = "1"
        else:
            t_dedim = defaults.t_dedim(t_dedim)
            resid_lambda_fn = gen_lambda_const(1.0)
            resid_Lambda = utils.meta.return_x

            _a_lambda = self.tilde_a_lambda

            t_symb = "t"
            t_unit = self.t_bounds.unit.get_symbol_unitsuff(app_unit_suff=t_dedim)
            dist_args_symb = "1/\\tilde{{a}}_{{\\lambda}}"

        def t_redim_fn(t, **kwargs):
            return self.t_bounds.unit.redim(x=t, do=not t_dedim, **kwargs)

        # Delta_t_k sample
        sample = np.diff(resid_Lambda(self.t_bounds.unit.dedim(self.events["t"])))
        # Theoretical Delta_t distribution under null: Poisson (time-homogeneous) point
        # process
        sample_H0_dist = scipy.stats.expon(scale=1/_a_lambda)

        # Boundaries
        if x_bounds is None:
            x_bounds = sample_H0_dist.ppf([0, 0.99])
            x_bounds[1] = np.max([x_bounds[1], np.max(sample)])

        # Plot histogram
        bins = np.linspace(*x_bounds, 41)
        plot_hist_kwargs = utils.meta.set_defaults(
                            _func=ax.hist,
                            args=plot_hist_args,
                            kwargs=plot_hist_kwargs,
                            )

        res = ax.hist(t_redim_fn(sample), bins=t_redim_fn(bins), density=False,
                      **plot_hist_kwargs)

        # Plot theoretical distribution PDF
        xs = np.linspace(*x_bounds, 201)
        A = np.diff(bins[:2])[0]*sample.size

        plot_f_kwargs = utils.meta.set_defaults(
                                    _func=ax.plot,
                                    _defaults={"color": utils.colour.c_vadd_diff(v=1)},
                                    args=plot_f_args,
                                    kwargs=plot_f_kwargs,
                                    )
        if False:
            utils.dict.append_if_not_in(plot_f_kwargs, keys_check=["color", "c"],
                                        value=utils.colour.c_vadd_diff(v=1))
        ax.plot(t_redim_fn(xs), A*sample_H0_dist.pdf(xs), **plot_f_kwargs)

        # Legend
        ax.legend(
            [ax.get_children()[k] for k in [bins.size-1]],
            [f"$\\mathrm{{Expon}}{{\\left({dist_args_symb}\\right)}}$ PDF"],     # noqa: E501, F541
            loc="upper right",
            )

        # Labels and limits
        ax.set_ylabel("(Expected) count")
        ax.set_xlabel(f"$\\Delta{{{t_symb}}}{t_unit}$")  # noqa: E501, F541
        if ylog:
            # ax.set_ylim(bottom=A*1E-4)
            # ax.axhline(1, ls=":", color=utils.colour.c_vadd_diff(v=1))
            if not t_dedim:
                res = np.histogram(sample, bins=bins, density=False)
            ax.set_ylim(bottom=A*sample_H0_dist.pdf(
                                    res[1][1+np.where(res[0] > 0)[0][-1]]
                                    ))
            ax.set_yscale("log", nonpositive="clip")

        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(left=t_redim_fn(0))


# =====================================
# FUNCTIONS
# =====================================


def gen_lambda_const(
                        a_lambda: Optional[Union[float, int,]] = None,
                        ) -> Callable:

    a_lambda = defaults.a_lambda(a_lambda)

    return _gen_lambda_const(a_lambda=a_lambda)


def MC_est_f_minmax(
                    f: Optional[Callable] = None,
                    bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                    a: Optional[Union[float, int,]] = None,
                    b: Optional[Union[float, int,]] = None,
                    unit: Optional[Unit] = None,
                    mm_N: Optional[int] = None,
                    mm_mode: Optional[str] = None,
                    ) -> np.ndarray:
    """
    Estimate (via Monte Carlo) minimum and maximum of function within local bounds.
    """

    f = defaults.set_default(val=f, default_call=gen_lambda_const)

    bounds = utils.bounds.Bounds(bounds=bounds, a=a, b=b, unit=unit)

    mm_N = defaults.mm_N(mm_N)

    if mm_mode == "random":
        X = bounds.bounds[0] + bounds.Delta(dedim=False)*scipy.stats.uniform().rvs(mm_N)
    else:
        X = bounds.linspace(num=mm_N+1)

    f_X = f(pd.Series(X))

    return np.array([np.min(f_X), np.max(f_X)])


def gen_points_2D(
                    lambda_fn: Optional[Callable] = None,
                    a_lambda: Optional[Union[float, int,]] = None,
                    bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                    a: Optional[Union[float, int,]] = None,
                    b: Optional[Union[float, int,]] = None,
                    unit: Optional[Unit] = None,
                    lambda_mm: Optional[Union[float, int,]] = None,
                    mm_N: Optional[int] = None,
                    mm_mode: Optional[str] = None,
                    ) -> Union[
                                Tuple[np.ndarray, np.ndarray],
                                np.ndarray,
                                ]:

    lambda_mm = defaults.lambda_mm(lambda_mm)

    if lambda_fn is None:
        lambda_fn = gen_lambda_const(a_lambda=a_lambda)

    bounds = utils.bounds.Bounds(bounds=bounds, a=a, b=b, unit=unit)

    # Simulate point process by generating uniform random points in a 2D area
    # where the number of random points is a random integer sampled from
    # Poisson(mu=AREA)
    T_X = bounds.Delta(dedim=True)
    T_Y = MC_est_f_minmax(f=lambda_fn, bounds=bounds, mm_N=mm_N, mm_mode=mm_mode)[1]
    if T_Y < lambda_mm:
        T_Y = lambda_mm
    N = scipy.stats.poisson.rvs(mu=T_X*T_Y)
    rvs = scipy.stats.uniform().rvs((N, 2))
    points_2D = pd.DataFrame({
                        "t": bounds.bounds[0] + bounds.Delta(dedim=False) * rvs[:, 0],
                        "lambda": 0.0 + (T_Y * rvs[:, 1]),
                    }).sort_values("t")

    # Filter points by selecting t where y_point < lambda_fn(t)
    points_2D["f"] = points_2D["lambda"] / lambda_fn(points_2D["t"])
    points_2D["accept"] = points_2D["f"] < 1

    return points_2D


def MC_sim_point_process_t(
                            points_2D: Optional[pd.core.frame.DataFrame] = None,
                            lambda_fn: Optional[Callable] = None,
                            a_lambda: Optional[Union[float, int,]] = None,
                            bounds: Optional[Union[
                                                    utils.bounds.Bounds,
                                                    np.ndarray,
                                                    list,
                                                    ]] = None,
                            a: Optional[Union[float, int,]] = None,
                            b: Optional[Union[float, int,]] = None,
                            unit: Optional[Unit] = None,
                            lambda_mm: Optional[Union[float, int,]] = None,
                            mm_N: Optional[int] = None,
                            mm_mode: Optional[str] = None,
                            ) -> np.ndarray:

    if points_2D is None:
        points_2D = gen_points_2D(
                                    lambda_fn=lambda_fn,
                                    a_lambda=a_lambda,
                                    bounds=bounds, a=a, b=b, unit=unit,
                                    lambda_mm=lambda_mm, mm_N=mm_N, mm_mode=mm_mode,
                                    )

    # Filter arrival times from points_2D by selecting y < lambda_fn(x)
    t_k = points_2D["t"][points_2D["f"] < 1]
    # Reindex arrival times
    t_k.index = np.arange(len(t_k))

    return t_k


def plot_t_k(
                ax: matplotlib.axes._axes.Axes,
                t_k: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                plot_scatter_args: Optional[tuple] = None,
                plot_scatter_kwargs: Optional[dict] = None,
                ) -> matplotlib.axes._axes.Axes:
    """
    Plot arrival times.
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Modify axis ticks
    ax.tick_params(left=False, right=False, labelleft=False)

    # Scatter plot of arrival times along t
    plot_scatter_kwargs = utils.meta.set_defaults(
                                                _func=ax.scatter,
                                                _defaults={"marker": "|"},
                                                args=plot_scatter_args,
                                                kwargs=plot_scatter_kwargs,
                                                )
    ax.scatter(t_k, np.zeros(np.size(t_k)), **plot_scatter_kwargs)

    # Limits and labels
    ax.set_xlabel("t")

    return ax
