"""
Module for point processes.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Callable, Tuple

# External packages
import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Internal packages
from mftsci import utils


# =====================================
# CLASSES
# =====================================


class PointProcess():

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
                    lambda_fn: Optional[Callable] = None,
                    ):

        self.set_name(name=name)
        self.set_t_bounds(t_bounds=t_bounds, t_0=t_0, t_1=t_1)
        self.set_lambda_fn(lambda_fn=lambda_fn)

    def set_name(
                    self,
                    name: str,
                    ):

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
                        ):

        self.t_bounds = utils.bounds.Bounds(bounds=t_bounds, a=t_0, b=t_1)

    def set_lambda_fn(
                        self,
                        lambda_fn: Callable,
                        Delta_Lambda_fn: Optional[Callable] = None,
                        ):

        if lambda_fn is None:
            lambda_fn = gen_lambda_const()

        self._lambda_fn = lambda_fn
        self.set_Delta_Lambda_fn(Delta_Lambda_fn=Delta_Lambda_fn)
        self.set_tilde_a_lambda()

    def set_Delta_Lambda_fn(
                            self,
                            Delta_Lambda_fn: Optional[Callable] = None,
                            ):

        if Delta_Lambda_fn is None:
            def Delta_Lambda_fn(t_0, t_1):
                return scipy.integrate.quad(self._lambda_fn, a=t_0, b=t_1)[0]

        self._Delta_Lambda_fn = Delta_Lambda_fn

    def lambda_fn(
                    self,
                    t: Optional[Union[float, int,]] = None,
                    ) -> Union[np.ndarray, float]:

        if t is None:
            t = 0.0

        return self._lambda_fn(t)

    def Delta_Lambda_fn(
                        self,
                        t_0: Optional[Union[float, int,]] = None,
                        t_1: Optional[Union[float, int,]] = None,
                        ) -> Union[np.ndarray, float]:

        if t_0 is None:
            t_0 = 0.0

        if t_1 is None:
            t_1 = 1.0

        return self._Delta_Lambda_fn(t_0=0.0, t_1=t_1)

    def Lambda_fn(
                    self,
                    t: Optional[Union[float, int,]] = None,
                    ) -> Union[np.ndarray, float]:

        return self.Delta_Lambda_fn(t_0=0.0, t_1=t)

    def calc_tilde_a_lambda(
                            self,
                            ) -> float:

        return self.Delta_Lambda_fn(t_0=self.t_bounds.bounds[0],
                                    t_1=self.t_bounds.bounds[1])/self.t_bounds.Delta()

    def set_tilde_a_lambda(
                            self,
                            ):

        self.tilde_a_lambda = self.calc_tilde_a_lambda()

    def set_mark_dist(
                        self,
                        mark_dist: Optional[
                            scipy.stats._distn_infrastructure.rv_continuous_frozen
                            ] = None,
                        ):

        if mark_dist is None:
            mark_dist = scipy.stats.expon()

        self.mark_dist = mark_dist

    def sim_marks(self):

        self.events["M"] = self.mark_dist.rvs(self.events["t"].size)

    def MC_sim_t_k(
                    self,
                    lambda_max_min: Optional[Union[float, int,]] = None,
                    ):

        # Calculate expected arrival intensity
        self.set_tilde_a_lambda()

        # Simulate point process by generating uniform random points in a 2D area
        # where the number of random points is a random integer sampled from
        # Poisson(mu=AREA)
        self.MC_Pt2D = gen_points_2D(f=self.lambda_fn,
                                     bounds=self.t_bounds.bounds,
                                     T_Y_min=lambda_max_min,
                                     ret_bool=False, sort=True)

        # Filter arrival times from points_2D by selecting y < lambda_fn(x)
        self.MC_PtBool = self.MC_Pt2D[1, :] < self.lambda_fn(self.MC_Pt2D[0, :])

        # Save selected arrival times in a DataFrame column
        self.events = pd.DataFrame({"t": self.MC_Pt2D[0, self.MC_PtBool]})

        # Save interarrival times in a DataFrame column
        self.events["Delta_t"] = self.events["t"].diff()

    def calc_N_t(
                    self,
                    t: Optional[Union[float, int,]] = None,
                    num: Optional[int] = None,
                    delta: Optional[Union[float, int,]] = None,
                    ) -> Union[np.ndarray, float]:

        if t is None:
            if delta is None:
                delta = 1.0
            t = self.t_bounds.xs(num=num, delta=delta)
        if np.isscalar(t):
            t = [t]

        bins = np.concatenate([[-np.inf], t])
        N_t = np.histogram(self.events["t"].values,
                           bins)[0].cumsum() - (self.events["t"].values < 0).sum()

        return N_t

    def plot_t_k(
                    self,
                    ax: matplotlib.axes._axes.Axes,
                    plot_scatter_args: Optional[tuple] = None,
                    plot_scatter_kwargs: Optional[dict] = None,
                    ):

        ax = plot_t_k(ax=ax, t_k=self.events["t"], lot_scatter_args=plot_scatter_args,
                      plot_scatter_kwargs=plot_scatter_kwargs)

        # Limits and labels
        ax.set_xlim(self.t_bounds.bounds)

    def MC_est_minmax(
                        self,
                        mm_N: Optional[int] = None,
                        mm_mode: Optional[str] = None,
                        ) -> np.ndarray:

        return MC_est_minmax(f=self.set_lambda_fn, bounds=self.t_bounds.bounds,
                             mm_N=mm_N, mm_mode=mm_mode)

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
                    ):

        if ax is None:
            fig, ax = plt.subplots()

        # Set time linspace
        ts = self.t_bounds.xs(num=ts_num)

        # Scatter plot of random MC points
        # ...less than intensity function (accepted)
        plot_accept_args, plot_accept_kwargs = \
            utils.meta.default_args_kwargs(plot_accept_args, plot_accept_kwargs)
        utils.dict.append_if_not_in(plot_accept_kwargs, keys_check=["color", "c"],
                                    value=utils.colour.c_vadd_diff(v=1))
        utils.dict.append_if_not_in(plot_accept_kwargs,
                                    keys_check=["marker"], value=".")
        ax.scatter(self.MC_Pt2D[0, self.MC_PtBool], self.MC_Pt2D[1, self.MC_PtBool],
                   *plot_accept_args, **plot_accept_kwargs)
        # ...greater than intensity function (rejected)
        plot_reject_args, plot_reject_kwargs = \
            utils.meta.default_args_kwargs(plot_reject_args, plot_reject_kwargs)
        utils.dict.append_if_not_in(plot_reject_kwargs, keys_check=["color", "c"],
                                    value=utils.colour.c_grad_cycle(N_c=1, N_grad=2))
        utils.dict.append_if_not_in(plot_reject_kwargs,
                                    keys_check=["marker"], value=".")
        ax.scatter(self.MC_Pt2D[0, ~self.MC_PtBool], self.MC_Pt2D[1, ~self.MC_PtBool],
                   *plot_reject_args, **plot_reject_kwargs)

        # Plot intensity function
        plot_lambda_args, plot_lambda_kwargs = \
            utils.meta.default_args_kwargs(plot_lambda_args, plot_lambda_kwargs)
        ax.plot(ts, self.lambda_fn(ts), *plot_lambda_args, **plot_lambda_kwargs)

        # Legend
        ax.legend(
                    [ax.get_children()[k] for k in [2, 0]],
                    [f"$\\lambda{{\\left(t\\right)}}$", f"$t_{{k}}$"],  # noqa: F541
                    loc="upper right",
                    )

        # Labels and Limits
        ax.set_ylabel(f"$\\lambda{{\\left(t\\right)}}$")    # noqa: F541
        ax.set_xlabel("$t$")
        ax.set_ylim(bottom=0)
        ax.set_xlim(self.t_bounds.bounds)

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
                                    ):

        if ax is None:
            fig, ax = plt.subplots()

        if ylog is None:
            ylog = False

        # Delta_t_k sample
        sample = self.events["Delta_t"][1:]
        # Theoretical Delta_t distribution under null: Poisson (time-homogeneous) point
        # process
        sample_H0_dist = scipy.stats.expon(scale=1/self.tilde_a_lambda)

        # Boundaries
        if x_bounds is None:
            x_bounds = sample_H0_dist.ppf([0, 0.99])
            x_bounds[1] = np.max([x_bounds[1], np.max(sample)])

        # Plot histogram
        bins = np.linspace(*x_bounds, 41)
        plot_hist_args, plot_hist_kwargs = \
            utils.meta.default_args_kwargs(plot_hist_args, plot_hist_kwargs)
        res = ax.hist(sample, bins=bins, density=False,
                      *plot_hist_args, **plot_hist_kwargs)

        # Plot theoretical distribution PDF
        xs = np.linspace(*x_bounds, 201)
        A = np.diff(bins[:2])[0]*sample.size
        plot_f_args, plot_f_kwargs = \
            utils.meta.default_args_kwargs(plot_f_args, plot_f_kwargs)
        utils.dict.append_if_not_in(plot_f_kwargs, keys_check=["color", "c"],
                                    value=utils.colour.c_vadd_diff(v=1))
        ax.plot(xs, A*sample_H0_dist.pdf(xs), *plot_f_args, **plot_f_kwargs)

        # Legend
        ax.legend(
            [ax.get_children()[k] for k in [bins.size-1]],
            [f"$\\mathrm{{Expon}}{{\\left(1/\\tilde{{a}}_{{\\lambda}}\\right)}}$ PDF"],     # noqa: E501, F541
            loc="upper right",
            )

        # Labels and limits
        ax.set_ylabel("(Expected) count")
        ax.set_xlabel(f"$\\Delta{{t}}$")    # noqa: F541
        if ylog:
            # ax.set_ylim(bottom=A*1E-4)
            # ax.axhline(1, ls=":", color=utils.colour.c_vadd_diff(v=1))
            ax.set_ylim(bottom=A*sample_H0_dist.pdf(
                                    res[1][1+np.where(res[0] > 0)[0][-1]]
                                    ))
            ax.set_yscale("log", nonpositive="clip")

        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)


# =====================================
# FUNCTIONS
# =====================================


def gen_lambda_const(
                        a_lambda: Optional[Union[float, int,]] = None,
                        ) -> Callable:

    if a_lambda is None:
        a_lambda = 1.0

    def f(t):
        return a_lambda * np.ones_like(t)

    return f


def MC_est_minmax(
                    f: Optional[Callable] = None,
                    a_lambda: Optional[Union[float, int,]] = None,
                    bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                    a: Optional[Union[float, int,]] = None,
                    b: Optional[Union[float, int,]] = None,
                    mm_N: Optional[int] = None,
                    mm_mode: Optional[str] = None,
                    ) -> np.ndarray:

    if f is None:
        f = gen_lambda_const(a_lambda=a_lambda)

    bounds = utils.bounds.calc_bounds(bounds=bounds, a=a, b=b)

    if mm_N is None:
        mm_N = 1000

    if mm_mode == "random":
        X = bounds[0] + (bounds[1]-bounds[0])*scipy.stats.uniform().rvs(mm_N)
    else:
        X = np.linspace(bounds[0], bounds[1], mm_N+1)

    return np.array([np.min(f(X)), np.max(f(X))])


def gen_points_2D(
                    f: Optional[Callable] = None,
                    a_lambda: Optional[Union[float, int,]] = None,
                    bounds: Optional[Union[
                                            utils.bounds.Bounds,
                                            np.ndarray,
                                            list,
                                            ]] = None,
                    a: Optional[Union[float, int,]] = None,
                    b: Optional[Union[float, int,]] = None,
                    T_Y_min: Optional[Union[float, int,]] = None,
                    mm_N: Optional[int] = None,
                    mm_mode: Optional[str] = None,
                    sort: Optional[bool] = None,
                    ret_bool: Optional[bool] = None,
                    ) -> Union[
                                Tuple[np.ndarray, np.ndarray],
                                np.ndarray,
                                ]:

    if sort is None:
        sort = True

    if ret_bool is None:
        ret_bool = True

    if f is None:
        f = gen_lambda_const(a_lambda=a_lambda)

    bounds = utils.bounds.calc_bounds(bounds=bounds, a=a, b=b)

    T_X = np.diff(bounds)[0]
    if T_Y_min is None:
        T_Y_min = 1.0
    T_Y = MC_est_minmax(f, bounds=bounds, mm_N=mm_N, mm_mode=mm_mode)[1]
    if T_Y < T_Y_min:
        T_Y = T_Y_min

    N = scipy.stats.poisson.rvs(mu=T_X*T_Y)

    points_2D = np.transpose(
                                np.array([bounds[0], 0.0])
                                + scipy.stats.uniform().rvs((N, 2))
                                * np.array([T_X, T_Y])
                                )
    if sort:
        points_2D = points_2D[:, np.argsort(points_2D[0, :])]

    if ret_bool:
        points_2D_bool = calc_points_2D_bool(f=f, points_2D=points_2D)

        return points_2D, points_2D_bool
    else:
        return points_2D


def calc_points_2D_bool(
                        f: Optional[Callable] = None,
                        a_lambda: Optional[Union[float, int,]] = None,
                        points_2D: Optional[np.ndarray] = None,
                        bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                        a: Optional[Union[float, int,]] = None,
                        b: Optional[Union[float, int,]] = None,
                        T_Y_min: Optional[Union[float, int,]] = None,
                        mm_N: Optional[int] = None,
                        mm_mode: Optional[str] = None,
                        sort: Optional[bool] = None,
                        ) -> np.ndarray:

    if f is None:
        f = gen_lambda_const(a_lambda=a_lambda)

    if points_2D is None:
        points_2D = gen_points_2D(f=f, bounds=bounds, a=a, b=b,
                                  T_Y_min=T_Y_min, mm_N=mm_N, mm_mode=mm_mode,
                                  sort=sort, ret_bool=False)

    return points_2D[1, :] < f(points_2D[0, :])


def calc_points_in_f(
                        f: Optional[Callable] = None,
                        a_lambda: Optional[Union[float, int,]] = None,
                        points_2D: Optional[np.ndarray] = None,
                        points_2D_bool: Optional[np.ndarray] = None,
                        bounds: Optional[Union[
                                                utils.bounds.Bounds,
                                                np.ndarray,
                                                list,
                                                ]] = None,
                        a: Optional[Union[float, int,]] = None,
                        b: Optional[Union[float, int,]] = None,
                        T_Y_min: Optional[Union[float, int,]] = None,
                        mm_N: Optional[int] = None,
                        sort: Optional[bool] = None,
                        mm_mode: Optional[str] = None,
                        ) -> np.ndarray:

    if f is None:
        f = gen_lambda_const(a_lambda=a_lambda)

    if points_2D is None:
        points_2D = gen_points_2D(f=f, bounds=bounds, a=a, b=b,
                                  T_Y_min=T_Y_min, mm_N=mm_N, mm_mode=mm_mode,
                                  sort=sort, ret_bool=False)

    if points_2D_bool is None:
        points_2D_bool = calc_points_2D_bool(f=f, points_2D=points_2D)

    return points_2D[:, points_2D_bool]


def MC_sim_point_process(
                            f: Optional[Callable] = None,
                            a_lambda: Optional[Union[float, int,]] = None,
                            bounds: Optional[Union[
                                                    utils.bounds.Bounds,
                                                    np.ndarray,
                                                    list,
                                                    ]] = None,
                            a: Optional[Union[float, int,]] = None,
                            b: Optional[Union[float, int,]] = None,
                            T_Y_min: Optional[Union[float, int,]] = None,
                            mm_N: Optional[int] = None,
                            sort: Optional[bool] = None,
                            mm_mode: Optional[str] = None,
                            ) -> np.ndarray:

    return calc_points_in_f(f=f, a_lambda=a_lambda, bounds=bounds, a=a, b=b,
                            T_Y_min=T_Y_min, mm_N=mm_N, mm_mode=mm_mode,
                            sort=True)[0, :]


def plot_t_k(
                ax: matplotlib.axes._axes.Axes,
                t_k: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                plot_scatter_args: Optional[tuple] = None,
                plot_scatter_kwargs: Optional[dict] = None,
                ) -> matplotlib.axes._axes.Axes:

    if ax is None:
        fig, ax = plt.subplots()

    # Modify axis ticks
    ax.tick_params(left=False, right=False, labelleft=False)

    # Scatter plot of arrival times along t
    plot_scatter_args, plot_scatter_kwargs = utils.meta.default_args_kwargs(
                                                plot_scatter_args,
                                                plot_scatter_kwargs,
                                                )
    utils.dict.append_if_not_in(plot_scatter_kwargs, keys_check="marker", value="|")
    ax.scatter(t_k, np.zeros(np.size(t_k)), *plot_scatter_args, **plot_scatter_kwargs)

    # Limits and labels
    ax.set_xlabel("t")

    return ax
