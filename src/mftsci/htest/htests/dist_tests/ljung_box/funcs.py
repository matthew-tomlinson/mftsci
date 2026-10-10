"""
Functions module for Ljung--Box test.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from __future__ import annotations
from typing import Optional, Union

# External packages
import numpy as np  # temp
import scipy  # temp
import pandas as pd     # temp
import matplotlib  # temp
import matplotlib.pyplot as plt     # temp
# from statsmodels.tsa.stattools import acf

# Internal packages
from mftsci.htest.htests.dist_tests.core import *   # noqa: F401, F403
from mftsci.htest.htests.dist_tests.core import ConfInterval, \
                                                TestMeta, TestRes, \
                                                DistTest, Dist, DistN, \
                                                alpha_to_q  # temp
from mftsci import utils    # temp
from mftsci.htest.htests.dist_tests.ljung_box import defaults


# =====================================
# CLASSES
# =====================================


class LjungBoxMeta(TestMeta):

    def __init__(
                    self,
                    ):

        TestMeta.__init__(self,
                          test_long_name="Ljung--Box",
                          test_short_name="LB")


class LjungBox(DistTest):

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
                    H_max: Optional[int] = None,
                    ):

        LjungBoxMeta.__init__(self)

        self.H_max = H_max
        DistTest.__init__(
                            self, sample=sample,
                            D_0=D_0, D_T=D_T,
                            app_PIT=app_PIT, inf_alpha=inf_alpha,
                            )

    def set_H_max(
                    self,
                    H_max: Optional[int] = None,
                    reset: Optional[bool] = None,
                    ):

        if H_max is not None or utils.meta.defaults.reset(reset=reset):
            self.H_max = calc_H_max(x=self.sample, H_max=H_max)

    def set_sample(
                    self,
                    sample: Optional[Union[np.ndarray, list,]] = None,
                    reset: Optional[bool] = None,
                    H_max: Optional[int] = None,
                    set_res: Optional[bool] = None,
                    app_PIT: Optional[bool] = None,
                    ):

        super().set_sample(sample=sample, set_res=False, reset=reset)
        self.set_H_max(H_max=H_max, reset=self.H_max is None)

        set_res = defaults.set_res(set_res)

        if set_res:
            if self.sample is not None:
                self.set_res(app_PIT=app_PIT)

    def calc_h(
                self,
                h: Optional[Union[np.ndarray, list, int,]] = None,
                H: Optional[int] = None,
                ) -> np.ndarray:

        if h is None:
            if H is None:
                H = self.H_max
            h = np.arange(1+H)

        return h

    # def set_res(
    #               self,
    #               sample: Optional[Union[np.ndarray, list,]] = None,
    #               H_max: Optional[int] = None,
    #               app_PIT: Optional[bool] = None,
    #               append_h_0=None,
    #               ):

    def set_res(
                self,
                app_PIT: Optional[bool] = None,
                H_max: Optional[int] = None,
                ) -> LjungBoxResult:

        app_PIT = defaults.app_PIT(app_PIT)

        self.set_H_max(H_max=H_max)

        self.res = LjungBoxResult(sample=self.get_sample(app_PIT=app_PIT),
                                  H_max=self.H_max, app_PIT=app_PIT)

        return self.res

    def calc_C_ppf(
                    self,
                    h: Optional[Union[np.ndarray, list, int,]] = None,
                    H: Optional[int] = None,
                    q: Union[np.ndarray, float,] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    N: Optional[int] = None,
                    ) -> np.ndarray:

        return calc_LB_C_ppf(h=self.calc_h(h=h, H=H), q=q, alpha=alpha,
                             N=self.calc_N_sample(N=N))

    def plot(
                self,
                ax: Optional[matplotlib.axes._axes.Axes] = None,
                app_PIT: Optional[bool] = None,
                H_max: Optional[int] = None,
                H_plot: Optional[int] = None,
                H_plot_skip: Optional[int] = None,
                h_test: Optional[int] = None,
                plot_emp_args: Optional[tuple] = None,
                plot_emp_kwargs: Optional[dict] = None,
                CI_args: Optional[tuple] = None,
                CI_kwargs: Optional[dict] = None,
                ):

        h_test = defaults.h_test(h_test)
        H_plot_skip = defaults.H_plot_skip(H_plot_skip)

        if ax is None:
            fig, ax = plt.subplots()

        app_PIT = defaults.app_PIT(app_PIT=app_PIT)

        # Sample
        self.set_res(H_max=H_max, app_PIT=app_PIT)
        res_df = self.res.calc_res_df()

        # hs = self.calc_h(H=H_plot)
        hs = res_df.index.values

        H_plot = np.min([np.max(hs), self.res.H_max])

        h_test = np.min([h_test, self.res.H_max])

        h_plot_slice = H_plot_skip + np.arange(1+H_plot-H_plot_skip)

        # Plot 95% and 99% confidence intervals
        CI_kwargs = utils.meta.set_defaults(_func=ConfInterval.__init__,
                                            args=CI_args, kwargs=CI_kwargs)
        CI = ConfInterval(**CI_kwargs)
        if CI.N > 0:
            CI_h = hs
            CI_y = self.calc_C_ppf(h=CI_h, alpha=CI.alpha,  N=self.sample.size)
            for k in range(CI.N):
                ax.fill_between(CI_h, y1=CI_y[k, 0, :], y2=CI_y[k, 1, :],
                                color=CI.color[k])
            # Add CI legend
            ax.legend(ax.get_children()[:CI.N], CI.calc_CI_strs(), loc="lower right")

        # Plot ACF
        plot_emp_kwargs = utils.meta.set_defaults(
                                                    _func=ax.bar,
                                                    _defaults={"width": 0.8},
                                                    args=plot_emp_args,
                                                    kwargs=plot_emp_kwargs,
                                                    )
        ax.bar(hs[h_plot_slice], self.res.rho[h_plot_slice], **plot_emp_kwargs)

        # Print KS test p-value on axis
        p_LB = self.res.p[h_test]
        p_str = CI.p_scif_reject(
                                    p=p_LB,
                                    brac_kwargs={"pltf": True},
                                    scif_kwargs={"pltf": True},
                                    )
        ax.annotate(
                    f"$p_{{\\mathrm{{LB}}_{{{h_test}}}}} = {p_str}$",
                    xycoords="axes fraction", xy=[0.02, 0.98],
                    horizontalalignment="left",
                    verticalalignment="top",
                    )

        # Labels and limits
        utils.plot.integer_ticks(ax=ax, xys="x", maj_mult=None, min_mult=None,
                                 app_minor=None, maj_offset=None, min_offset=None)
        ax.set_ylabel("Lag-$h$ ACF")
        ax.set_xlabel("$h$")
        ax.set_ylim([-1, 1])
        ax.set_xlim(np.array([0, 0.5+H_plot]))

        return ax


class LjungBoxResult(TestRes, LjungBoxMeta):

    def __init__(
                    self,
                    sample: Union[np.ndarray, list,],
                    H_max: Optional[int] = None,
                    app_PIT: Optional[bool] = None,
                    ):

        LjungBoxMeta.__init__(self)
        self.app_PIT = app_PIT
        self.run(sample=sample, H_max=H_max)

    def __defaultkeys__(
                        self,
                        ) -> list:

        return ["rho", "stat", "p"]

    def calc_res_index(
                        self,
                        ) -> pd.core.indexes.base.Index:

        # h_s = np.arange(1+self.H_max)
        # if not self.append_h_0:
        #     h_s = h_s[1:]

        return pd.Index(np.arange(1+self.H_max), name="h")

    def calc_res_df(
                    self,
                    keys: Optional[Union[np.ndarray, list, str,]] = None,
                    keys_in: Optional[Union[np.ndarray, list, str,]] = None,
                    keys_rm: Optional[Union[np.ndarray, list, str,]] = None,
                    ind_slice: Optional[Union[
                                                pd.core.indexes.base.Index,
                                                np.ndarray,
                                                list,
                                                object,
                                                ]] = None,
                    flatten: Optional[bool] = None,
                    ) -> pd.core.frame.DataFrame:

        if flatten:
            if ind_slice is None:
                ind_slice = slice(1, 1+self.H_max)

        return super().calc_res_df(keys=keys, keys_in=keys_in, keys_rm=keys_rm,
                                   ind_slice=ind_slice, flatten=flatten)

    def run(
            self,
            sample: Union[np.ndarray, list, None,],
            H_max: Optional[int] = None,
            ):

        self.H_max = calc_H_max(x=sample, H_max=H_max)

        if sample is not None:
            acf = utils.meta.require_optional(
                        module="statsmodels.tsa.stattools",
                        attr="acf"
                        )
            (
                self.rho, self.stat, self.p
                ) = acf(
                            x=sample, nlags=self.H_max,
                            qstat=True, adjusted=False,
                            )

            for attr in ["stat", "p"]:
                self.__dict__[attr] = np.concatenate([[np.nan], self.__dict__[attr]])
        else:
            for attr in self.__defaultkeys__():
                setattr(self, attr, np.repeat(None, 1+self.H_max))


# =====================================
# FUNCTIONS
# =====================================


def calc_LB_C_ppf(
                    h: Optional[Union[np.ndarray, list, int,]] = None,
                    q: Union[np.ndarray, float,] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    N: Optional[int] = None,
                    ) -> np.ndarray:

    tail = defaults.tail()
    h = defaults.h(h)

    if N is None:
        N = defaults.N(N)
        N = np.max([N,  np.max(h-1)])

    if q is None:
        alpha = defaults.alpha(alpha)
        q = alpha_to_q(alpha=alpha, tail=tail)

    return np.multiply.outer(scipy.stats.norm().ppf(q),
                             np.sqrt(N - h)/np.sqrt((N*(N+2))))


def calc_H_max(
                x: Optional[Union[np.ndarray, list, float, int,]] = None,
                H_max: Optional[int] = None,
                ) -> int:

    if x is None and H_max is None:
        H_max = defaults.H_max()
    elif x is None:
        pass
    elif H_max is None:
        H_max = np.size(x)-1
    else:
        H_max = np.min([H_max, np.size(x)-1])

    return H_max
