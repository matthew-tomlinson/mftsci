"""
Module for confidence intervals.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from __future__ import annotations
from typing import Optional, Union, Literal, Tuple

# External packages
import numpy as np
import matplotlib

# Internal packages
from mftsci.htest.core import *     # noqa: F401, F403
from mftsci import utils    # temp

# =====================================
# CLASSES
# =====================================


class ConfInterval():

    def __init__(
                    self,
                    CI: Optional[ConfInterval] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    tail: Optional[Literal["both", "left", "right",]] = None,
                    color: Optional[Union[np.ndarray, list, str,]] = None,
                    c_alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    c_cs: Optional[Union[np.ndarray, list,]] = None,
                    c_N_grad: Optional[Union[np.ndarray, list, int,]] = None,
                    c_step: Optional[Union[np.ndarray, list, float, int,]] = None,
                    c_i_0: Optional[Union[np.ndarray, list, int,]] = None,
                    ):

        if CI is not None:
            self.set_all(**CI.get_all(), reset=True)
            reset = False
        else:
            reset = True

        self.set_all(alpha=alpha, tail=tail,
                     color=color, c_alpha=c_alpha, c_cs=c_cs, c_N_grad=c_N_grad,
                     c_step=c_step, c_i_0=c_i_0, reset=reset)

    def set_all(
                self,
                alpha: Optional[Union[np.ndarray, list, float,]] = None,
                tail: Optional[Literal["both", "left", "right",]] = None,
                color: Optional[Union[np.ndarray, list, str,]] = None,
                c_alpha: Optional[Union[np.ndarray, list, str,]] = None,
                c_cs: Optional[Union[np.ndarray, list,]] = None,
                c_N_grad: Optional[Union[np.ndarray, list, int,]] = None,
                c_step: Optional[Union[np.ndarray, list, float, int,]] = None,
                c_i_0: Optional[Union[np.ndarray, list, int,]] = None,
                reset: Optional[bool] = None,
                ):

        self.set_alpha(alpha=alpha, reset=reset)
        self.set_tail(tail=tail, reset=reset)
        self.set_color(color=color, c_alpha=c_alpha, c_cs=c_cs, c_N_grad=c_N_grad,
                       c_step=c_step, c_i_0=c_i_0, reset=reset)

    def get_all(
                self,
                ) -> dict:

        return {k: self.__dict__[k] for k in ["alpha", "tail", "color"]}

    def set_alpha(
                    self,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    reset: Optional[bool] = None,
                    ):

        if reset is None:
            reset = False

        if alpha is not None:
            self.alpha = alpha
        elif reset:
            self.alpha = default_alphas()
        self.alpha = utils.arr.iterarr(self.alpha)

        self.sort_alpha()
        self.set_N()

    def sort_alpha(
                    self,
                    ):

        if not np.isscalar(self.alpha):
            self.alpha = np.sort(self.alpha)

    def set_N(
                self,
                ):

        self.N = np.size(self.alpha)

    def set_tail(
                    self,
                    tail: Optional[Literal["both", "left", "right",]] = None,
                    reset: Optional[bool] = None,
                    ):

        if reset is None:
            reset = False

        if tail is not None:
            self.tail = tail
        elif reset:
            self.tail = default_tail()

    def set_color(
                    self,
                    color: Optional[Union[np.ndarray, list, str,]] = None,
                    c_alpha: Optional[Union[np.ndarray, list, str,]] = None,
                    c_cs: Optional[Union[np.ndarray, list,]] = None,
                    c_N_grad: Optional[Union[np.ndarray, list, int,]] = None,
                    c_step: Optional[Union[np.ndarray, list, float, int,]] = None,
                    c_i_0: Optional[Union[np.ndarray, list, int,]] = None,
                    reset: Optional[bool] = None,
                    ):

        if reset is None:
            reset = False

        if color is not None:
            self.color = matplotlib.colors.to_rgba_array(color, alpha=c_alpha)
        elif np.any([a is not None for a in [c_cs, c_N_grad, c_step, c_i_0]]) or reset:
            if c_N_grad is None:
                c_N_grad = 2
            if c_i_0 is None:
                c_i_0 = 1

            self.color = utils.colour.c_grad_cycle(cs=c_cs, N_c=self.N, N_grad=c_N_grad,
                                                   step=c_step, i_0=c_i_0,
                                                   max_contrast_0=True)

    def calc_CI_strs(
                        self,
                        dp=None,
                        ) -> np.ndarray:

        g_str = utils.scif.calc_scif(val=100*self.gamma(), pow10=0, dp=dp,
                                     vec_unif=True, vec_unif_mode=1)

        return np.array([f"{g}% CI" for g in g_str])

    def p_scif_reject(
                        self,
                        p: Optional[Union[np.ndarray, list, float,]] = None,
                        scif_args: Optional[tuple] = None,
                        scif_kwargs: Optional[dict] = None,
                        brac_args: Optional[tuple] = None,
                        brac_kwargs: Optional[dict] = None,
                        ) -> Union[np.ndarray, str,]:

        return calc_p_scif_reject(
                                    p=p, alpha=self.alpha,
                                    scif_args=scif_args, scif_kwargs=scif_kwargs,
                                    brac_args=brac_args, brac_kwargs=brac_kwargs,
                                    )

    def reject(
                self,
                p: Optional[Union[np.ndarray, list, float,]] = None,
                ) -> Union[np.ndarray, bool,]:

        return calc_reject(p=p, alpha=self.alpha)

    def q(
            self,
            tail: Optional[Literal["both", "left", "right",]] = None,
            ) -> Union[np.ndarray, float,]:

        self.set_tail(tail=tail)
        return alpha_to_q(alpha=self.alpha)

    def gamma(
                self,
                ) -> Union[np.ndarray, float,]:

        return 1 - self.alpha


# =====================================
# FUNCTIONS
# =====================================


def alpha_to_q(
                alpha: Optional[Union[np.ndarray, list, float,]] = None,
                tail: Optional[Literal["both", "left", "right",]] = None,
                ) -> Union[np.ndarray, float,]:

    alpha = default_alpha(alpha=alpha)
    tail = default_tail(tail=tail)

    if tail == "left":
        q = alpha
    elif tail == "right":
        q = 1 - alpha
    else:
        q = 0.5 + np.multiply.outer(0.5-alpha/2, np.array([-1, 1]))

    return q


def q_to_salpha(
                q: Optional[Union[np.ndarray, list, float,]] = None,
                tail: Optional[Literal["both", "left", "right",]] = None,
                ) -> Union[np.ndarray, float,]:

    q = default_q(q=q)
    tail = default_tail(tail=tail)

    if tail == "left":
        salpha = -q
    elif tail == "right":
        salpha = 1-q
    else:
        q_shift = q - 0.5
        q_sign = (
                    np.array(np.sign(q_shift), dtype=int)
                    + np.array(q_shift == 0, dtype=int)
                    )
        salpha = q_sign*(0.5 - np.abs(q_shift))

    return salpha


def calc_p_scif(
                p: Optional[Union[np.ndarray, list, float,]] = None,
                scif_args: Optional[tuple] = None,
                scif_kwargs: Optional[dict] = None,
                ) -> Union[np.ndarray, str,]:

    scif_args, scif_kwargs = utils.meta.default_args_kwargs(scif_args, scif_kwargs)
    utils.dict.append_if_not_in(d=scif_kwargs, keys_check=["dp"], value=1)
    utils.dict.append_if_not_in(d=scif_kwargs, keys_check=["scif_format"], value="e")

    return utils.scif.calc_scif(p, *scif_args, **scif_kwargs)


def calc_p_scif_reject(
                        p: Optional[Union[np.ndarray, list, float,]] = None,
                        alpha: Optional[Union[np.ndarray, list, float,]] = None,
                        scif_args: Optional[tuple] = None,
                        scif_kwargs: Optional[dict] = None,
                        brac_args: Optional[tuple] = None,
                        brac_kwargs: Optional[dict] = None,
                        ) -> Union[np.ndarray, str,]:

    p_scif = calc_p_scif(p=p, scif_args=scif_args, scif_kwargs=scif_kwargs)
    brac_reject_ps = calc_brac_reject_ps(p=p, alpha=alpha, brac_args=brac_args,
                                         brac_kwargs=brac_kwargs)

    if np.size(brac_reject_ps) == 0:
        return p_scif
    else:
        return utils.str.capp_ps(s=p_scif,
                                 pre=brac_reject_ps[..., 0],
                                 suff=brac_reject_ps[..., 1])


def calc_reject(
                p: Optional[Union[np.ndarray, list, float,]] = None,
                alpha: Optional[Union[np.ndarray, list, float,]] = None,
                ) -> Union[np.ndarray, bool,]:

    if p is None:
        p = 1

    if alpha is None:
        alpha = []

    return np.less.outer(p, alpha)


def calc_brac_reject_ps(
                        p: Optional[Union[np.ndarray, list, float,]] = None,
                        alpha: Optional[Union[np.ndarray, list, float,]] = None,
                        brac_args: Optional[tuple] = None,
                        brac_kwargs: Optional[dict] = None,
                        ) -> np.ndarray:

    if alpha is None:
        alpha = []

    reject = calc_reject(p=p, alpha=alpha)

    if np.size(alpha) == 0:
        a_brac = np.full((*np.shape(p), 2), "")
    else:
        brac_args, brac_kwargs = utils.meta.default_args_kwargs(brac_args, brac_kwargs)
        utils.dict.append_if_not_in(d=brac_kwargs, keys_check=["brac_type"], value="[]")
        # brac_kwargs = utils.dict.filter_keys(brac_kwargs, keys_rm=["phantom"])
        base_brac = np.array([
                                utils.str.calc_brac(phantom=b,
                                                    *brac_args, **brac_kwargs)
                                for b in range(2)
                                ])
        a_brac = base_brac[1 - reject, :]

        if not np.isscalar(alpha):
            D_alpha = len(np.shape(alpha))
            axes = np.arange(D_alpha) - (1+D_alpha)
            a_brac, N_i, shape_m = utils.str.reshape_iter_pre(a=a_brac, axes=axes,
                                                              col_iter=False)
            a_brac[..., 1, :] = np.flip(a_brac[..., 1, :], axis=-1)
            a_brac = utils.str.concat(a_brac, axes=-1)

    return a_brac


# Defaults
# -------------------------------------


def default_alpha(
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    ) -> Union[np.ndarray, float,]:

    if alpha is None:
        alpha = 0.05
    elif isinstance(alpha, list):
        alpha = np.array(alpha)
    return alpha


def default_alphas(
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    ) -> Union[np.ndarray, float,]:

    if alpha is None:
        alpha = np.array([0.01, 0.05])
    return alpha


def default_q(
                q: Optional[Union[np.ndarray, list, float,]] = None,
                ) -> Union[np.ndarray, float,]:

    if q is None:
        q = 0.5
    elif isinstance(q, list):
        q = np.array(q)
    return q


def default_tail(
                    tail: Optional[Literal["both", "left", "right",]] = None,
                    ) -> str:

    if tail is None:
        tail = "both"
    return tail


def default_CI_color(
                        color: Optional[Union[np.ndarray, list, str,]] = None,
                        ) -> Union[np.ndarray, str,]:

    if color is None:
        color = ["#d7d7d7", "#ebebeb"]
    return color


def calc_plot_CI(
                    plot_CI: Optional[bool] = None,
                    alpha: Optional[Union[np.ndarray, list, float,]] = None,
                    color: Optional[Union[np.ndarray, list, str,]] = None,
                    ) -> Tuple[
                                bool,
                                int,
                                Union[np.ndarray, float,],
                                Union[np.ndarray, str,],
                                ]:

    if plot_CI is None:
        plot_CI = True
    alpha = default_alpha(alpha=alpha)
    color = default_CI_color(color=color)
    N = len(alpha)
    if N == 0:
        plot_CI = False

    return plot_CI, N, alpha, color
