"""
Functions module for multiprocessing utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Tuple

# External packages
import numpy as np
from multiprocessing import Pool

# Internal packages
from mftsci.utils.meta import smap
from mftsci.utils.timer import Timer
from mftsci.utils.multiproc import defaults


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def mp_funs(
            funs: Union[np.ndarray, list],
            ) -> list:

    pool = Pool(processes=len(funs))

    res = pool.map(smap, funs)

    return res


def mp_funs_factor(
                    funs: Union[np.ndarray, list],
                    factor: Optional[int] = None,
                    print_mp_time: Optional[bool] = None,
                    write_mp_time: Optional[bool] = None,
                    write_mp_dir: Optional[bool] = None,
                    write_mp_fn: Optional[bool] = None,
                    ) -> Tuple[list, list]:

    print_mp_time = defaults.print_mp_time(print_mp_time)
    write_mp_time = defaults.write_mp_time(write_mp_time)

    if write_mp_time:
        write_mp_fn = defaults.write_mp_fn(write_mp_fn)

    fn = calc_rlist(funs, factor=factor)

    n_runs = len(fn)
    res = [None for i in range(n_runs)]
    time_run = [None for i in range(n_runs)]

    for i in range(n_runs):
        preamble = f"\t{'mp run'} {1+i}/{n_runs} (n_proc = {len(fn[i])})"
        print(preamble)
        timer_run = Timer(print_timer=print_mp_time, print_tab="\t",
                          write_timer=write_mp_time, write_dir=write_mp_dir,
                          write_fn=write_mp_fn, write_preamble=preamble)
        res[i] = mp_funs(funs=fn[i])
        timer_run.stop()
        time_run[i] = timer_run.get_timer_dict()

    return calc_flist(res), time_run


def calc_rlist(
                flist: list,
                factor: Optional[int] = None,
                ) -> list:

    n_funs = len(flist)
    if factor is None:
        factor = calc_factor(n_funs=n_funs)

    n_full_runs = n_funs//factor
    n_rem = n_funs % factor

    nlist = [[flist[i*factor+j] for j in range(factor)] for i in range(n_full_runs)]
    if n_rem != 0:
        nlist.append([flist[n_full_runs*factor+j] for j in range(n_rem)])

    return nlist


def calc_factor(
                n_funs: int,
                max_factor: Optional[int] = None,
                ) -> int:

    max_factor = defaults.write_mp_time(max_factor)

    factors = (1+np.array(range(max_factor)))
    n_runs = (n_funs//factors) + ((n_funs % factors) != 0)

    return factors[np.where(n_runs == n_runs[-1])[0][0]]


def calc_flist(
                rlist: list,
                ) -> list:

    return [rlist[i][j] for i in range(len(rlist)) for j in range(len(rlist[i]))]
