"""
Core module for `htests` subpackage.
"""
# =====================================
# IMPORT
# =====================================
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.htest.confidence_interval import *  # noqa: F401, F403
from mftsci import utils    # temp


# =====================================
# CLASSES
# =====================================


class TestMeta():

    def __init__(
                    self,
                    test_long_name: str,
                    test_short_name: str,
                    ):

        self._test_long_name = test_long_name
        self._test_short_name = test_short_name


class TestWrapper():

    def __init__(
                    self,
                    *args, **kwargs,
                    ):

        self.res = None

    def __repr__(
                    self,
                    ) -> str:

        return self.res.__repr__()

    def set_res(
                self,
                ):
        pass


class TestRes(TestMeta):

    def __init__(
                    self,
                    test_long_name: str,
                    test_short_name: str,
                    ):

        TestMeta.__init__(self,
                          test_long_name=test_long_name,
                          test_short_name=test_short_name)

    def __repr__(
                    self,
                    ) -> str:

        return f"{self._test_short_name}_result:\n{self.calc_res_df().__repr__()}"

    if False:
        def _repr_html_(
                        self,
                        ) -> str:

            return f"{self._test_short_name}_result:\n{self.calc_res_df()._repr_html_()}"   # noqa: E501

    def __defaultkeys__(
                        self,
                        ) -> list:

        return ["stat", "p"]

    def calc_res_keys(
                        self,
                        keys: Optional[Union[np.ndarray, list, str,]] = None,
                        keys_in: Optional[Union[np.ndarray, list, str,]] = None,
                        keys_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        ) -> np.ndarray:

        if keys is None:
            keys = self.__defaultkeys__()

        keys = utils.arr.filter(a=keys, a_in=keys_in, a_rm=keys_rm)

        return keys

    def calc_res_index(
                        self,
                        ) -> list:

        return [0]

    def calc_metafunc_index(
                                self,
                                *args, **kwargs,
                                ) -> Callable:

        return utils.meta.return_x

    def calc_metafunc_value(
                                self,
                                *args, **kwargs,
                                ) -> Callable:

        return utils.meta.return_x

    def calc_res_dict(
                        self,
                        keys: Optional[Union[np.ndarray, list, str,]] = None,
                        keys_in: Optional[Union[np.ndarray, list, str,]] = None,
                        keys_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        fvalue: Optional[Callable] = None,
                        ) -> dict:

        fvalue = utils.meta.metafunc_decorator(
                    metafunc=self.calc_metafunc_value(),
                    )(fvalue)

        keys = self.calc_res_keys(keys=keys, keys_in=keys_in, keys_rm=keys_rm)

        return {k: fvalue(getattr(self, k)) for k in keys}

    def calc_res_df(
                    self,
                    keys: Optional[Union[np.ndarray, list, str,]] = None,
                    keys_in: Optional[Union[np.ndarray, list, str,]] = None,
                    keys_rm: Optional[Union[np.ndarray, list, str,]] = None,
                    fvalue: Optional[Callable] = None,
                    findex: Optional[Callable] = None,
                    ind_slice: Optional[Union[
                                                pd.core.indexes.base.Index,
                                                pd.core.indexes.multi.MultiIndex,
                                                np.ndarray,
                                                list,
                                                object,
                                                ]] = None,
                    flatten: Optional[bool] = None,
                    ) -> pd.core.frame.DataFrame:

        if flatten is None:
            flatten = False

        findex = utils.meta.metafunc_decorator(
                        metafunc=self.calc_metafunc_index(),
                        )(findex)

        index = findex(self.calc_res_index())

        df = pd.DataFrame(self.calc_res_dict(
                                            keys=keys,
                                            keys_in=keys_in,
                                            keys_rm=keys_rm,
                                            fvalue=fvalue,
                                            ), index=index)
        if ind_slice is not None:
            df = df[ind_slice]

        if flatten:
            df = df.unstack().to_frame().sort_index(level=1).T
            df.columns = df.columns.map(lambda x: f"{x[0]}_{x[1]}")

        return df
