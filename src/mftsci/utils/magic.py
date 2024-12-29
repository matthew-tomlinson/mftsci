"""
Module for magic method utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Callable

# External packages
import numpy as np

# Internal packages
from mftsci.utils.meta import get_magic_method_names
from mftsci.utils.arr import filter


# =====================================
# CLASSES
# =====================================

class MagicProp():

    def __init__(
                    self,
                    ):
        pass

    def __mprop__(
                    self,
                    ) -> Union[str, None]:

        return None

    def __mpropkw__(
                    self,
                    ) -> Union[str, None]:

        return None

    def __setmpropnames__(
                            self,
                            mprop: Optional[str] = None,
                            mpropkw: Optional[str] = None,
                            ):

        _setmpropnames(
                        _class=self,
                        mprop=mprop,
                        mpropkw=mpropkw,
                        )

    def __setmpropmethods__(
                            self,
                            ret_class_val: Optional[bool] = None,
                            ret_class_val_bool: Optional[bool] = None,
                            noval: Optional[Union[np.ndarray, list, str,]] = None,
                            noval_in: Optional[Union[np.ndarray, list, str,]] = None,
                            noval_rm: Optional[Union[np.ndarray, list, str,]] = None,
                            val: Optional[Union[np.ndarray, list, str,]] = None,
                            val_in: Optional[Union[np.ndarray, list, str,]] = None,
                            val_rm: Optional[Union[np.ndarray, list, str,]] = None,
                            val_bool: Optional[Union[np.ndarray, list, str,]] = None,
                            val_bool_in: Optional[Union[np.ndarray, list, str,]] = None,
                            val_bool_rm: Optional[Union[np.ndarray, list, str,]] = None,
                            ):

        _setmpropmethods(
                            _class=self,
                            ret_class_val=ret_class_val,
                            ret_class_val_bool=ret_class_val_bool,
                            noval=noval,
                            noval_in=noval_in,
                            noval_rm=noval_rm,
                            val=val,
                            val_in=val_in,
                            val_rm=val_rm,
                            val_bool=val_bool,
                            val_bool_in=val_bool_in,
                            val_bool_rm=val_bool_rm,
                            )

    def __setmprop__(
                        self,
                        mprop: Optional[str] = None,
                        mpropkw: Optional[str] = None,
                        ret_class_val: Optional[bool] = None,
                        ret_class_val_bool: Optional[bool] = None,
                        noval: Optional[Union[np.ndarray, list, str,]] = None,
                        noval_in: Optional[Union[np.ndarray, list, str,]] = None,
                        noval_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        val: Optional[Union[np.ndarray, list, str,]] = None,
                        val_in: Optional[Union[np.ndarray, list, str,]] = None,
                        val_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        val_bool: Optional[Union[np.ndarray, list, str,]] = None,
                        val_bool_in: Optional[Union[np.ndarray, list, str,]] = None,
                        val_bool_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        ):

        self.__setmpropnames__(
                                mprop=mprop,
                                mpropkw=mpropkw
                                )
        self.__setmpropmethods__(
                                    ret_class_val=ret_class_val,
                                    ret_class_val_bool=ret_class_val_bool,
                                    noval=noval,
                                    noval_in=noval_in,
                                    noval_rm=noval_rm,
                                    val=val,
                                    val_in=val_in,
                                    val_rm=val_rm,
                                    val_bool=val_bool,
                                    val_bool_in=val_bool_in,
                                    val_bool_rm=val_bool_rm,
                                    )

    def __getmprop__(
                        self,
                        val=None,
                        ) -> object:

        if val is None:
            val = self

        if self.__mprop__() is None:
            return val
        elif isinstance(val, type(self)):
            return getattr(val, self.__mprop__())
        else:
            return val

    def __copymprop__(
                        self,
                        val=None,
                        ) -> object:

        if val is None:
            val = self

        if self.__mprop__() is None:
            return val
        elif isinstance(val, type(self)):
            return val
        else:
            return type(self)(val)


# =====================================
# FUNCTIONS
# =====================================


def _mprop_novalue(
                    method_name: str,
                    ) -> Callable:

    def magic_method(
                        self,
                        *args, **kwargs,
                    ):
        return getattr(self.__getmprop__(), method_name)(*args, **kwargs)
    return magic_method


def _mprop_value(
                    method_name,
                    ret_class: Optional[bool] = None
                    ) -> Callable:

    if ret_class is None:
        ret_class = False

    def _magic_method(
                        self,
                        value,
                        *args, **kwargs,
                        ):
        return getattr(self.__getmprop__(), method_name)(self.__getmprop__(val=value),
                                                         *args, **kwargs)

    if ret_class:
        def magic_method(self, value, *args, **kwargs):
            return type(self)(**{self.__mpropkw__(): _magic_method(self, value,
                                                                   *args, **kwargs)})
    else:
        magic_method = _magic_method

    return magic_method


def _setmprop(
                _class,
                mprop: Optional[str] = None,
                mpropkw: Optional[str] = None,
                ret_class_val: Optional[bool] = None,
                ret_class_val_bool: Optional[bool] = None,
                noval: Optional[Union[np.ndarray, list, str,]] = None,
                noval_in: Optional[Union[np.ndarray, list, str,]] = None,
                noval_rm: Optional[Union[np.ndarray, list, str,]] = None,
                val: Optional[Union[np.ndarray, list, str,]] = None,
                val_in: Optional[Union[np.ndarray, list, str,]] = None,
                val_rm: Optional[Union[np.ndarray, list, str,]] = None,
                val_bool: Optional[Union[np.ndarray, list, str,]] = None,
                val_bool_in: Optional[Union[np.ndarray, list, str,]] = None,
                val_bool_rm: Optional[Union[np.ndarray, list, str,]] = None,
                ):

    _setmpropnames(
                    _class=_class,
                    mprop=mprop,
                    mpropkw=mpropkw
                    )
    _setmpropmethods(
                        _class=_class,
                        ret_class_val=ret_class_val,
                        ret_class_val_bool=ret_class_val_bool,
                        noval=noval,
                        noval_in=noval_in,
                        noval_rm=noval_rm,
                        val=val,
                        val_in=val_in,
                        val_rm=val_rm,
                        val_bool=val_bool,
                        val_bool_in=val_bool_in,
                        val_bool_rm=val_bool_rm,
                        )


def _setmpropnames(
                    _class,
                    mprop: Optional[str] = None,
                    mpropkw: Optional[str] = None,
                    ):

    if mpropkw is None:
        mpropkw = mprop

    setattr(_class, "__mprop__", lambda self : mprop)   # noqa: E203
    setattr(_class, "__mpropkw__", lambda self : mpropkw)   # noqa: E203


def _setmpropmethods(
                        _class,
                        ret_class_val: Optional[bool] = None,
                        ret_class_val_bool: Optional[bool] = None,
                        noval: Optional[Union[np.ndarray, list, str,]] = None,
                        noval_in: Optional[Union[np.ndarray, list, str,]] = None,
                        noval_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        val: Optional[Union[np.ndarray, list, str,]] = None,
                        val_in: Optional[Union[np.ndarray, list, str,]] = None,
                        val_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        val_bool: Optional[Union[np.ndarray, list, str,]] = None,
                        val_bool_in: Optional[Union[np.ndarray, list, str,]] = None,
                        val_bool_rm: Optional[Union[np.ndarray, list, str,]] = None,
                        ):

    if ret_class_val is None:
        ret_class_val = True
    if ret_class_val_bool is None:
        ret_class_val_bool = False

    _names = get_magic_method_names()

    if noval is False:
        _names["noval"] = []
    elif noval is None:
        noval = _names["noval"]
    _names["noval"] = filter(a=noval, a_in=noval_in, a_rm=noval_rm)

    if val is False:
        _names["val"] = []
    elif val is None:
        val = _names["val"]
    _names["val"] = filter(a=val, a_in=val_in, a_rm=val_rm)

    if val_bool is False:
        _names["val_bool"] = []
    elif val_bool is None:
        val_bool = _names["val_bool"]
    _names["val_bool"] = filter(a=val_bool, a_in=val_bool_in, a_rm=val_bool_rm)

    for method_name in _names["noval"]:
        setattr(_class, method_name, _mprop_novalue(method_name))
    for method_name in _names["val"]:
        setattr(_class, method_name, _mprop_value(method_name,
                                                  ret_class=ret_class_val))
    for method_name in _names["val_bool"]:
        setattr(_class, method_name, _mprop_value(method_name,
                                                  ret_class=ret_class_val_bool))
