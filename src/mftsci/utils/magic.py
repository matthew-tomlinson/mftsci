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
    """
    Class to represent a magic property handler.

    Methods
    -------
    __init__():
        Initializes the magic property handler.
    __mprop__() -> Union[str, None]:
        Returns the magic property name.
    __mpropkw__() -> Union[str, None]:
        Returns the magic property keyword.
    __setmpropnames__(mprop: str = None, mpropkw: str = None):
        Sets the magic property names.
    __setmpropmethods__(
                        ret_class_val: bool = None,
                        ret_class_val_bool: bool = None,
                        noval: Union[np.ndarray, list, str] = None,
                        noval_in: Union[np.ndarray, list, str] = None,
                        noval_rm: Union[np.ndarray, list, str] = None,
                        val: Union[np.ndarray, list, str] = None,
                        val_in: Union[np.ndarray, list, str] = None,
                        val_rm: Union[np.ndarray, list, str] = None,
                        val_bool: Union[np.ndarray, list, str] = None,
                        val_bool_in: Union[np.ndarray, list, str] = None,
                        val_bool_rm: Union[np.ndarray, list, str] = None,
                        ):
        Sets the magic property methods.
    __setmprop__(
                    mprop: str = None,
                    mpropkw: str = None,
                    ret_class_val: bool = None,
                    ret_class_val_bool: bool = None,
                    noval: Union[np.ndarray, list, str] = None,
                    noval_in: Union[np.ndarray, list, str] = None,
                    noval_rm: Union[np.ndarray, list, str] = None,
                    val: Union[np.ndarray, list, str] = None,
                    val_in: Union[np.ndarray, list, str] = None,
                    val_rm: Union[np.ndarray, list, str] = None,
                    val_bool: Union[np.ndarray, list, str] = None,
                    val_bool_in: Union[np.ndarray, list, str] = None,
                    val_bool_rm: Union[np.ndarray, list, str] = None,
                    ):
        Sets the magic property.
    __getmprop__(val=None):
        Gets the magic property value.
    __copymprop__(val=None):
        Copies the magic property value.
    """

    def __init__(
                    self,
                    ):
        """
        Initializes the magic property handler.
        """

        pass

    def __mprop__(
                    self,
                    ) -> Union[str, None]:
        """
        Returns the magic property name.

        Returns
        -------
        Union[str, None]
            The magic property name, or None if not set.
        """

        return None

    def __mpropkw__(
                    self,
                    ) -> Union[str, None]:
        """
        Returns the magic property keyword.

        Returns
        -------
        Union[str, None]
            The magic property keyword, or None if not set.
        """

        return None

    def __setmpropnames__(
                            self,
                            mprop: Optional[str] = None,
                            mpropkw: Optional[str] = None,
                            ):
        """
        Sets the magic property names.

        Parameters
        ----------
        mprop : str, optional
            The magic property name, by default None.
        mpropkw : str, optional
            The magic property keyword, by default None.
        """

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
        """
        Sets the magic property methods.

        Parameters
        ----------
        ret_class_val : bool, optional
            Return class value, by default None.
        ret_class_val_bool : bool, optional
            Return class value as boolean, by default None.
        noval : Union[np.ndarray, list, str], optional
            No value methods, by default `get_magic_method_names()['noval']`.
        noval_in : Union[np.ndarray, list, str], optional
            No value inclusion methods, by default None.
        noval_rm : Union[np.ndarray, list, str], optional
            No value removal methods, by default None.
        val : Union[np.ndarray, list, str], optional
            Value methods, by default `get_magic_method_names()['val']`.
        val_in : Union[np.ndarray, list, str], optional
            Value inclusion methods, by default None.
        val_rm : Union[np.ndarray, list, str], optional
            Value removal methods, by default None.
        val_bool : Union[np.ndarray, list, str], optional
            Value boolean methods, by default `get_magic_method_names()['val_bool']`.
        val_bool_in : Union[np.ndarray, list, str], optional
            Value boolean inclusion methods, by default None.
        val_bool_rm : Union[np.ndarray, list, str], optional
            Value boolean removal methods, by default None.
        """

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
        """
        Sets the magic property.

        Parameters
        ----------
        mprop : str, optional
            The magic property name, by default None.
        mpropkw : str, optional
            The magic property keyword, by default None.
        ret_class_val : bool, optional
            Return class value, by default None.
        ret_class_val_bool : bool, optional
            Return class value as boolean, by default None.
        noval : Union[np.ndarray, list, str], optional
            No value methods, by default `get_magic_method_names()['noval']`.
        noval_in : Union[np.ndarray, list, str], optional
            No value inclusion methods, by default None.
        noval_rm : Union[np.ndarray, list, str], optional
            No value removal methods, by default None.
        val : Union[np.ndarray, list, str], optional
            Value methods, by default `get_magic_method_names()['val']`.
        val_in : Union[np.ndarray, list, str], optional
            Value inclusion methods, by default None.
        val_rm : Union[np.ndarray, list, str], optional
            Value removal methods, by default None.
        val_bool : Union[np.ndarray, list, str], optional
            Value boolean methods, by default `get_magic_method_names()['val_bool']`.
        val_bool_in : Union[np.ndarray, list, str], optional
            Value boolean inclusion methods, by default None.
        val_bool_rm : Union[np.ndarray, list, str], optional
            Value boolean removal methods, by default None.
        """

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
        """
        Gets the magic property value.

        Parameters
        ----------
        val : optional
            The value to get, by default None.

        Returns
        -------
        object
            The magic property value.
        """

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
        """
        Returns an instance of the class copied from the magic property value.

        Parameters
        ----------
        val : optional
            The value to copy, by default None.

        Returns
        -------
        object
            An instance of the class copied from the magic property value.
        """

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
    """
    Creates a magic method that does not take `value` as an argument.

    Parameters
    ----------
    method_name : str
        The name of the method.

    Returns
    -------
    Callable
        The magic method.
    """
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
    """
    Creates a magic method that does take `value` as an argument.

    Parameters
    ----------
    method_name : str
        The name of the method.
    ret_class : bool, optional
        Whether to return the class instance, by default None.

    Returns
    -------
    Callable
        The magic method.
    """
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
    """
    Sets the magic property for a class.

    Parameters
    ----------
    _class : class
        The class to set the magic property for.
    mprop : str, optional
        The magic property name, by default None.
    mpropkw : str, optional
        The magic property keyword, by default None.
    ret_class_val : bool, optional
        Return class value, by default None.
    ret_class_val_bool : bool, optional
        Return class value as boolean, by default None.
    noval : Union[np.ndarray, list, str], optional
        No value methods, by default `get_magic_method_names()['noval']`.
    noval_in : Union[np.ndarray, list, str], optional
        No value inclusion methods, by default None.
    noval_rm : Union[np.ndarray, list, str], optional
        No value removal methods, by default None.
    val : Union[np.ndarray, list, str], optional
        Value methods, by default `get_magic_method_names()['val']`.
    val_in : Union[np.ndarray, list, str], optional
        Value inclusion methods, by default None.
    val_rm : Union[np.ndarray, list, str], optional
        Value removal methods, by default None.
    val_bool : Union[np.ndarray, list, str], optional
        Value boolean methods, by default `get_magic_method_names()['val_bool']`.
    val_bool_in : Union[np.ndarray, list, str], optional
        Value boolean inclusion methods, by default None.
    val_bool_rm : Union[np.ndarray, list, str], optional
        Value boolean removal methods, by default None.
    """

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
    """
    Sets the magic property names for a class.

    Parameters
    ----------
    _class : class
        The class to set the magic property names for.
    mprop : str, optional
        The magic property name, by default None.
    mpropkw : str, optional
        The magic property keyword, by default None.
    """

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
    """
    Sets the magic property methods for a class.

    Parameters
    ----------
    _class : class
        The class to set the magic property methods for.
    ret_class_val : bool, optional
        Return class value, by default None.
    ret_class_val_bool : bool, optional
        Return class value as boolean, by default None.
    noval : Union[np.ndarray, list, str], optional
        No value methods, by default `get_magic_method_names()['noval']`.
    noval_in : Union[np.ndarray, list, str], optional
        No value inclusion methods, by default None.
    noval_rm : Union[np.ndarray, list, str], optional
        No value removal methods, by default None.
    val : Union[np.ndarray, list, str], optional
        Value methods, by default `get_magic_method_names()['val']`.
    val_in : Union[np.ndarray, list, str], optional
        Value inclusion methods, by default None.
    val_rm : Union[np.ndarray, list, str], optional
        Value removal methods, by default None.
    val_bool : Union[np.ndarray, list, str], optional
        Value boolean methods, by default `get_magic_method_names()['val_bool']`.
    val_bool_in : Union[np.ndarray, list, str], optional
        Value boolean inclusion methods, by default None.
    val_bool_rm : Union[np.ndarray, list, str], optional
        Value boolean removal methods, by default None.
    """

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
