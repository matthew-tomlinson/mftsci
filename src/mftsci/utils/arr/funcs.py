"""
Functions module for array utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union, Callable, Tuple

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils import meta
from mftsci.utils.arr import defaults
from mftsci.utils.arr.core import *     # noqa: F401, F403
from mftsci.utils.arr.core import convert


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


_attr_vector = np.vectorize(meta.attr, excluded=["attr", "args", "kwargs"])
_fattr_vector = np.vectorize(meta.fattr, excluded=["args", "kwargs"])


def overwrite_a_bool(
                            a: Union[np.ndarray, float, int, bool, str,],
                            a1: Optional[Union[np.ndarray, float, int, bool, str,]],
                            a_bool: Optional[Union[np.ndarray, list, bool,]] = None,
                            ):

    # a = defaults.a(a)
    if a1 is None:
        a1 = a

    if a_bool is None:
        a_bool = True

    if np.any(a_bool):
        if np.isscalar(a):
            if np.isscalar(a_bool):
                a = a1
            elif np.isscalar(a1):
                a0 = a
                dtype = pd.Series([a0, a1]).dtype
                a = a_bool.copy().astype(dtype)
                a[a_bool] = a1
                a[~a_bool] = a0

                if False:
                    if isinstance(a_bool, pd.Series) or isinstance(pd.DataFrame):
                        a = a_bool.replace(False, a).replace(True, a1)
                    else:
                        a = np.full(np.shape(a_bool), a)
                        a[a_bool] = a1
            else:
                a1[~a_bool] = a
                a = a1
        elif np.isscalar(a1):
            if np.isscalar(a_bool):
                a[a == a] = a1
            else:
                a[a_bool] = a1
        else:
            if np.isscalar(a_bool):
                a = a1
            else:
                a[a_bool] = a1[a_bool]

    return a


np_round = meta.vectorize_pd_alt()(np.round)


def _round_str_scalar(
                        val: Union[float, int],
                        decimals: Optional[int] = None,
                        val_plus: Optional[bool] = None,
                        rm_sign: Optional[bool] = None,
                        ):

    if rm_sign is None:
        rm_sign = False
    if rm_sign:
        val_plus = True
    elif val_plus is None:
        val_plus = True

    if decimals is None:
        decimals = 0

    if val_plus:
        val_plus_str = "+"
    else:
        val_plus_str = ""

    s = f"{val:{val_plus_str}.{decimals}f}"

    if rm_sign:
        s = s[1:]
    return s


def _lead_zeros_scalar(
                        val: Union[float, int],
                        zeros: Optional[int] = None,
                        val_plus: Optional[bool] = None,
                        rm_sign: Optional[bool] = None,
                        ):

    if rm_sign is None:
        rm_sign = False
    if rm_sign:
        val_plus = True
    elif val_plus is None:
        val_plus = True

    if zeros is None:
        zeros = 0

    if val_plus:
        val_plus_str = "+"
        zeros += 1
    else:
        val_plus_str = ""

    s = f"{val:{val_plus_str}0{zeros}d}"

    if rm_sign:
        s = s[1:]
    return s


round_str = meta.vectorize_pd_alt()(_round_str_scalar)
lead_zeros = meta.vectorize_pd_alt()(_lead_zeros_scalar)


def attr(
            a: Union[np.ndarray, list, float, int, bool, str,],
            attr: Optional[str] = None,
            ) -> object:
    """
    Returns the specified attribute of each element of the input array or list if
    provided, otherwise returns the input array.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,]
        The input array or list.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    attr : str, optional
        The attribute function to apply.

    Returns
    -------
    np.ndarray
        The specified attribute of each element or the input array.

    Examples
    --------
    >>> attr(np.array([1. - 1.j, 2., 3. + 1.j]), attr="real")
    np.array([1., 2., 3.])
    """

    return _attr_vector(
                        obj=a,
                        attr=attr
                        )


def fattr(
            a: Union[np.ndarray, list, float, int, bool, str,],
            attr: Optional[str] = None,
            args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            ) -> object:
    """
    Runs the specified attribute function with arguments on each element of the input
    array or list.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,]
        The input array or list.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    attr : str, optional
        The function to apply.
    args : tuple, optional
        The positional arguments to pass to the function.
    kwargs : dict, optional
        The keyword arguments to pass to the function.

    Returns
    -------
    np.ndarray
        The output of the specified attribute function for each element or the input
        array.

    Examples
    --------
    >>> fattr(np.array(["apple", "banana"]), attr="capitalize")
    np.array(["Apple", "Banana"])
    """

    return _fattr_vector(
                        obj=a,
                        attr=attr,
                        args=args,
                        kwargs=kwargs,
                        )


def calc_shape(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                ) -> np.ndarray:
    """
    Calculates the shape of the input as a NumPy array.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The input whose shape is to be calculated. If None, the default function is
        used to initialize an empty array.
        Can be of types: np.ndarray, list, float, int, bool, or str.

    Returns
    -------
    np.ndarray
        The shape of the input as a NumPy array.

    Examples
    --------
    >>> calc_shape([])
    array([0])

    >>> calc_shape(5)
    array([])

    >>> calc_shape([1, 2, 3])
    array([3])
    """

    a = defaults.a(a=a)
    shape = np.array(np.shape(a))

    return shape


def calc_axes(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                ndim: Optional[int] = None,
                ) -> np.ndarray:
    """
    Calculates the axes of the input based on its shape or number of dimensions.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The input whose axes are to be calculated. If None, the default function is
        used to initialize an empty array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    shape : Union[np.ndarray, list, tuple,], optional
        The shape of the input, used to determine the number of dimensions if provided.
    ndim : int, optional
        The number of dimensions. If None, it is determined based on the shape or the
        input array.

    Returns
    -------
    np.ndarray
        The axes of the input as a NumPy array.

    Examples
    --------
    >>> calc_axes([])
    array([0])

    >>> calc_axes(5)
    array([])

    >>> calc_axes([1, 2, 3])
    array([0])
    """

    if ndim is None:
        if shape is None:
            ndim = convert(a=a, conserve_scalar=True).ndim
        else:
            ndim = len(shape)

    axes = np.arange(ndim)

    return axes


def calc_length(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                ) -> int:
    """
    Calculates the length of the first axis of an input based on its shape.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The input whose length is to be calculated. If None, the default function is
        used to initialize an empty array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    shape : Union[np.ndarray, list, tuple,], optional
        The shape of the input, used to determine the length if provided.

    Returns
    -------
    int
        The length of the input based on its shape.

    Examples
    --------
    >>> calc_length([])
    0

    >>> calc_length(5)
    0

    >>> calc_length([1, 2, 3])
    1
    """

    if shape is None:
        shape = calc_shape(a=a)
    if len(shape) == 0:
        length = 0
    else:
        length = shape[0]

    return length


def unique(
            a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            appear_order: Optional[bool] = None,
            *args, **kwargs,
            ) -> np.ndarray:
    """
    Returns the unique elements of the input array, optionally preserving their order
    of appearance.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The input array. If None, the default function is used to initialize an empty
        array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    appear_order : bool, optional
        If True, the order of first appearance of the unique elements is preserved.
        Default is True.
    *args :
        Additional arguments to pass to `np.unique`.
    **kwargs :
        Additional keyword arguments to pass to `np.unique`.

    Returns
    -------
    np.ndarray
        The unique elements of the input array.

    Examples
    --------
    >>> unique([1, 3, 3, 3, 2, 2])
    array([1, 3, 2])

    >>> unique([1, 3, 3, 3, 2, 2], appear_order=True)
    array([1, 3, 2])

    >>> unique([1, 3, 3, 3, 2, 2], appear_order=False)
    array([1, 2, 3])
    """

    appear_order = defaults.appear_order(appear_order)

    if appear_order:
        return_index = True
        uniq, idx = np.unique(a, return_index=return_index, *args, **kwargs)
        uniq = uniq[np.argsort(idx)]
    else:
        uniq = np.unique(a, *args, **kwargs)

    return uniq


def expand_dims(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                axis: Optional[Union[np.ndarray, list, tuple,]] = None,
                caxes: Optional[Union[np.ndarray, list, tuple,]] = None,
                taxes: Optional[Union[np.ndarray, list, tuple,]] = None,
                tndim: Optional[int] = None,
                tshape: Optional[Union[np.ndarray, list, tuple,]] = None,
                ta: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                expand_back: Optional[bool] = None,
                ) -> np.ndarray:
    """
    Expands the dimensions of the input array along the specified axis or axes.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The input array. If None, the default function is used to initialize an empty
        array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    axis : Union[np.ndarray, list, tuple,], optional
        The axis or axes along which to expand the dimensions of the input array.
    caxes : Union[np.ndarray, list, tuple,], optional
        The current axes of the input array.
    taxes : Union[np.ndarray, list, tuple,], optional
        The target axes for the expanded dimensions.
    tndim : int, optional
        The target number of dimensions.
    tshape : Union[np.ndarray, list, tuple,], optional
        The target shape of the input array.
    ta : Union[np.ndarray, list, float, int, bool, str,], optional
        The target array for calculating axes or shape.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    expand_back : bool, optional
        If True, the dimensions are expanded at the end of the axes. Default is False.

    Returns
    -------
    np.ndarray
        The input array with expanded dimensions.

    Examples
    --------
    >>> expand_dims([1, 2, 3], axis=0)
    array([[1, 2, 3]])

    >>> expand_dims(5, axis=1)
    array([[5]])
    """

    expand_back = defaults.expand_back(expand_back)

    a = convert(a=a, conserve_scalar=False)
    if axis is None:
        if taxes is None:
            taxes = calc_axes(a=ta, shape=tshape, ndim=tndim)

        if caxes is None:
            caxes = calc_axes(a=a)
            if expand_back:
                caxes += np.max(taxes) - np.max(caxes)

        axis = filter(a=taxes, a_rm=caxes)
    else:
        axis = iterarr(axis)

    return np.expand_dims(a, axis=tuple(axis))


def pad(
        a_s: Optional[list] = None,
        master_shape: Optional[Union[np.ndarray, list, tuple,]] = None,
        expand_back: Optional[bool] = None,
        *args, **kwargs,
        ) -> list:
    """
    Pads the input arrays to match the master shape.

    Parameters
    ----------
    a_s : list, optional
        The list of input arrays. If None, the default function is used to initialize
        an empty array.
        Can be of types: list.
    master_shape : Union[np.ndarray, list, tuple,], optional
        The shape to which the input arrays will be padded.
    expand_back : bool, optional
        If True, the dimensions are expanded at the end of the axes. Default is False.
    *args :
        Additional arguments to pass to `np.pad`.
    **kwargs :
        Additional keyword arguments to pass to `np.pad`.

    Returns
    -------
    list
        The list of padded arrays.

    Examples
    --------
    >>> pad([[1, 2, 3], [4, 5]])
    [array([1, 2, 3]), array([4, 5, 0])]

    >>> pad([5], master_shape=(3,))
    [array([5, 0, 0])]
    """

    if a_s is None:
        a_s = defaults.a()
    if type(a_s) is not list:
        a_s = [a_s]

    kwargs = meta.merge_args_kwargs(_func=np.pad, args=args, kwargs=kwargs)

    shapes = [calc_shape(a_i) for a_i in a_s]
    ndims = [len(s) for s in shapes]
    if master_shape is None:
        master_ndim = np.max(ndims)
    else:
        master_ndim = len(master_shape)

    if master_ndim == 0:
        xp_a_s = [
                    a_s[k] if ndims[k] == 0
                    else a_s[k][
                                tuple(slice(0, 1) for d in range(ndims[k]))
                                ] for k in range(len(ndims))
                ]

    elif kwargs.get("mode") == "outer":
        xp_a_s = list(np.meshgrid(*a_s))
    else:
        x_a_s = [
                    expand_dims(
                                a=a_i, tndim=master_ndim,
                                expand_back=expand_back,
                                )
                    for a_i in a_s
                ]
        x_shapes = np.array([calc_shape(a_i) for a_i in x_a_s])

        if master_shape is None:
            master_shape = np.max(x_shapes, axis=0)

        pad_widths = np.array([
                                np.transpose([
                                    np.zeros(master_ndim, dtype=int),
                                    master_shape-x_shapes[k]])
                                for k in range(x_shapes.shape[0])])
        pad_widths_eff = np.copy(pad_widths)
        pad_widths_eff[pad_widths_eff < 0] = 0

        master_slice = tuple(slice(0, d) for d in master_shape)

        kwargs.pop("pad_with", None)

        xp_a_s = [np.pad(
                            x_a_s[k],
                            pad_width=pad_widths_eff[k],
                            **kwargs
                        )[master_slice] for k in range(x_shapes.shape[0])]

    return xp_a_s


def moveaxis_destination(
                            source: Optional[Union[np.ndarray, list, tuple,]] = None,
                            destination: Optional[Union[
                                                        np.ndarray,
                                                        list,
                                                        tuple,
                                                        ]] = None,
                            ) -> np.ndarray:
    """
    Calculates the destination indices for moving axes based on the source.

    Parameters
    ----------
    source : Union[np.ndarray, list, tuple,], optional
        The source axes. If None, the default function is used to initialize an empty
        array.
        Can be of types: np.ndarray, list, or tuple.
    destination : Union[np.ndarray, list, tuple,], optional
        The destination axes. If None, it is set to the range of source size subtracted
        by source size.

    Returns
    -------
    np.ndarray
        The destination indices for moving axes.

    Examples
    --------
    >>> moveaxis_destination([0, 1, 2])
    array([-3, -2, -1])

    >>> moveaxis_destination([0, 1, 2], [2, 0, 1])
    array([2, 0, 1])
    """

    source = convert(a=source, conserve_scalar=False)
    if destination is None:
        destination = np.arange(source.size) - source.size

    return destination


def moveaxes(
                a: Optional[Union[np.ndarray, list, float, int, str]] = None,
                source: Optional[Union[np.ndarray, list, tuple,]] = None,
                destination: Optional[Union[np.ndarray, list, tuple,]] = None,
                ) -> np.ndarray:
    """
    Moves axes of an array to new positions.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, str], optional
        The input array. If None, the default function is used to initialize an empty
        array.
        Can be of types: np.ndarray, list, float, int, or str.
    source : Union[np.ndarray, list, tuple,], optional
        The original positions of the axes to move.
    destination : Union[np.ndarray, list, tuple,], optional
        The new positions of the axes.

    Returns
    -------
    np.ndarray
        The array with moved axes.

    Examples
    --------
    >>> moveaxes([[1, 2, 3], [4, 5, 6]], source=[0, 1], destination=[1, 0])
    array([[1, 4],
           [2, 5],
           [3, 6]])

    >>> moveaxes([1, 2, 3], source=[0], destination=[-1])
    array([1, 2, 3])
    """

    a = convert(a=a, conserve_scalar=False)

    source = convert(a=source, conserve_scalar=False)
    destination = moveaxis_destination(source=source, destination=destination)

    a = np.moveaxis(a, source=source, destination=destination)

    return a


def reshape_iter_pre(
                        a: Optional[Union[
                                            np.ndarray, list,
                                            float, int,
                                            bool, str,
                                        ]] = None,
                        axes: Optional[Union[np.ndarray, list, tuple,]] = None,
                        col_iter: Optional[bool] = None,
                        ) -> Tuple[np.ndarray, int, Union[np.ndarray, list, tuple,]]:
    """
    Reshapes the input array for iteration based on the specified axes.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str], optional
        The input array. If None, the default function is used to initialize an empty
        array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    axes : Union[np.ndarray, list, tuple,], optional
        The axes to move for reshaping.
    col_iter : bool, optional
        If True, the iteration is column-wise. Default is True.

    Returns
    -------
    Tuple[np.ndarray, int, Union[np.ndarray, list, tuple,]]
        The reshaped array, the size of the first dimension, and the original shape
        components.

    Examples
    --------
    >>> reshape_iter_pre([1, 2, 3], axes=[0])
    (array([[1, 2, 3]]), 1, [array([], dtype=int) array([3])])

    >>> reshape_iter_pre([[1, 2], [3, 4]])
    (array([1, 2, 3, 4]), 4, [array([2, 2]), array([], dtype=int)])

    >>> reshape_iter_pre([[1, 2], [3, 4]], axes=[1])
    (array([[1, 2],
            [3, 4]]), 2, [array([2]), array([2])])
    """

    col_iter = defaults.col_iter(col_iter)

    a = convert(a=a, conserve_scalar=False)
    shape_0 = calc_shape(a)

    if col_iter:
        axes = convert(a=axes, conserve_scalar=False)
    else:
        if axes is None:
            axes = np.arange(len(shape_0))
        axes = np.unique(axes)

    a = moveaxes(a=a, source=axes)
    shape_1 = calc_shape(a)
    shape_m = [
                shape_1[:shape_1.size-axes.size],
                shape_1[shape_1.size-axes.size:],
                ]

    if col_iter:
        N_i = np.prod(shape_m[0])
        a = np.reshape(a, np.array([N_i, *shape_m[1]], dtype=int))
    else:
        N_i = np.prod(shape_m[1])
        a = np.reshape(a, np.array([*shape_m[0], N_i], dtype=int))

    return a, N_i, shape_m


def reshape_iter_post(
                        a: Optional[Union[
                                            np.ndarray, list,
                                            float, int,
                                            bool, str,
                                        ]] = None,
                        shape_m_0: Optional[Union[np.ndarray, list, tuple,]] = None,
                        reshape: Optional[bool] = None,
                        ) -> np.ndarray:
    """
    Reshapes the input array back to its original form after iteration.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str], optional
        The input array. If None, the default function is used to initialize an empty
        array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    shape_m_0 : Union[np.ndarray, list, tuple,], optional
        The first part of the original shape of the array before iteration.
    reshape : bool, optional
        If True, reshapes the array back to its original form. Default is True.

    Returns
    -------
    np.ndarray
        The reshaped array.

    Examples
    --------
    >>> reshape_iter_post([[1], [2], [3]], shape_m_0=[3], reshape=True)
    array([[1],
           [2],
           [3]])

    >>> reshape_iter_post([1, 2, 3], shape_m_0=[1, 3], reshape=True)
    array([[1, 2, 3]])
    """

    reshape = defaults.reshape(reshape)

    a = convert(a=a, conserve_scalar=False)

    if reshape:
        shape = calc_shape(a)
        if shape_m_0 is None:
            shape_m_0 = shape[0]
        shape_m_1_post = shape[1:]
        a = np.reshape(a, np.array([*shape_m_0, *shape_m_1_post], dtype=int))

    return a


def fmap(
        a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
        f: Optional[Callable] = None,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        axes: Optional[Union[np.ndarray, list, tuple,]] = None,
        reshape: Optional[bool] = None,
        ) -> np.ndarray:
    """
    Applies a function to the input array over specified axes and reshapes the result.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str], optional
        The input array. If None, the default function is used to initialize an empty
        array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    f : Callable, optional
        The function to apply to the input array.
    args : tuple, optional
        The positional arguments to pass to the function `f`.
    kwargs : dict, optional
        The keyword arguments to pass to the function `f`.
    axes : Union[np.ndarray, list, tuple,], optional
        The axes over which to apply the function.
    reshape : bool, optional
        If True, reshapes the result back to the original form after applying the
        function. Default is True.

    Returns
    -------
    np.ndarray
        The array with the function applied and reshaped.

    Examples
    --------
    >>> fmap([1, 2, 3], f=np.mean, axes=[0])
    array([2.])

    >>> fmap([[1, 2], [3, 4]], f=np.sum, axes=[1])
    array([3, 7])
    """

    if f is not None:
        a = convert(a=a, conserve_scalar=False)

        args, kwargs = meta.default_args_kwargs(args=args, kwargs=kwargs)

        a, N_i, shape_m = reshape_iter_pre(a=a, axes=axes, col_iter=True)

        r = np.array([f(a[i], *args, **kwargs) for i in range(N_i)])

        a = reshape_iter_post(a=r, shape_m_0=shape_m[0], reshape=reshape)

    return a


def f2_axes(
            a_maj: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            a_min: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            f2: Optional[Callable] = None,
            caxes: Optional[Union[np.ndarray, list, tuple,]] = None,
            corigin: Optional[int] = None,
            cend: Optional[bool] = None,
            args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            ) -> np.ndarray:
    """
    Applies a binary function to two arrays, aligning the axes as specified.

    Parameters
    ----------
    a_maj : Union[np.ndarray, list, float, int, bool, str,], optional
        The major array to which the function is applied. If None, the default function
        is used to initialize an empty array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    a_min : Union[np.ndarray, list, float, int, bool, str,], optional
        The minor array to which the function is applied. If None, the default function
        is used to initialize an empty array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    f2 : Callable, optional
        The binary function to apply to the arrays. Default is `np.multiply`.
    caxes : Union[np.ndarray, list, tuple,], optional
        The axes along which to align the minor array with the major array.
    corigin : int, optional
        The starting index for the alignment of axes.
    cend : bool, optional
        If True, the alignment starts from the end of the axes.
    args : tuple, optional
        The positional arguments to pass to the function `f2`.
    kwargs : dict, optional
        The keyword arguments to pass to the function `f2`.

    Returns
    -------
    np.ndarray
        The result of applying the binary function to the aligned arrays.

    Examples
    --------
    >>> f2_axes([1, 2, 3], [4, 5, 6])
    array([4, 10, 18])

    >>> f2_axes([[1, 2], [3, 4]], [5, 6], caxes=[1])
    array([[5, 12],
           [15, 24]])
    """

    f2 = defaults.f2(f2)
    cend = defaults.cend(cend)

    args, kwargs = meta.default_args_kwargs(args=args, kwargs=kwargs)

    a_maj = convert(a=a_maj, conserve_scalar=False)
    a_min = convert(a=a_min, conserve_scalar=False)

    axes_maj = calc_axes(a_maj)
    axes_min = calc_axes(a_min)

    if caxes is None:
        if corigin is None:
            if cend:
                corigin = a_maj.ndim - a_min.ndim
            else:
                corigin = 0

        caxes = corigin + axes_min

    a_min = expand_dims(a=a_min, caxes=caxes, taxes=axes_maj)

    return f2(a_maj, a_min, *args, **kwargs)


def iterarr(
            a: Union[np.ndarray, list, float, int, bool, str,],
            shape: Optional[Union[np.ndarray, list, tuple,]] = None,
            dtype: Optional[np.dtype] = None,
            as_list: Optional[bool] = None,
        ) -> Union[np.ndarray, list]:
    """
    Resizes the input array to a 1D array and optionally converts it to a list.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,]
        The input array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    shape : Union[np.ndarray, list, tuple,], optional
        The desired shape for the output array.
    dtype : np.dtype, optional
        The desired data type for the output array.
    as_list : bool, optional
        If True, converts the output array to a list. Default is False.

    Returns
    -------
    Union[np.ndarray, list]
        The resized array, or list if `as_list` is True.

    Raises
    ------
    ValueError
        If the size of `a` does not match the specified output shape.

    Examples
    --------
    >>> iterarr(1)
    array([1])

    >>> iterarr("test")
    array(["test"])

    >>> iterarr([1, 2, 3])
    array([1, 2, 3])

    >>> iterarr([[1, 2, 3],
                [4, 5, 6]])
    array([1, 2, 3, 4, 5, 6])

    >>> iterarr([1, 2, 3], dtype=np.float32)
    array([1., 2., 3.], dtype=float32)

    >>> iterarr([1, 2, 3], as_list=True)
    [1, 2, 3]
    """

    as_list = defaults.as_list(as_list)

    if not np.isin([type(a)], [np.ndarray, np.matrix]):
        a = np.array(a)
    if shape is None:
        shape = a.size
    if np.prod(shape) != a.size:
        raise ValueError("Mismatch between size of a and specified output shape")

    if dtype is not None:
        a = np.array(np.resize(a, shape), dtype=dtype)
    else:
        a = np.resize(a, shape)

    if as_list:
        a = [*a]

    return a


def append(
            a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            app: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            ) -> np.ndarray:
    """
    Appends two arrays or lists and returns the concatenated array.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,], optional
        The first input array or list. If None, an empty list is used.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    app : Union[np.ndarray, list, float, int, bool, str,], optional
        The second input array or list to append. If None, an empty list is used.
        Can be of types: np.ndarray, list, float, int, bool, or str.

    Returns
    -------
    np.ndarray
        The concatenated array.

    Examples
    --------
    >>> append([1, 2, 3], [4, 5, 6])
    array([1, 2, 3, 4, 5, 6])

    >>> append(1, 2)
    array([1, 2])

    >>> append([], [1, 2, 3])
    array([1, 2, 3])
    """

    if a is None:
        a = []
    a = iterarr(a)

    if app is None:
        app = []
    app = iterarr(app)

    return np.concatenate([a, app])


def filter(
            a: Union[np.ndarray, list, float, int, bool, str,],
            a_in: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            a_rm: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            ) -> np.ndarray:
    """
    Filters elements in an array based on optional inclusion and exclusion criteria.

    Parameters
    ----------
    a : Union[np.ndarray, list, float, int, bool, str,]
        The input array.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    a_in : Union[np.ndarray, list, float, int, bool, str,], optional
        The elements to include.
        Can be of types: np.ndarray, list, float, int, bool, or str.
    a_rm : Union[np.ndarray, list, float, int, bool, str,], optional
        The elements to remove.
        Can be of types: np.ndarray, list, float, int, bool, or str.

    Returns
    -------
    np.ndarray
        The filtered array.

    Examples
    --------
    >>> filter([1, 2, 3, 4, 5], a_in=[1, 2, 3], a_rm=[2])
    array([1, 3])

    >>> filter([1, 2, 3, 4, 5], a_in=[2, 4])
    array([2, 4])

    >>> filter([1, 2, 3, 4, 5], a_rm=[2, 4])
    array([1, 3, 5])
    """

    a = iterarr(a)

    if a_in is not None:
        a_in = iterarr(a_in)
        a_in_bool = np.isin(a, a_in)
    else:
        a_in_bool = np.repeat(True, a.size)

    if a_rm is not None:
        a_rm = iterarr(a_rm)
        a_rm_bool = ~np.isin(a, a_rm)
    else:
        a_rm_bool = np.repeat(True, a.size)

    return a[np.logical_and(a_in_bool, a_rm_bool)]


def index_N(
            shape: Union[np.ndarray, list, tuple,],
            N: Optional[int] = None,
            reshape: Optional[bool] = None,
            ) -> np.ndarray:
    """
    Calculates the index array for a given shape and optionally reshapes it.

    Parameters
    ----------
    shape : Union[np.ndarray, list, tuple,]
        The shape of the array for which to calculate the index array.
    N : int, optional
        The number of elements to index. Default is 0.
    reshape : bool, optional
        If True, reshapes the index array to match the original shape. Default is True.

    Returns
    -------
    np.ndarray
        The index array.

    Examples
    --------
    >>> index_N([2, 3], N=4)
    array([1, 1])

    >>> index_N([2, 3], N=4, reshape=False)
    array([1, 1])
    """

    shape = np.array(shape, dtype=int)

    N = defaults.N(N)
    reshape = defaults.reshape(reshape)

    N_shape = np.array(np.shape(N), dtype=int)
    N_i = np.prod(N_shape, dtype=int)

    N = np.reshape(N, N_i)

    cp = np.flip(np.cumprod(np.flip(shape)))
    cp_shift = np.concatenate([cp, [1]])[1:]

    index = np.remainder.outer(N, cp)//cp_shift

    if reshape:
        index = np.reshape(index, np.concatenate([N_shape, [shape.size]]))

    return index


def N_index(
            shape: Union[np.ndarray, list, tuple,],
            index: Optional[Union[int, np.ndarray]] = None,
            ) -> int:
    """
    Calculates the linear index for a given shape and index array.

    Parameters
    ----------
    shape : Union[np.ndarray, list, tuple,]
        The shape of the array for which to calculate the linear index.
    index : Union[int, np.ndarray], optional
        The index array or integer. If None, an array of zeros is used.
        Can be of types: int or np.ndarray.

    Returns
    -------
    int
        The linear index.

    Examples
    --------
    >>> N_index([2, 3], index=[0, 1])
    1

    >>> N_index([2, 3], index=[1, 2])
    5

    >>> N_index([2, 3])
    0
    """

    shape = np.array(shape, dtype=int)

    if index is None:
        index = np.zeros(shape.size, dtype=int)

    cp = np.flip(np.cumprod(np.flip(shape)))
    cp_shift = np.concatenate([cp, [1]])[1:]

    return index @ cp_shift


def index_I_max(
                i: Optional[Union[int, np.ndarray]] = None,
                I_max: Optional[int] = None,
                flip: Optional[bool] = None,
                circle: Optional[bool] = None,
                ) -> Union[int, np.ndarray]:
    """
    Adjusts the index `i` based on maximum index `I_max`, with optional flipping and
    circular behavior.

    Parameters
    ----------
    i : Union[int, np.ndarray], optional
        The index or indices to be adjusted. If None, an array of zeros is used.
        Can be of types: int or np.ndarray.
    I_max : int, optional
        The maximum index for circular behavior.
    flip : bool, optional
        If True, flips the indices. Default is False.
    circle : bool, optional
        If True, applies circular behavior based on `I_max`. Default is True.

    Returns
    -------
    Union[int, np.ndarray]
        The adjusted index or indices.

    Examples
    --------
    >>> index_I_max(5, I_max=10)
    5

    >>> index_I_max(11, I_max=10)
    1

    >>> index_I_max(5, I_max=10, flip=True)
    4

    >>> index_I_max([1, 2, 3], I_max=5, circle=False)
    array([1, 2, 3])
    """

    flip = defaults.flip(flip)
    circle = defaults.circle(circle)

    if flip:
        i = -(1+i)

    if circle and I_max is not None:
        i = i % I_max

    return i


def capp_bias(
            X,
            ):

    if hasattr(X, "values"):
        X = X.values

    ones = np.ones((X.shape[0], 1), dtype=float)

    return np.concat([ones, X], axis=1)


def calc_val_1_0_N(
                    app_val_1: Optional[bool] = None,
                    app_val_1_default: Optional[bool] = None,
                    val_1: Optional[object] = None,
                    val_0: Optional[object] = None,
                    N: Optional[int] = None,
                    ) -> Union[np.ndarray, object,]:
    """
    Calculates values based on a boolean flag for a given number of elements.

    Parameters
    ----------
    app_val_1 : bool, optional
        The boolean flag to determine the output values. Default is inferred from
        `app_val_1_default`.
    app_val_1_default : bool, optional
        The default value for `app_val_1`. Default is True.
    val_1 : object, optional
        The value to assign when `app_val_1` is True.
        Can be of types: float, int, bool, str, or any object.
    val_0 : object, optional
        The value to assign when `app_val_1` is False.
        Can be of types: float, int, bool, str, or any object.
    N : int, optional
        The number of elements to generate.

    Returns
    -------
    Union[np.ndarray, object]
        The calculated values.

    Examples
    --------
    >>> calc_val_1_0_N(app_val_1=[True, False, True], val_1=1, val_0=0, N=3)
    array([1, 0, 1])

    >>> calc_val_1_0_N(val_1='Y', val_0='N', N=2)
    array(['Y', 'Y'])
    """

    if app_val_1 is None:
        app_val_1_default = defaults.app_val_1_default(app_val_1_default)
        app_val_1 = app_val_1_default

    app_val_1 = pad([app_val_1], master_shape=[N], mode="wrap")[0]
    val_0 = pad([val_0], master_shape=[N], mode="wrap")[0]
    val_1 = pad([val_1], master_shape=[N], mode="wrap")[0]

    val_1[np.logical_not(app_val_1)] = val_0[np.logical_not(app_val_1)]

    return val_1


def buffer_add(
                b: Optional[float] = None,
                x: Optional[np.ndarray] = None,
                ) -> np.ndarray:
    """
    Adds a buffer value to the input array.

    Parameters
    ----------
    b : float, optional
        The buffer value to add. Default is 0.0.
    x : np.ndarray, optional
        The input array. Default is an array [0, 1].

    Returns
    -------
    np.ndarray
        The buffered array.

    Examples
    --------
    >>> buffer_add(b=0.5, x=np.array([1, 2]))
    array([0.5, 2.5])

    >>> buffer_add()
    array([0. ,  1.])
    """

    x = defaults.x(x)
    b = defaults.set_default(b, 0.0)

    x_out = x + b*expand_dims(np.array([-1, 1]), axis=calc_axes(a=x)[1:])

    return x_out


def buffer(
            b: Optional[float] = None,
            x: Optional[np.ndarray] = None,
            ) -> np.ndarray:
    """
    Applies a buffer value to the input array symmetrically around its midpoint.

    Parameters
    ----------
    b : float, optional
        The buffer multiplier. Default is 1.0.
    x : np.ndarray, optional
        The input array. Default is an array [0, 1].

    Returns
    -------
    np.ndarray
        The buffered array.

    Examples
    --------
    >>> buffer(b=1.2, x=np.array([1, 2]))
    array([0.9, 2.1])

    >>> buffer()
    array([0.,  1.])
    """

    x = defaults.x(x)
    b = defaults.set_default(b, 1.0)

    mid = (x[1, ...]+x[0, ...])/2
    diff = (x[1, ...]-x[0, ...])/2

    x_out = mid + b*expand_dims(np.array([-1, 1]), axis=calc_axes(a=x)[1:])*diff

    return x_out


def collapse_same(
        val,
        app_tolist: Optional[bool] = None,
        ):

    if not np.isscalar(val):
        if np.ndim(val) == 0:
            app_tolist = True
        else:
            try:
                uniques = np.unique(val)
                same = len(uniques) == 1
            except:     # noqa: E722
                same = False

            if same:
                val = uniques[0]
                app_tolist = False
            else:
                if app_tolist is None:
                    app_tolist = False

        if app_tolist:
            if hasattr(val, "tolist"):
                val = val.tolist()

    return val
