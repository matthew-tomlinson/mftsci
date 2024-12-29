"""
Module for array utilities.
"""
# =====================================
# IMPORT
# =====================================
# Native packages
from typing import Optional, Union, Callable, Tuple

# External packages
import numpy as np

# Internal packages
from mftsci.utils.core import *     # noqa: F401, F403
from mftsci.utils import meta


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


_attr_vector = np.vectorize(meta.attr, excluded=["attr", "args", "kwargs"])
_fattr_vector = np.vectorize(meta.fattr, excluded=["args", "kwargs"])


def attr(
            a: Union[np.ndarray, list, float, int, bool, str,],
            attr: Optional[str] = None,
            ) -> object:

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

    return _fattr_vector(
                        obj=a,
                        attr=attr,
                        args=args,
                        kwargs=kwargs,
                        )


def default(
            a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
            ) -> np.ndarray:

    if a is None:
        a = []
    if np.isscalar(a):
        a = [a]
    a = np.array(a)

    return a


def calc_shape(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                ) -> np.ndarray:

    if a is None:
        a = default(a=a)
    shape = np.array(np.shape(a))

    return shape


def calc_axes(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                ndim: Optional[int] = None,
                ) -> np.ndarray:

    if ndim is None:
        if shape is None:
            if np.isscalar(a):
                ndim = 0
            else:
                ndim = default(a=a).ndim
        else:
            ndim = len(shape)

    axes = np.arange(ndim)

    return axes


def calc_length(
                a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
                shape: Optional[Union[np.ndarray, list, tuple,]] = None,
                ) -> int:

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

    if appear_order is None:
        appear_order = True

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

    if expand_back is None:
        expand_back = False

    a = default(a=a)
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

    if a_s is None:
        a_s = default()
    if type(a_s) is not list:
        a_s = [a_s]

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

        xp_a_s = [np.pad(
                            x_a_s[k],
                            pad_width=pad_widths_eff[k],
                            *args, **kwargs
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

    source = default(a=source)
    if destination is None:
        destination = np.arange(source.size) - source.size

    return destination


def moveaxes(
                a: Optional[Union[np.ndarray, list, float, int, str]] = None,
                source: Optional[Union[np.ndarray, list, tuple,]] = None,
                destination: Optional[Union[np.ndarray, list, tuple,]] = None,
                ) -> np.ndarray:

    a = default(a=a)

    source = default(a=source)
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

    if col_iter is None:
        col_iter = True

    a = default(a=a)
    shape_0 = calc_shape(a)

    if col_iter:
        axes = default(a=axes)
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

    if reshape is None:
        reshape = True

    a = default(a=a)

    if reshape:
        shape = calc_shape(a)
        if shape_m_0 is None:
            shape_m_0 = shape[0]
        shape_m_1_post = shape[1:]
        a = np.reshape(a, np.array([*shape_m_0, *shape_m_1_post], dtype=int))

    return a


def f(
        a: Optional[Union[np.ndarray, list, float, int, bool, str,]] = None,
        f: Optional[Callable] = None,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        axes: Optional[Union[np.ndarray, list, tuple,]] = None,
        reshape: Optional[bool] = None,
        ) -> np.ndarray:

    if f is not None:
        a = default(a=a)

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

    if f2 is None:
        f2 = np.multiply

    if cend is None:
        cend = False

    args, kwargs = meta.default_args_kwargs(args=args, kwargs=kwargs)

    a_maj = default(a=a_maj)
    a_min = default(a=a_min)

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

    if as_list is None:
        as_list = False

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

    shape = np.array(shape, dtype=int)

    if N is None:
        N = 0

    if reshape is None:
        reshape = True

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

    if flip is None:
        flip = False

    if circle is None:
        circle = True

    if flip:
        i = -(1+i)

    if circle and I_max is not None:
        i = i % I_max

    return i


def calc_val_1_0_N(
                    app_val_1: Optional[bool] = None,
                    app_val_1_default: Optional[bool] = None,
                    val_1: Optional[object] = None,
                    val_0: Optional[object] = None,
                    N: Optional[int] = None,
                    ) -> Union[np.ndarray, object,]:

    if app_val_1 is None:
        if app_val_1_default is None:
            app_val_1_default = True
        app_val_1 = app_val_1_default

    app_val_1 = pad(app_val_1, master_shape=[N], mode="wrap")[0]
    val_0 = pad(val_0, master_shape=[N], mode="wrap")[0]
    val_1 = pad(val_1, master_shape=[N], mode="wrap")[0]

    val_1[np.logical_not(app_val_1)] = val_0[np.logical_not(app_val_1)]

    return val_1


def buffer_add(
                b: Optional[float] = None,
                x: Optional[np.ndarray] = None,
                ) -> np.ndarray:

    if x is None:
        x = np.array([0, 1])
    if b is None:
        b = 0.0

    x_out = x + b*expand_dims(np.array([-1, 1]), axis=calc_axes(a=x)[1:])

    return x_out


def buffer(
            b: Optional[float] = None,
            x: Optional[np.ndarray] = None,
            ) -> np.ndarray:

    if x is None:
        x = np.array([0, 1])
    if b is None:
        b = 1.0

    mid = (x[1, ...]+x[0, ...])/2
    diff = (x[1, ...]-x[0, ...])/2

    x_out = mid + b*expand_dims(np.array([-1, 1]), axis=calc_axes(a=x)[1:])*diff

    return x_out
