"""
Functions module for json utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Literal

# External packages
import numpy as np
import pandas as pd

# Internal packages
from mftsci.utils.json import defaults  # noqa: F401
from mftsci.utils.meta import require_optional
from mftsci.utils.arr import collapse_same
from mftsci.utils.pd import to_subdtype


# =====================================
# CLASSES
# =====================================


# =====================================
# FUNCTIONS
# =====================================


def to_json(
        val: object,
        keep_np: Optional[bool] = None,
        pd_keep_index: Optional[bool] = None,
        pd_keep_columns: Optional[Literal[True, "list", False,]] = None,
        ):

    vdict = {}

    if keep_np is None:
        keep_np = True

    if pd_keep_index is None:
        pd_keep_index = True

    if pd_keep_columns is None:
        pd_keep_columns = True

    if isinstance(val, type):
        vdict["__class__"] = "type"
        vdict["__module__"] = val.__module__
        vdict["__name__"] = val.__name__
    else:
        cls = type(val)
        module = cls.__module__
        name = cls.__name__

        if isinstance(val, dict):
            val = {k: to_json(v) for k, v in val.items()}
        elif isinstance(val, list):
            val = [to_json(v) for v in val]
        elif isinstance(val, tuple):
            val = tuple([to_json(v) for v in val])
        elif module == "builtins":
            pass
        elif isinstance(val, pd.Index):
            if isinstance(val, pd.MultiIndex):
                if val.nlevels == 1:
                    val = val.get_level_values(0)

            if isinstance(val, pd.MultiIndex):
                if False:
                    vdict["__class__"] = "pandas.MultiIndex"
                    vdict["__init__"] = "from_tuples"
                    vdict["__kwargs__"] = {
                                        "tuples": to_json(val.values, keep_np=False),
                                        "names": to_json(val.names),
                                        }
                else:
                    vdict["__class__"] = "pandas.MultiIndex"
                    vdict["__init__"] = "from_frame"
                    vdict["__kwargs__"] = {
                                            "df": to_json(
                                                val.to_frame().reset_index(drop=True),
                                                pd_keep_index=False,
                                                pd_keep_columns="list"
                                                ),
                                            }
            else:
                vdict["__class__"] = "pandas.Index"
                vdict["__kwargs__"] = {
                                        "dtype": str(val.dtype),
                                        "data": to_json(
                                                    to_subdtype(val).values,
                                                    keep_np=False,
                                                    ),
                                        "name": to_json(val.name),
                                    }

        elif isinstance(val, pd.DataFrame):
            vdict["__class__"] = "pandas.DataFrame"

            object_cols = val.dtypes == object
            val.loc[:, object_cols] = val.loc[:, object_cols].map(
                                        to_json,
                                        )
            vdict["__kwargs__"] = {}
            if pd_keep_index:
                vdict["__kwargs__"]["index"] = to_json(val.index)
            if pd_keep_columns:
                if pd_keep_columns == "list":
                    vdict["__kwargs__"]["columns"] = to_json([*val.columns])
                else:
                    vdict["__kwargs__"]["columns"] = to_json(val.columns)
            vdict["__kwargs__"]["dtype"] = collapse_same(
                                            val.dtypes.astype(str),
                                            app_tolist=True,
                                            )
            vdict["__kwargs__"]["data"] = to_json(
                                            to_subdtype(val).values,
                                            keep_np=False,
                                            )

        elif isinstance(val, pd.Series):
            vdict["__class__"] = "pandas.Series"
            if val.dtype == object:
                val = val.apply(to_json)
            vdict["__kwargs__"] = {}
            if pd_keep_index:
                vdict["__kwargs__"]["index"] = to_json(val.index)
            vdict["__kwargs__"]["dtype"] = str(val.dtype)
            vdict["__kwargs__"]["data"] = to_json(
                                            to_subdtype(val).values,
                                            keep_np=False,
                                            )

        elif hasattr(val, "tolist") and module.split(".", 1)[0] == "numpy":
            if not np.isscalar(val) and keep_np:
                if np.ndim(val) > 0:
                    name = cls.__name__
                    if name == "ndarray":
                        name = "array"
                    vdict["__class__"] = ".".join([cls.__module__, name])
            vdtype = getattr(val, "dtype", None)
            val = val.tolist()

            if vdtype == object:
                val = to_json(val)
            if len(vdict) > 0:
                if vdict["__class__"] == "numpy.array":
                    vdict["__kwargs__"] = {"dtype": str(vdtype)}
                vdict["__args__"] = (val,)
        else:
            vdict["__class__"] = ".".join([module, name])
            vdict["__module__"] = module
            vdict["__name__"] = name
            if hasattr(val, "__dict__"):
                vdict["__dict__"] = to_json(val.__dict__)

    if len(vdict) > 0:
        return vdict
    else:
        return val


def from_json(
        json,
        ):

    if isinstance(json, dict):
        _type = json.get("__class__", None)
        if _type is not None:
            _type = str(_type).rsplit(".", 1)
            reset = True
            if len(_type) > 1:
                # try:
                cls = require_optional(*_type)
                _init = json.get("__init__", None)
                if _init is not None:
                    cls = getattr(cls, _init, cls)

                args = from_json(json.get("__args__", ()))
                kwargs = from_json(json.get("__kwargs__", {}))
                dict_ = json.get("__dict__", None)
                dtype = kwargs.pop("dtype", None)
                try:
                    json = cls(
                                *args,
                                **kwargs,
                                )
                except:     # noqa: E722
                    json = cls.__new__(cls)

                if dict_ is not None:
                    for k, v in from_json(dict_).items():
                        setattr(json, k, v)

                if hasattr(json, "astype"):
                    if dtype is not None:
                        if not np.isscalar(dtype) and isinstance(json, pd.DataFrame):
                            dtype = pd.Series(dtype, index=json.columns)
                        json = json.astype(dtype)

                reset = False

            if _type[0] == "type":
                return require_optional(*[json[v] for v in ["__module__", "__name__"]])

            if reset:
                _type = None
            else:
                if isinstance(json, pd.DataFrame):
                    object_cols = json.dtypes == object
                    json.loc[:, object_cols] = json.loc[:, object_cols].map(from_json)
                elif isinstance(json, pd.Series):
                    if json.dtype == object:
                        json = json.apply(from_json)

        if _type is None:
            json = {k: from_json(v) for k, v in json.items()}

    elif isinstance(json, list):
        json = [from_json(v) for v in json]
    elif isinstance(json, tuple):
        json = tuple([from_json(v) for v in json])

    return json
