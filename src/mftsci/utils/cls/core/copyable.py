
"""
Core module for config_cls.core.copyable
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages

# Internal packages
from ...meta import (
                                is_modified,
                                )


# =====================================
# CLASSES
# =====================================


class Copyable:

    _stable = True
    _origin = None

    @classmethod
    def _clscopy(
            cls,
            registry=None,
            suffix=None,
            defs=None,
            **kwargs,
            ):

        if registry is None:
            registry = {}

        if defs is None:
            defs = {}

        if cls in registry:
            return registry[cls]

        if suffix is None:
            suffix = "Copy"

        ncls = type(
                    f"{cls.__name__}{suffix}",
                    (cls,),
                    defs,
                    )
        ncls._stable = False
        ncls._origin = cls

        ncls = cls._clscopy_process(
                        ncls=ncls, registry=registry, suffix=suffix,
                        **kwargs,
                        )
        if (ncls.__module__[:9] != "__main__") or (not suffix):
            ncls.__module__ = ".".join(["__main__", cls.__module__])

        return ncls

    @classmethod
    def _clscopy_process(
            cls,
            ncls,
            registry=None,
            suffix=None,
            ):

        return ncls

    @classmethod
    def _clsorigin(cls):

        _olist = []
        org = cls._origin
        while org is not None:
            _olist.append(org)
            org = org._origin
        return _olist

    @classmethod
    def _is_stable(cls):

        if cls.__module__[:8] == "__main__":
            cls._stable = False

        return cls._stable


# =====================================
# FUNCTIONS
# =====================================


def is_stable(cls):

    if isinstance(cls, type) and issubclass(cls, Copyable):
        return cls._is_stable()
    else:
        return not is_modified(cls)


def make_key(
        cls,
        _is_stable=None,
        ):

    if _is_stable is None:
        _is_stable = is_stable(cls)

    if _is_stable:
        _m = f"{cls.__module__}."
    else:
        _m = f"runtime:{cls.__module__}."
    return f"{_m}{cls.__name__}"
