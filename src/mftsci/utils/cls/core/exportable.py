
"""
Core module for config_cls.core.exportable
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
import importlib
import json

# External packages
import numpy as np

# Internal packages
from .copyable import Copyable, is_stable
from ...json import to_json, from_json


# =====================================
# CLASSES
# =====================================

class Exportable(Copyable):

    @classmethod
    def export_spec(
            cls,
            registy=None,
            ):
        return export_spec(cls=cls, registry=registy)

    @classmethod
    def _export_spec_visit_process(
            cls,
            spec=None,
            ):
        if spec is None:
            spec = {}
        state = cls._export_spec_visit_state()
        if state:
            spec["state"] = to_json(state)

        return spec

    @classmethod
    def _export_spec_visit_state(cls):
        state = {}
        return state

    @classmethod
    def save(cls, path):
        with open(path, "w") as f:
            json.dump(export_spec(cls), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            spec = json.load(f)

        return cls.import_spec(spec)

    @classmethod
    def _import_spec_set_state(
            cls,
            key,
            value,
            ):

        setattr(cls, key, value)

    @classmethod
    def _import_spec_process(
            cls,
            entry=None,
            registry=None,
            ):

        if entry is not None:
            _state = entry.get("state", {})
            if _state:
                _state = from_json(_state)
                for k, v in _state.items():
                    cls._import_spec_set_state(key=k, value=v)

        return entry

    @classmethod
    def import_spec(
            cls,
            spec,
            ):

        return import_spec(spec=spec)


# =====================================
# FUNCTIONS
# =====================================


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


def _export_spec_visit(
        cls,
        registry=None,
        ):

    if registry is None:
        registry = {}

    key = make_key(cls)
    if key in registry:
        return key

    _is_stable = is_stable(cls)

    if _is_stable:
        registry[key] = {"type": "ref"}
    else:
        registry[key] = {"type": "inline"}
        if isinstance(cls, type):
            registry[key]["bases"] = tuple([
                                    _export_spec_visit(b, registry=registry)
                                    for b in cls.__bases__
                                    ])
            is_exportable = issubclass(cls, Exportable)
            if is_exportable:
                registry[key] = cls._export_spec_visit_process(
                                            spec=registry[key],
                                            registry=registry,
                                            )

    return key


def export_spec(
        cls,
        registry=None,
        ):

    if registry is None:
        registry = {}

    key = _export_spec_visit(cls=cls, registry=registry)

    return {"root": key, "registry": registry}


def import_spec(spec):

    registry = {}
    _inline = {}
    _name = {}
    _module = {}

    # first pass, create classes
    for key, entry in spec["registry"].items():

        _inline[key] = entry["type"] == "inline"

        if _inline[key]:
            _module[key], _name[key] = key.split(":")[1].rsplit(".", 1)
        else:
            _module[key], _name[key] = key.rsplit(".", 1)
            try:
                cls = getattr(importlib.import_module(_module[key]), _name[key])
            except Exception:
                _inline[key] = True
        if _inline[key]:
            cls = None
        registry[key] = cls

    else:
        _keys = np.array([k for k in registry.keys() if registry[k] is None])
        runtime_dep = np.array([
                        np.isin(_keys, spec["registry"][k]["bases"])
                        for k in _keys
                        ])

        N_key = _keys.size

        gen = np.zeros(N_key, dtype=int)
        at_end_0 = gen.astype(bool)

        for i in range(N_key):
            at_end = ~runtime_dep[:, ~at_end_0].any(axis=1)
            if np.all(at_end == at_end_0):
                raise Exception("Missing link!")
            gen[at_end] -= 1
            if np.all(at_end):
                break
            at_end_0 = at_end

        gen -= np.min(gen)
        _keys = _keys[np.argsort(gen)]

    for key in _keys:
        entry = spec["registry"][key]
        cls = registry[key]
        _bases = entry.get("bases", tuple())
        bases = tuple(registry[b] for b in _bases)
        cls = type(_name[key], bases, {})
        cls.__module__ = _module[key]
        registry[key] = cls

    for key in _keys:
        entry = spec["registry"][key]
        cls = registry[key]
        if isinstance(cls, type) and issubclass(cls, Exportable):
            cls._import_spec_process(entry=entry, registry=registry)

    return registry[spec["root"]]
