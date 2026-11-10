
"""
Core module for config_cls.core.configurable
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages

# Internal packages
from ...meta import hybridmethod
from ...dict import walk
from .exportable import (     # noqa: F401
                        Exportable, Copyable,
                        is_stable, make_key,
                        _export_spec_visit,
                        export_spec, import_spec,
                        )
from .context import Context


# =====================================
# CLASSES
# =====================================

class Configurable(Exportable):

    _config = {}
    _shared = set()

    # ------------ CLASS-LEVEL CONFIG ------------
    @classmethod
    def _is_stable(cls):
        _is_stable = super()._is_stable()
        if not _is_stable:
            return _is_stable
        else:
            for v in cls._config.values():
                _is_stable = is_stable(v)
                if not _is_stable:
                    return _is_stable
            return _is_stable

    @classmethod
    def _clscopy_process(
            cls,
            ncls=None,
            registry=None,
            suffix=None,
            copy_child=None,
            **kwargs,
            ):

        if copy_child is None:
            copy_child = True

        if copy_child:
            new_config = {}
            for name, value in getattr(cls, "_config", {}).items():
                if isinstance(value, type) and issubclass(value, Configurable):
                    new_value = value._clscopy(registry=registry, suffix=suffix)
                else:
                    new_value = value

                new_config[name] = new_value

            ncls._config = new_config
            ncls._shared = set(getattr(cls, "_shared", set()))

        return ncls

    @classmethod
    def config(
            cls,
            kwargs=None,
            new_class=None,
            suffix=None,
            copy_child=None,
            app_pass=None,
            inherit=None,
            ):

        if new_class is None:
            new_class = False

        if new_class:
            ncls = cls._clscopy(
                        suffix=suffix,
                        copy_child=copy_child,
                        )
            ncls.config(
                kwargs=kwargs,
                app_pass=app_pass,
                inherit=inherit,
                new_class=False,
                )

            return ncls

        if kwargs is None:
            kwargs = {}

        if len(kwargs) == 0:
            return cls

        if app_pass is None:
            app_pass = False

        if inherit is None:
            inherit = False

        if inherit:
            kwargs = {k: v for k, v in kwargs.items() if k in cls._config}

        for k, v in kwargs.items():
            use_k = True
            if isinstance(k, tuple):
                use_k = False
                if len(k) > 0:
                    k_0 = k[0]
                    if len(k) == 1:
                        use_k = True
                        k = k_0
                    elif k_0 in cls._config:
                        cls._config[k_0].config(
                            kwargs={tuple(k[1:]): v},
                            app_pass=app_pass,
                            inherit=inherit,
                            new_class=False,
                            )
            if use_k:
                if k in cls._config:
                    cls._config[k] = v

        if app_pass:
            for k_1 in cls._config.keys():
                if issubclass(cls._config[k_1], Configurable):
                    cls._config[k_1].config(
                        kwargs=kwargs,
                        app_pass=True,
                        inherit=inherit,
                        new_class=False,
                        )

        return cls

    @hybridmethod
    def typecheck(
            cls,
            object,
            ):

        if isinstance(cls, type):
            tcls = cls
        else:
            tcls = type(cls)

        if isinstance(object, type):
            return issubclass(object, tcls)
        else:
            return isinstance(object, tcls)

    @hybridmethod
    def _get_config_hybrid(
            cls,
            object,
            ):

        _config = getattr(object, "_config", None)

        if not isinstance(cls, type):
            if _config is not None:
                _config = {
                            k: getattr(object, k, None)
                            for k in _config if hasattr(object, k)
                            }

        return _config

    @hybridmethod
    def walk(
            cls,
            path=None,
            f_path_leaf=None,
            cond_path_leaf_res=None,
            path_to_key=None,
            result=None,
            leaf_only=None,
            at_empty=None,
            get_store=None,
            is_node=None,
            is_leaf=None,
            leaf_if_no_store=None,
            force_dag=None,
            app_to_srs=None,
            init_depth=None,
            max_rdepth=None,
            null_key=None,
            f_kwargs=None,
            cond_kwargs=None,
            ):

        if get_store is None:
            get_store = cls._get_config_hybrid

        if at_empty is None:
            at_empty = "i"

        return walk(
                    node=cls,
                    path=path,
                    f_path_leaf=f_path_leaf,
                    cond_path_leaf_res=cond_path_leaf_res,
                    path_to_key=path_to_key,
                    result=result,
                    leaf_only=leaf_only,
                    at_empty=at_empty,
                    get_store=get_store,
                    is_node=is_node,
                    is_leaf=is_leaf,
                    leaf_if_no_store=leaf_if_no_store,
                    force_dag=force_dag,
                    app_to_srs=app_to_srs,
                    init_depth=init_depth,
                    max_rdepth=max_rdepth,
                    null_key=null_key,
                    f_kwargs=f_kwargs,
                    cond_kwargs=cond_kwargs,
                    )

    # ------------ INSTANCE BUILD ------------

    @classmethod
    def _export_spec_visit_process(
            cls,
            spec=None,
            registry=None
            ):

        spec = super()._export_spec_visit_process(spec)
        _config = cls._export_spec_visit_config(registry=registry)
        if _config:
            spec["config"] = _config

        return spec

    @classmethod
    def _export_spec_visit_config(
                cls,
                registry=None,
                ):
        if not hasattr(cls, "_config"):
            return None
        out = {
                k: _export_spec_visit(
                        v, registry=registry
                        )
                for k, v in cls._config.items()
                }
        return out

    @classmethod
    def _import_spec_process(
            cls,
            entry=None,
            registry=None,
            ):

        super()._import_spec_process(entry=entry, registry=registry)
        if entry is not None and registry is not None:
            _config = entry.get("config", {})
            if _config:
                cls._config = {
                                k: registry[v]
                                for k, v in _config.items()
                                }

    # ------------ INSTANCE BUILD ------------

    def __init__(
            self,
            _context=None,
            _path="",
            ):

        if _context is None:
            _context = Context()

        self._context = _context
        self._path = _path
        self._instances = {}

        cls = type(self)

        for name, cls_value in cls._config.items():
            if name in cls._shared:
                if name in self._context.store:
                    value = self._context.get(name)
                else:
                    value = self._init(cls_value, name)
                    self._context.set(name, value)
            else:
                path = f"{_path}.{name}" if _path else name
                value = self._init(cls_value, path)
                self._instances[name] = value

    def _init(
            self,
            cls_value,
            path,
            ):

        if isinstance(cls_value, type):
            if issubclass(cls_value, Configurable):
                value = cls_value(
                            _context=self._context,
                            _path=path,
                            )
            else:
                value = cls_value()
        else:
            value = cls_value

        return value

    @hybridmethod
    def get(
            cls,
            name,
            ):

        found_name = False
        if hasattr(cls, "_instances"):
            if name in cls._instances:
                val = cls._instances.get(name)
                found_name = True
        if not found_name:
            val = cls._config.get(name)

        return val

    def set(self, name, value):
        if name not in type(self)._shared:
            raise ValueError(f"{name} is not shared")
        self._context.set(name, value)

    def __getattr__(self, name):

        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            try:
                cls = type(self)
                if name in cls._shared:
                    context = object.__getattribute__(self, "_context")
                    return context.get(name)
                else:
                    instances = object.__getattribute__(self, "_instances")
                return instances.get(name)
            except AttributeError as e1:    # noqa: F841
                pass
            raise e

    def __setattr__(self, name, value):
        if hasattr(self, "_instances"):
            if name in self._instances:
                self._instances[name] = value
                return

        object.__setattr__(self, name, value)
        return


# =====================================
# FUNCTIONS
# =====================================
