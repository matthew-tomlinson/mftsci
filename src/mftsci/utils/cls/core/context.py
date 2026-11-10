
"""
Core module for config_cls.core.context
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages

# Internal packages


# =====================================
# CLASSES
# =====================================


class Context:
    def __init__(
            self,
            parent=None,
            ):
        self.store = {}
        self.parent = parent

    def get(
            self,
            key,
            ):
        if key in self.store:
            return self.store[key]
        if self.parent:
            return self.parent.get(key)
        raise KeyError(key)

    def set(self, key, value):
        self.store[key] = value
