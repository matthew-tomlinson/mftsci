"""
Defaults module for sklearn.base.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Literal

# External packages

# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401


# =====================================
# DEFAULTS
# =====================================

def ftype(
        ftype: Optional[Literal["o", "i"]] = None,
        ) -> Literal["o", "i"]:

    ftype = set_default(ftype, "o")
    return str(ftype)[:1].lower()
