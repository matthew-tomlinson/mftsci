"""
Defaults module for html utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union

# External packages

# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401


# =====================================
# DEFAULTS
# =====================================


def margin(
       margin: Optional[Union[float, int, str,]] = None,
        ) -> Union[Union[float, int, str,]]:

    return set_default(val=margin, default=_margin)


def padding(
       app_master: Optional[Union[float, int, str,]] = None,
        ) -> Union[Union[float, int, str,]]:

    return set_default(val=padding, default=0)


# =====================================
# VARIABLES
# =====================================

_margin = 0
