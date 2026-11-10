"""
Defaults module for random utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union
import datetime
import string

# External packages
import pandas as pd

# Internal packages
from mftsci.utils.meta import set_default
from mftsci.utils.meta.defaults import reset    # noqa: F401
from mftsci.utils.str import str_to_int    # noqa: F401

# =====================================
# DEFAULTS
# =====================================


def seed(
       seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
        ) -> Union[int, float, str, bytes, bytearray]:

    return set_default(
            val=seed,
            default=str(pd.Timestamp(datetime.datetime.now(datetime.timezone.utc))),
            )


def seed_int(
       seed: Optional[Union[int, str]] = None,
        ) -> Union[int, str]:

    seed = set_default(
            val=seed,
            default=int(
                        (
                            pd.Timestamp(datetime.datetime.now(datetime.timezone.utc))
                            - pd.Timestamp(0, tz="utc")
                            ) / pd.Timedelta(seconds=1E-6)
                            )
            )

    if isinstance(seed, str):
        seed = str_to_int(seed)

    return seed


def source_str(
       source: Optional[str] = None,
        ) -> str:

    return set_default(val=source, default=string.ascii_uppercase)


def length_str(
       length: Optional[str] = None,
        ) -> str:

    return set_default(val=length, default=6)
