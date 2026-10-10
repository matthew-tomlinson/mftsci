"""
Functions module for random utilities.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules
from typing import Optional, Union
import random

# External packages
import numpy as np

# Internal packages
from mftsci.utils.random import defaults
from mftsci.utils.str import concat as _str_concat, str_to_int  # noqa: F401


# =====================================
# CLASSES
# =====================================


class RandomGenerator():

    def __init__(
                    self,
                    seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
                ):

        self.set_rng(seed=seed)

    def set_rng(
                    self,
                    seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
                ):

        seed = defaults.seed(seed)
        self.rng = random.Random(x=seed)


class RandomGeneratorNp():

    def __init__(
                    self,
                    seed: Optional[int] = None,
                ):

        self.set_rng(seed=seed)

    def set_rng(
                    self,
                    seed: Optional[Union[int, str]] = None,
                ):

        seed = defaults.seed_int(seed)
        self.rng = np.random.default_rng(seed=seed)


class _BaseStringGenerator():

    def __init__(
                    self,
                    source: Optional[str] = None,
                    default_length: Optional[int] = None,
                ):

        self.set_source(source=source, reset=True)
        self.set_default_length(default_length=default_length, reset=True)

    def set_source(
                self,
                source: Optional[str] = None,
                reset: Optional[bool] = None,
            ):

        if source is not None:
            self.source = source
        elif defaults.reset(reset):
            self.source = defaults.source_str()

    def set_default_length(
                self,
                default_length: Optional[int] = None,
                reset: Optional[bool] = None,
            ):

        if default_length is not None:
            self.default_length = default_length
        elif defaults.reset(reset):
            self.default_length = defaults.length_str()

    def gen_strings(
                    self,
                    length: Optional[int] = None,
                    N: Optional[int] = None,
                    ):

        length = defaults.set_default(val=length, default=self.default_length)
        N = defaults.set_default(val=N, default=1)

        return ["".join(self.rng.choices(self.source, k=length)) for n in range(N)]


class RandomStringGenerator(_BaseStringGenerator, RandomGenerator):

    def __init__(
                    self,
                    seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
                    source: Optional[str] = None,
                    default_length: Optional[int] = None,
                ):

        RandomGenerator.__init__(self, seed=seed)
        _BaseStringGenerator.__init__(
                                        self,
                                        source=source,
                                        default_length=default_length,
                                        )


class RandomStringGeneratorNp(_BaseStringGenerator, RandomGeneratorNp):

    def __init__(
                    self,
                    seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
                    source: Optional[str] = None,
                    default_length: Optional[int] = None,
                ):

        RandomGeneratorNp.__init__(self, seed=seed)
        self.set_default_length(default_length=default_length, reset=True)
        self.set_source(source=source, reset=True)

    def set_source(
            self,
            source: Optional[str] = None,
            reset: Optional[bool] = None,
            ):

        if source is not None:
            if isinstance(source, str):
                source = [*source]
            self.source = source
        elif defaults.reset(reset):
            self.source = [*defaults.source_str()]

    def gen_strings(
                    self,
                    length: Optional[int] = None,
                    N: Optional[int] = None,
                    ):

        length = defaults.set_default(val=length, default=self.default_length)
        N = defaults.set_default(val=N, default=1)

        return _str_concat(self.rng.choice(a=self.source, size=(N, length)), axes=1)


# =====================================
# FUNCTIONS
# =====================================
