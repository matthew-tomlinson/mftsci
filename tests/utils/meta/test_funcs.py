"""
Unit test module for mftsci.utils.magic.funcs.
"""
# =====================================
# IMPORT
# =====================================
# Built-in modules

# External packages
import pytest
import numpy as np

# Internal packages
from mftsci.utils.meta import *


# =====================================
# UNIT TESTS
# =====================================


def test_pack_args_kwargs():
    # Test case 1: Mixed positional and keyword arguments
    assert pack_args_kwargs(1, 2, a=3, b=4) == ((1, 2), {'a': 3, 'b': 4})


def test_pack_args_kwargs_kw():
    # Test case 1: Mixed positional and keyword arguments
    assert pack_args_kwargs_kw(1, 2, a=3, b=4) == \
            {'args': (1, 2), 'kwargs': {'a': 3, 'b': 4}}


def test_default_args():
    # Test case 1: Tuple of arguments
    assert default_args((1, 2)) == (1, 2)
    # Test case 2: None as value
    assert default_args(None) == ()


def test_default_kwargs():
    # Test case 1: Dictionary and additional keyword arguments
    assert default_kwargs({'a': 1}, b=2) == {'a': 1, 'b': 2}
    # Test case 2: None as value and additional keyword arguments
    assert default_kwargs(None, a=1) == {'a': 1}


def test_default_args_kwargs():
    # Test case 1: Tuple and dictionary of arguments
    assert default_args_kwargs((1,), {'a': 2}, b=3) == ((1,), {'a': 2, 'b': 3})
    # Test case 2: None as both arguments and additional keyword arguments
    assert default_args_kwargs(None, None, a=1) == ((), {'a': 1})


def test_get_ovarnames():
    # Test case 1: Function with multiple arguments
    def example_func(a, b, c=3): pass
    assert get_ovarnames(example_func) == ('a', 'b', 'c')


def test_get_okdefaults():
    # Test case 1: Function with keyword-only arguments
    def example_func(a, b, *, c=3, d=4): pass
    assert get_okdefaults(example_func) == {'c': 3, 'd': 4}


def test_merge_args_kwargs():
    # Test case 1: Merging positional and keyword arguments into function parameters
    def example_func(a, b): pass
    assert merge_args_kwargs(_func=example_func,
                             args=(1,), kwargs={'b': 2}) == {'a': 1, 'b': 2}


def test_set_defaults():
    # Test case 1: Setting default values and callable defaults
    assert set_defaults(_defaults={'a': 1},
                        _defaults_call={'b': lambda: 2},
                        kwargs={'c': 3}) == {'a': 1, 'b': 2, 'c': 3}


def test_attr():
    # Test case 1: Retrieving an attribute
    class Example:
        def __init__(self):
            self.value = 42
    example = Example()
    assert attr(example, 'value') == 42
    # Test case 2: Returning the object itself
    assert attr(example) == example


def test_fattr():
    # Test case 1: Calling an attribute method
    class Example:
        def __init__(self):
            self.value = 42
        def add(self, x):
            return self.value + x
    example = Example()
    assert fattr(example, 'add', args=(8,)) == 50
    # Test case 2: Returning the object itself
    assert fattr(example) == example


def test_get_magic_method_names():
    # Test case 1: Retrieving all magic method names
    assert get_magic_method_names() == {'noval': ['__str__', '__repr__'],
                                        'val': ['__add__', '__sub__', '__mul__',
                                                '__truediv__', '__mod__', '__pow__'],
                                        'val_bool': ['__eq__', '__lt__', '__le__',
                                                     '__gt__', '__ge__']}
    # Test case 2: Retrieving magic method names for a specific key
    assert get_magic_method_names('val') == ['__add__', '__sub__', '__mul__',
                                             '__truediv__', '__mod__', '__pow__']


def test_wraps_ovar():
    # Test case 1: Basic functionality
    @wraps_ovar()
    def example_func(a, b): pass
    assert example_func.__ovarnames__ == ('a', 'b')


def test_defaults_kwargs():
    # Test case 1: Setting default keyword arguments
    @defaults_kwargs(_defaults={'a': 1}, _defaults_call={'b': lambda: 2})
    def example_func(a, b):
        return a + b
    assert example_func() == 3


def test_metafunc_decorator():
    # Test case 1: Applying a metafunction to the output of a function
    @metafunc_decorator(lambda x: x * 2)
    def example_func(a, b):
        return a + b
    assert example_func(1, 2) == 6


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
