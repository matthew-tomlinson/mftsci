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
from mftsci.utils.magic import *
from mftsci.utils.magic import _mprop_novalue, _mprop_value, _setmprop, \
                                _setmpropnames, _setmpropmethods


# =====================================
# CLASSES
# =====================================

class MagicClass(MagicProp):

    def __init__(self):

        super.__init__()
        self.prop = 0


_setmprop(
            _class=MagicClass, mprop="prop",
            ret_class_val=True, ret_class_val_bool=False,
            )


# =====================================
# FIXTURES
# =====================================


@pytest.fixture
def magic_prop():
    """Fixture to create a MagicProp instance."""
    return MagicProp()


@pytest.fixture
def magic_class():
    """Fixture to create a MagicProp instance."""
    return MagicClass()


# =====================================
# CLASS TESTS
# =====================================


def test_init(magic_prop):
    """Test the initialization of the MagicProp class."""
    assert isinstance(magic_prop, MagicProp)
    # assert isinstance(magic_class, MagicClass)


def test_mprop(magic_prop):
    """Test the __mprop__ method."""
    assert magic_prop.__mprop__() is None


def test_mpropkw(magic_prop):
    """Test the __mpropkw__ method."""
    assert magic_prop.__mpropkw__() is None


if False:
    def test_setmpropnames(magic_prop):
        """Test the __setmpropnames__ method."""
        magic_prop.__setmpropnames__(mprop="prop", mpropkw="prop_kw")
        assert magic_prop.__mprop__() == "prop"
        assert magic_prop.__mpropkw__() == "prop_kw"


def test_setmpropmethods(magic_prop):
    """Test the __setmpropmethods__ method."""
    magic_prop.__setmpropmethods__(noval=["__str__"], val=["__repr__"])
    assert hasattr(magic_prop, "__str__")
    assert hasattr(magic_prop, "__repr__")


if False:
    def test_setmprop(magic_prop):
        """Test the __setmprop__ method."""
        magic_prop.__setmprop__(mprop="prop", mpropkw="prop_kw")
        assert magic_prop.__mprop__() == "prop"
        assert magic_prop.__mpropkw__() == "prop_kw"


def test_getmprop(magic_prop):
    """Test the __getmprop__ method."""
    assert magic_prop.__getmprop__() == magic_prop
    # assert magic_class.__get_mprop__() == 0


def test_copymprop(magic_prop):
    """Test the __copymprop__ method."""
    copied_prop = magic_prop.__copymprop__()
    assert copied_prop == magic_prop


# =====================================
# FUNCTION TESTS
# =====================================


if False:
    def test_mprop_novalue():
        """Test the _mprop_novalue function."""
        method = _mprop_novalue("method")
        class TestClass:
            def method(self):
                return "method called"
            def __getmprop__(self):
                return self
        test_instance = TestClass()
        setattr(test_instance, "method", method)
        assert test_instance.method() == "method called"

    def test_mprop_value():
        """Test the _mprop_value function."""
        method = _mprop_value("method")
        class TestClass:
            def method(self, value):
                return f"method called with {value}"
            def __getmprop__(self, val=None):
                return val if val is not None else self
        test_instance = TestClass()
        setattr(test_instance, "method", method)
        assert test_instance.method("value") == "method called with value"


def test_setmprop_function():
    """Test the _setmprop function."""
    class TestClass:
        pass
    _setmprop(TestClass, mprop="test_prop", mpropkw="test_prop_kw")
    assert TestClass().__mprop__() == "test_prop"
    assert TestClass().__mpropkw__() == "test_prop_kw"


def test_setmpropnames_function():
    """Test the _setmpropnames function."""
    class TestClass:
        pass
    _setmpropnames(TestClass, mprop="test_prop", mpropkw="test_prop_kw")
    assert TestClass().__mprop__() == "test_prop"
    assert TestClass().__mpropkw__() == "test_prop_kw"


def test_setmpropmethods_function():
    """Test the _setmpropmethods function."""
    class TestClass:
        pass
    _setmpropmethods(TestClass, noval=["__str__"], val=["__repr__"])
    assert hasattr(TestClass, "__str__")
    assert hasattr(TestClass, "__repr__")


# =====================================
# MAIN
# =====================================


# Run the tests
if __name__ == "__main__":
    pytest.main()
