from dlib import array
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle

try:
    from types import FloatType
except ImportError:
    FloatType = float

from pytest import raises


def test_array_init_with_number():
    a = array(5)
    assert len(a) == 5
    for i in range(5):
        assert a[i] == 0
        assert type(a[i]) == FloatType


def test_array_init_with_negative_number():
    with raises(Exception):
        array(-5)


def test_array_init_with_zero():
    a = array(0)
    assert len(a) == 0


def test_array_init_with_list():
    a = array([0, 1, 2, 3, 4])
    assert len(a) == 5
    for idx, val in enumerate(a):
        assert idx == val
        assert type(val) == FloatType


def test_array_init_with_empty_list():
    a = array([])
    assert len(a) == 0


def test_array_init_without_argument():
    a = array()
    assert len(a) == 0


def test_array_init_with_tuple():
    a = array((0, 1, 2, 3, 4))
    for idx, val in enumerate(a):
        assert idx == val
        assert type(val) == FloatType


def test_array_serialization_empty():
    a = array()
    # cPickle with protocol 2 required for Python 2.7
    # see http://pybind11.readthedocs.io/en/stable/advanced/classes.html#custom-constructors
    ser = pickle.dumps(a, 2)
    deser = pickle.loads(ser)
    assert a == deser


def test_array_serialization():
    a = array([0, 1, 2, 3, 4])
    ser = pickle.dumps(a, 2)
    deser = pickle.loads(ser)
    assert a == deser


def test_array_extend():
    a = array()
    a.extend([0, 1, 2, 3, 4])
    assert len(a) == 5
    for idx, val in enumerate(a):
        assert idx == val
        assert type(val) == FloatType


def test_array_string_representations_empty():
    a = array()
    assert str(a) == ""
    assert repr(a) == "array[]"


def test_array_string_representations():
    a = array([1, 2, 3])
    assert str(a) == "1\n2\n3"
    assert repr(a) == "array[1, 2, 3]"


def test_array_clear():
    a = array(10)
    a.clear()
    assert len(a) == 0


def test_array_resize():
    a = array(10)
    a.resize(100)
    assert len(a) == 100

    for i in range(100):
        assert a[i] == 0
