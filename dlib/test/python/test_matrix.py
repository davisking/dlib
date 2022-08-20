from dlib import matrix
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle
from pytest import raises

try:
    import numpy
    have_numpy = True
except ImportError:
    have_numpy = False 


def test_matrix_empty_init():
    m = matrix()
    assert m.nr() == 0
    assert m.nc() == 0
    assert m.shape == (0, 0)
    assert len(m) == 0
    assert repr(m) == "< dlib.matrix containing: >"
    assert str(m) == ""


def test_matrix_from_list():
    m = matrix([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
    assert m.nr() == 3
    assert m.nc() == 3
    assert m.shape == (3, 3)
    assert len(m) == 3
    assert repr(m) == "< dlib.matrix containing: \n0 1 2 \n3 4 5 \n6 7 8 >"
    assert str(m) == "0 1 2 \n3 4 5 \n6 7 8"

    deser = pickle.loads(pickle.dumps(m, 2))

    for row in range(3):
        for col in range(3):
            assert m[row][col] == deser[row][col]


def test_matrix_from_list_with_invalid_rows():
    with raises(ValueError):
        matrix([[0, 1, 2],
                [3, 4],
                [5, 6, 7]])


def test_matrix_from_list_as_column_vector():
    m = matrix([0, 1, 2])
    assert m.nr() == 3
    assert m.nc() == 1
    assert m.shape == (3, 1)
    assert len(m) == 3
    assert repr(m) == "< dlib.matrix containing: \n0 \n1 \n2 >"
    assert str(m) == "0 \n1 \n2"


if have_numpy:
    def test_matrix_from_object_with_2d_shape():
        m1 = numpy.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
        m = matrix(m1)
        assert m.nr() == 3
        assert m.nc() == 3
        assert m.shape == (3, 3)
        assert len(m) == 3
        assert repr(m) == "< dlib.matrix containing: \n0 1 2 \n3 4 5 \n6 7 8 >"
        assert str(m) == "0 1 2 \n3 4 5 \n6 7 8"


    def test_matrix_from_object_without_2d_shape():
        with raises(IndexError):
            m1 = numpy.array([0, 1, 2])
            matrix(m1)


def test_matrix_from_object_without_shape():
    with raises(AttributeError):
        matrix("invalid")


def test_matrix_set_size():
    m = matrix()
    m.set_size(5, 5)

    assert m.nr() == 5
    assert m.nc() == 5
    assert m.shape == (5, 5)
    assert len(m) == 5
    assert repr(m) == "< dlib.matrix containing: \n0 0 0 0 0 \n0 0 0 0 0 \n0 0 0 0 0 \n0 0 0 0 0 \n0 0 0 0 0 >"
    assert str(m) == "0 0 0 0 0 \n0 0 0 0 0 \n0 0 0 0 0 \n0 0 0 0 0 \n0 0 0 0 0"

    deser = pickle.loads(pickle.dumps(m, 2))

    for row in range(5):
        for col in range(5):
            assert m[row][col] == deser[row][col]
