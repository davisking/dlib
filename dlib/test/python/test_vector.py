from dlib import vector, vectors, vectorss, dot
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle
from pytest import raises


def test_vector_empty_init():
    v = vector()
    assert len(v) == 0
    assert v.shape == (0, 1)
    assert str(v) == ""
    assert repr(v) == "dlib.vector([])"


def test_vector_init_with_number():
    v = vector(3)
    assert len(v) == 3
    assert v.shape == (3, 1)
    assert str(v) == "0\n0\n0"
    assert repr(v) == "dlib.vector([0, 0, 0])"


def test_vector_set_size():
    v = vector(3)

    v.set_size(0)
    assert len(v) == 0
    assert v.shape == (0, 1)

    v.resize(10)
    assert len(v) == 10
    assert v.shape == (10, 1)
    for i in range(10):
        assert v[i] == 0


def test_vector_init_with_list():
    v = vector([1, 2, 3])
    assert len(v) == 3
    assert v.shape == (3, 1)
    assert str(v) == "1\n2\n3"
    assert repr(v) == "dlib.vector([1, 2, 3])"


def test_vector_getitem():
    v = vector([1, 2, 3])
    assert v[0] == 1
    assert v[-1] == 3
    assert v[1] == v[-2]


def test_vector_slice():
    v = vector([1, 2, 3, 4, 5])
    v_slice = v[1:4]
    assert len(v_slice) == 3
    for idx, val in enumerate([2, 3, 4]):
        assert v_slice[idx] == val

    v_slice = v[-3:-1]
    assert len(v_slice) == 2
    for idx, val in enumerate([3, 4]):
        assert v_slice[idx] == val

    v_slice = v[1:-2]
    assert len(v_slice) == 2
    for idx, val in enumerate([2, 3]):
        assert v_slice[idx] == val


def test_vector_invalid_getitem():
    v = vector([1, 2, 3])
    with raises(IndexError):
        v[-4]
    with raises(IndexError):
        v[3]


def test_vector_init_with_negative_number():
    with raises(Exception):
        vector(-3)


def test_dot():
    v1 = vector([1, 0])
    v2 = vector([0, 1])
    v3 = vector([-1, 0])
    assert dot(v1, v1) == 1
    assert dot(v1, v2) == 0
    assert dot(v1, v3) == -1


def test_vector_serialization():
    v = vector([1, 2, 3])
    ser = pickle.dumps(v, 2)
    deser = pickle.loads(ser)
    assert str(v) == str(deser)


def generate_test_vectors():
    vs = vectors()
    vs.append(vector([0, 1, 2]))
    vs.append(vector([3, 4, 5]))
    vs.append(vector([6, 7, 8]))
    assert len(vs) == 3
    return vs


def generate_test_vectorss():
    vss = vectorss()
    vss.append(generate_test_vectors())
    vss.append(generate_test_vectors())
    vss.append(generate_test_vectors())
    assert len(vss) == 3
    return vss


def test_vectors_serialization():
    vs = generate_test_vectors()
    ser = pickle.dumps(vs, 2)
    deser = pickle.loads(ser)
    assert vs == deser


def test_vectors_clear():
    vs = generate_test_vectors()
    vs.clear()
    assert len(vs) == 0


def test_vectors_resize():
    vs = vectors()
    vs.resize(100)
    assert len(vs) == 100
    for i in range(100):
        assert len(vs[i]) == 0


def test_vectors_extend():
    vs = vectors()
    vs.extend([vector([1, 2, 3]), vector([4, 5, 6])])
    assert len(vs) == 2


def test_vectorss_serialization():
    vss = generate_test_vectorss()
    ser = pickle.dumps(vss, 2)
    deser = pickle.loads(ser)
    assert vss == deser


def test_vectorss_clear():
    vss = generate_test_vectorss()
    vss.clear()
    assert len(vss) == 0


def test_vectorss_resize():
    vss = vectorss()
    vss.resize(100)
    assert len(vss) == 100
    for i in range(100):
        assert len(vss[i]) == 0


def test_vectorss_extend():
    vss = vectorss()
    vss.extend([generate_test_vectors(), generate_test_vectors()])
    assert len(vss) == 2
