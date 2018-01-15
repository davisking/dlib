from dlib import range, ranges, rangess
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle
from pytest import raises


def test_range():
    r = range(0, 10)
    assert r.begin == 0
    assert r.end == 10
    assert str(r) == "0, 10"
    assert repr(r) == "dlib.range(0, 10)"
    assert len(r) == 10

    ser = pickle.dumps(r, 2)
    deser = pickle.loads(ser)

    for a, b in zip(r, deser):
        assert a == b


# TODO: make this init parameterization an exception?
def test_range_wrong_order():
    r = range(5, 0)
    assert r.begin == 5
    assert r.end == 0
    assert str(r) == "5, 0"
    assert repr(r) == "dlib.range(5, 0)"
    assert len(r) == 0


def test_range_with_negative_elements():
    with raises(TypeError):
        range(-1, 1)
    with raises(TypeError):
        range(1, -1)


def test_ranges():
    rs = ranges()
    assert len(rs) == 0

    rs.resize(5)
    assert len(rs) == 5
    for r in rs:
        assert r.begin == 0
        assert r.end == 0

    rs.clear()
    assert len(rs) == 0

    rs.extend([range(1, 2), range(3, 4)])
    assert rs[0].begin == 1
    assert rs[0].end == 2
    assert rs[1].begin == 3
    assert rs[1].end == 4

    ser = pickle.dumps(rs, 2)
    deser = pickle.loads(ser)
    assert rs == deser


def test_rangess():
    rss = rangess()
    assert len(rss) == 0

    rss.resize(5)
    assert len(rss) == 5
    for rs in rss:
        assert len(rs) == 0

    rss.clear()
    assert len(rss) == 0

    rs1 = ranges()
    rs1.append(range(1, 2))
    rs1.append(range(3, 4))

    rs2 = ranges()
    rs2.append(range(5, 6))
    rs2.append(range(7, 8))

    rss.extend([rs1, rs2])
    assert rss[0][0].begin == 1
    assert rss[0][1].begin == 3
    assert rss[1][0].begin == 5
    assert rss[1][1].begin == 7
    assert rss[0][0].end == 2
    assert rss[0][1].end == 4
    assert rss[1][0].end == 6
    assert rss[1][1].end == 8

    ser = pickle.dumps(rss, 2)
    deser = pickle.loads(ser)
    assert rss == deser
