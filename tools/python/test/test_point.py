from dlib import point, points
try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle


def test_point():
    p = point(27, 42)
    assert repr(p) == "point(27, 42)"
    assert str(p) == "(27, 42)"
    assert p.x == 27
    assert p.y == 42
    ser = pickle.dumps(p, 2)
    deser = pickle.loads(ser)
    assert deser.x == p.x
    assert deser.y == p.y

def test_point_assignment():
    p = point(27, 42)
    p.x = 16
    assert p.x == 16
    assert p.y == 42
    p.y = 31
    assert p.x == 16
    assert p.y == 31

def test_point_init_kwargs():
    p = point(y=27, x=42)
    assert repr(p) == "point(42, 27)"
    assert str(p) == "(42, 27)"
    assert p.x == 42
    assert p.y == 27


def test_points():
    ps = points()

    ps.resize(5)
    assert len(ps) == 5
    for i in range(5):
        assert ps[i].x == 0
        assert ps[i].y == 0

    ps.clear()
    assert len(ps) == 0

    ps.extend([point(1, 2), point(3, 4)])
    assert len(ps) == 2

    ser = pickle.dumps(ps, 2)
    deser = pickle.loads(ser)
    assert deser[0].x == 1
    assert deser[0].y == 2
    assert deser[1].x == 3
    assert deser[1].y == 4
