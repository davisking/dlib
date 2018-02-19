from dlib import find_max_global, find_min_global
from pytest import raises


def test_global_optimization_nargs():
    w0 = find_max_global(lambda *args: sum(args), [0, 0, 0], [1, 1, 1], 10)
    w1 = find_min_global(lambda *args: sum(args), [0, 0, 0], [1, 1, 1], 10)
    assert w0 == ([1, 1, 1], 3)
    assert w1 == ([0, 0, 0], 0)

    w2 = find_max_global(lambda a, b, c, *args: a + b + c - sum(args), [0, 0, 0], [1, 1, 1], 10)
    w3 = find_min_global(lambda a, b, c, *args: a + b + c - sum(args), [0, 0, 0], [1, 1, 1], 10)
    assert w2 == ([1, 1, 1], 3)
    assert w3 == ([0, 0, 0], 0)

    with raises(Exception):
        find_max_global(lambda a, b: 0, [0, 0, 0], [1, 1, 1], 10)
    with raises(Exception):
        find_min_global(lambda a, b: 0, [0, 0, 0], [1, 1, 1], 10)
    with raises(Exception):
        find_max_global(lambda a, b, c, d, *args: 0, [0, 0, 0], [1, 1, 1], 10)
    with raises(Exception):
        find_min_global(lambda a, b, c, d, *args: 0, [0, 0, 0], [1, 1, 1], 10)
