from random import Random

from dlib import chinese_whispers
from pytest import raises


def test_chinese_whispers():
    assert len(chinese_whispers([])) == 0
    assert len(chinese_whispers([(0, 0), (1, 1)])) == 2

    # Test that values from edges are actually used and that correct values are returned
    labels = chinese_whispers([(0, 0), (0, 1), (1, 1)])
    assert len(labels) == 2
    assert labels[0] == labels[1]
    labels = chinese_whispers([(0, 0), (1, 1)])
    assert len(labels) == 2
    assert labels[0] != labels[1]


def test_chinese_whispers_with_distance():
    assert len(chinese_whispers([(0, 0, 1)])) == 1
    assert len(chinese_whispers([(0, 0, 1), (0, 1, 0.5), (1, 1, 1)])) == 2

    # Test that values from edges and distances are actually used and that correct values are returned
    labels = chinese_whispers([(0, 0, 1), (0, 1, 1), (1, 1, 1)])
    assert len(labels) == 2
    assert labels[0] == labels[1]
    labels = chinese_whispers([(0, 0, 1), (0, 1, 0.0), (1, 1, 1)])
    assert len(labels) == 2
    assert labels[0] != labels[1]

    # Non-trivial test
    edges = []
    r = Random(0)
    for i in range(100):
        edges.append((i, i, 1))
        edges.append((i, r.randint(0, 99), r.random()))
    assert len(chinese_whispers(edges)) == 100


def test_chinese_whispers_type_checks():
    """
    Tests contract (expected errors) in case client provides wrong types
    """
    with raises(TypeError):
        chinese_whispers()
    with raises(TypeError):
        chinese_whispers('foo')
    with raises(RuntimeError):
        chinese_whispers(['foo'])
    with raises(IndexError):
        chinese_whispers([(0,)])
    with raises(IndexError):
        chinese_whispers([(0, 1, 2, 3)])
    with raises(RuntimeError):
        chinese_whispers([('foo', 'bar')])
