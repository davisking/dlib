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

    # A single edge merges the pair. Self-loops only declare nodes.
    labels = chinese_whispers([(0, 0, 1), (0, 1, 1), (1, 1, 1)])
    assert len(labels) == 2
    assert labels[0] == labels[1]
    labels = chinese_whispers([(0, 0, 1), (0, 1, 0.5), (1, 1, 1)])
    assert len(labels) == 2
    assert labels[0] == labels[1]

    # Edge weights decide between competing neighbors. Two strong pairs joined
    # by a weak bridge stay split, while a strong bridge merges them.
    weak = [(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1),
            (0, 1, 10), (2, 3, 10), (1, 2, 0.1)]
    labels = chinese_whispers(weak)
    assert len(labels) == 4
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

    strong = [(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1),
              (0, 1, 10), (2, 3, 10), (1, 2, 30)]
    labels = chinese_whispers(strong)
    assert len(labels) == 4
    assert labels[0] == labels[1] == labels[2] == labels[3]

    # Chain with self-loops merges instead of splitting off an endpoint.
    labels = chinese_whispers([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)])
    assert len(labels) == 3
    assert labels[0] == labels[1] == labels[2]

    # Isolated nodes stay separate.
    edges = [(i, i, 1) for i in range(100)]
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
