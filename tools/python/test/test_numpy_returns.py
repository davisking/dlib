import sys
import pickle
import pkgutil

import dlib
import pytest

# Paths are relative to dlib root
image_path = "examples/faces/Tom_Cruise_avp_2014_4.jpg"
shape_path = "tools/python/test/shape.pkl"
face_chip_path = "tools/python/test/test_face_chip.npy"

def is_numpy_installed():
    if pkgutil.find_loader("numpy"):
        return True
    else:
        return False

def load_pickled_object(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)

def get_test_image_and_shape():
    img = dlib.load_rgb_image(image_path)
    shape = load_pickled_object(shape_path)
    return img, shape
 
def get_test_face_chips():
    rgb_img, shape = get_test_image_and_shape()
    shapes = dlib.full_object_detections()
    shapes.append(shape)
    return dlib.get_face_chips(rgb_img, shapes)

def get_test_face_chip():
    rgb_img, shape = get_test_image_and_shape()
    return dlib.get_face_chip(rgb_img, shape)

# The tests below will be skipped for Python 2.7 as there is a pickle issue
# with pybind objects:
# https://github.com/pybind/pybind11/issues/271
# The tests will also be skipped if numpy is not installed
@pytest.mark.skipif(sys.version_info < (3, 0) or not is_numpy_installed(), reason="requires Python 3 and numpy")
def test_get_face_chip():
    import numpy
    face_chip = get_test_face_chip()
    expected = numpy.load(face_chip_path)
    assert numpy.array_equal(face_chip, expected)

@pytest.mark.skipif(sys.version_info < (3, 0) or not is_numpy_installed(), reason="requires Python 3 and numpy")
def test_get_face_chips():
    import numpy
    face_chips = get_test_face_chips()
    expected = numpy.load(face_chip_path)
    assert numpy.array_equal(face_chips[0], expected)

@pytest.mark.skipif(sys.version_info < (3, 0) or not is_numpy_installed(), reason="requires Python 3 and numpy")
def test_regression_issue_1220_get_face_chip():
    """
    Memory leak in Python get_face_chip
    https://github.com/davisking/dlib/issues/1220
    """
    face_chip = get_test_face_chip()
    # we expect two references:
    # 1.) the local variable 
    # 2.) the temporary passed to getrefcount
    assert sys.getrefcount(face_chip) == 2

@pytest.mark.skipif(sys.version_info < (3, 0) or not is_numpy_installed(), reason="requires Python 3 and numpy")
def test_regression_issue_1220_get_face_chips():
    """
    Memory leak in Python get_face_chip
    https://github.com/davisking/dlib/issues/1220
    """
    face_chips = get_test_face_chips()
    count = sys.getrefcount(face_chips)
    assert count == 2
    count = sys.getrefcount(face_chips[0])
    assert count == 2