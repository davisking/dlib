import sys
import pickle

import dlib
import pytest

import utils

# Paths are relative to dlib root
image_path = "examples/faces/Tom_Cruise_avp_2014_4.jpg"
shape_path = "tools/python/test/shape.pkl"
face_chip_path = "tools/python/test/test_face_chip.npy"

def get_test_image_and_shape():
    img = dlib.load_rgb_image(image_path)
    shape = utils.load_pickled_compatible(shape_path)
    return img, shape
 
def get_test_face_chips():
    rgb_img, shape = get_test_image_and_shape()
    shapes = dlib.full_object_detections()
    shapes.append(shape)
    return dlib.get_face_chips(rgb_img, shapes)

def get_test_face_chip():
    rgb_img, shape = get_test_image_and_shape()
    return dlib.get_face_chip(rgb_img, shape)

# The tests below will be skipped if Numpy is not installed
@pytest.mark.skipif(not utils.is_numpy_installed(), reason="requires numpy")
def test_get_face_chip():
    import numpy
    face_chip = get_test_face_chip()
    expected = numpy.load(face_chip_path)
    assert numpy.array_equal(face_chip, expected)

@pytest.mark.skipif(not utils.is_numpy_installed(), reason="requires numpy")
def test_get_face_chips():
    import numpy
    face_chips = get_test_face_chips()
    expected = numpy.load(face_chip_path)
    assert numpy.array_equal(face_chips[0], expected)

@pytest.mark.skipif(not utils.is_numpy_installed(), reason="requires numpy")
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

@pytest.mark.skipif(not utils.is_numpy_installed(), reason="requires numpy")
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