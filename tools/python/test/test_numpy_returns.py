import dlib
import sys
import pickle
import numpy
import pytest

image_path = "examples/faces/Tom_Cruise_avp_2014_4.jpg"
shape_path = "tools/python/test/shape.pkl"
face_chip_path = "tools/python/test/test_face_chip.pkl"

def get_test_image_and_shape():
    img = dlib.load_rgb_image(image_path)
    shape = None
    with open(shape_path, "rb") as shape_file:
        shape = pickle.load(shape_file) 
    return img, shape
 
def get_test_face_chips():
    rgb_img, shape = get_test_image_and_shape()
    shapes = dlib.full_object_detections()
    shapes.append(shape)
    return dlib.get_face_chips(rgb_img, shapes)

def get_test_face_chip():
    rgb_img, shape = get_test_image_and_shape()
    return dlib.get_face_chip(rgb_img, shape)
 
def test_get_face_chip():
    face_chip = get_test_face_chip()
    expected = numpy.load(face_chip_path)
    assert numpy.array_equal(face_chip, expected)

def test_get_face_chips():
    face_chips = get_test_face_chips()
    expected = numpy.load(face_chip_path)
    assert numpy.array_equal(face_chips[0], expected)

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