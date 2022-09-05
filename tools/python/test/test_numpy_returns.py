import sys
import pickle

import dlib
import pytest

import utils

# Paths are relative to dlib root
image_path = "examples/faces/Tom_Cruise_avp_2014_4.jpg"
shape_path = "tools/python/test/shape.pkl"


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


@pytest.mark.skipif(not utils.is_numpy_installed(), reason="requires numpy")
def test_partition_pixels():
    truth = (102, 159, 181)
    img, shape = get_test_image_and_shape()

    assert dlib.partition_pixels(img) == truth[0]
    assert dlib.partition_pixels(img, 3) == truth

    # Call all these versions of this mainly to make sure binding to
    # various image types works.
    assert dlib.partition_pixels(img[:, :, 0].astype("uint8")) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype("float32")) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype("float64")) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype("uint16")) == 125
    assert dlib.partition_pixels(img[:, :, 0].astype("uint32")) == 125


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
