import dlib
import sys
import cv2
import numpy
import pytest

def get_test_image_and_shape():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../../../python_examples/shape_predictor_5_face_landmarks.dat")

    img = cv2.imread("../../../python_examples/face.jpg")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_img)
    shape = predictor(rgb_img, dets[0])
    return rgb_img, shape
 
def get_test_face_chips():
    rgb_img, shape = get_test_image_and_shape()
    shapes = dlib.full_object_detections()
    shapes.append(shape)
    return dlib.get_face_chips(rgb_img, shapes)

def get_test_face_chip():
    rgb_img, shape = get_test_image_and_shape()
    return dlib.get_face_chip(rgb_img, shape)
 
@pytest.mark.skip
def test_get_face_chip():
    face_chip = get_test_face_chip()
    expected = numpy.load("test_face_chip.npy")
    assert numpy.array_equal(face_chip, expected)

@pytest.mark.skip
def test_get_face_chips():
    face_chips = get_test_face_chips()
    expected = numpy.load("test_face_chip.npy")
    assert numpy.array_equal(face_chips[0], expected)

@pytest.mark.skip
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
    
@pytest.mark.skip
def test_regression_issue_1220_get_face_chips():
    """
    Memory leak in Python get_face_chip
    https://github.com/davisking/dlib/issues/1220
    """
    face_chips = get_test_face_chips()
    assert sys.getrefcount(face_chips) == 2
    # we expect three references because the list wraps
    assert sys.getrefcount(face_chips[0]) == 3
