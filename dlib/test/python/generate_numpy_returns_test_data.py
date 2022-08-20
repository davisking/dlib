#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This utility generates the test data required for the tests contained in test_numpy_returns.py
#
#   Also note that this utility requires Numpy which can be installed
#   via the command:
#       pip install numpy
import sys
import dlib
import numpy as np
import utils

if len(sys.argv) != 2:
    print(
        "Call this program like this:\n"
        "   ./generate_numpy_returns_test_data.py shape_predictor_5_face_landmarks.dat\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")
    exit()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(sys.argv[1])

img = dlib.load_rgb_image("../../../examples/faces/Tom_Cruise_avp_2014_4.jpg")
dets = detector(img)
shape = predictor(img, dets[0])

utils.save_pickled_compatible(shape, "shape.pkl")

face_chip = dlib.get_face_chip(img, shape)
np.save("test_face_chip", face_chip)