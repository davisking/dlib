#!/usr/bin/python
#
# This example shows how to use find_candidate_object_locations().  The
# function takes an input image and generates a set of candidate rectangles
# which are expected to bound any objects in the image.
# It is based on the paper:
#    Segmentation as Selective Search for Object Recognition by Koen E. A. van de Sande, et al.
#
# Typically, you would use this as part of an object detection pipeline.
# find_candidate_object_locations() nominates boxes that might contain an
# object and you then run some expensive classifier on each one and throw away
# the false alarms.  Since find_candidate_object_locations() will only generate
# a few thousand rectangles it is much faster than scanning all possible
# rectangles inside an image.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy 

import dlib

image_file = '../examples/faces/2009_004587.jpg'
img = dlib.load_rgb_image(image_file)

# Locations of candidate objects will be saved into rects
rects = []
dlib.find_candidate_object_locations(img, rects, min_size=500)

print("number of rectangles found {}".format(len(rects))) 
for k, d in enumerate(rects):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
