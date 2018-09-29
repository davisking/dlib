#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how faces were jittered and augmented to create training
#   data for dlib's face recognition model.  It takes an input image and
#   disturbs the colors as well as applies random translations, rotations, and
#   scaling.

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
#
#   The image file used in this example is in the public domain:
#   https://commons.wikimedia.org/wiki/File:Tom_Cruise_avp_2014_4.jpg
import sys

import dlib

def show_jittered_images(window, jittered_images):
    '''
        Shows the specified jittered images one by one
    '''
    for img in jittered_images:
        window.set_image(img)
        dlib.hit_enter_to_continue()

if len(sys.argv) != 2:
    print(
        "Call this program like this:\n"
        "   ./face_jitter.py shape_predictor_5_face_landmarks.dat\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")
    exit()

predictor_path = sys.argv[1]
face_file_path = "../examples/faces/Tom_Cruise_avp_2014_4.jpg"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Load the image using dlib
img = dlib.load_rgb_image(face_file_path)

# Ask the detector to find the bounding boxes of each face.
dets = detector(img)

num_faces = len(dets)

# Find the 5 face landmarks we need to do the alignment.
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))

# Get the aligned face image and show it
image = dlib.get_face_chip(img, faces[0], size=320)
window = dlib.image_window()
window.set_image(image)
dlib.hit_enter_to_continue()

# Show 5 jittered images without data augmentation
jittered_images = dlib.jitter_image(image, num_jitters=5)
show_jittered_images(window, jittered_images)

# Show 5 jittered images with data augmentation
jittered_images = dlib.jitter_image(image, num_jitters=5, disturb_colors=True)
show_jittered_images(window, jittered_images)
