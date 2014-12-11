#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to use dlib's implementation of the paper:
#   One Millisecond Face Alignment with an Ensemble of Regression Trees by
#   Vahid Kazemi and Josephine Sullivan, CVPR 2014
#
#   In particular, we will train a face landmarking model based on a small
#   dataset and then evaluate it.  If you want to visualize the output of the
#   trained model on some images then you can run the
#   face_landmark_detection.py example program with sp.dat as the input
#   model.
#
#   It should also be noted that this kind of model, while often used for face
#   landmarking, is quite general and can be used for a variety of shape
#   prediction tasks.  But here we demonstrate it only on a simple face
#   landmarking task.
#
# COMPILING THE DLIB PYTHON INTERFACE
#   Dlib comes with a compiled python interface for python 2.7 on MS Windows. If
#   you are using another python version or operating system then you need to
#   compile the dlib python interface before you can use this file.  To do this,
#   run compile_dlib_python_module.bat.  This should work on any operating
#   system so long as you have CMake and boost-python installed.
#   On Ubuntu, this can be done easily by running the command:
#       sudo apt-get install libboost-python-dev cmake
import os
import sys
import glob

import dlib
from skimage import io


# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
# where it is.
if len(sys.argv) != 2:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_shape_predictor.py ../examples/faces")
    exit()
faces_folder = sys.argv[1]

options = dlib.shape_predictor_training_options()
# Now make the object responsible for training the model.
# This algorithm has a bunch of parameters you can mess with.  The
# documentation for the shape_predictor_trainer explains all of them.
# You should also read Kazemi paper which explains all the parameters
# in great detail.  However, here I'm just setting three of them
# differently than their default values.  I'm doing this because we
# have a very small dataset.  In particular, setting the oversampling
# to a high amount (300) effectively boosts the training set size, so
# that helps this example.
options.oversampling_amount = 300
# I'm also reducing the capacity of the model by explicitly increasing
# the regularization (making nu smaller) and by using trees with
# smaller depths.
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

# This function does the actual training.  It will save the final predictor to
# predictor.dat.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the face parts.
training_xml_path = os.path.join(faces_folder, "training_with_face_landmarks.xml")
testing_xml_path = os.path.join(faces_folder, "testing_with_face_landmarks.xml")

dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)

# Now that we have a facial landmark predictor we can test it.  The first
# statement tests it on the training data.  It will print the mean average error
print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_shape_predictor(training_xml_path, "predictor.dat")))
# However, to get an idea if it really worked without overfitting we need to
# run it on images it wasn't trained on.  The next line does this.  Happily, we
# see that the object detector works perfectly on the testing images.
print("Testing accuracy: {}".format(
    dlib.test_shape_predictor(testing_xml_path, "predictor.dat")))

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk. We also need to load a face detector to provide the
# initial estimate of the facial location
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictor.dat")

# Now let's run the detector and predictor over the images in the faces folder
# and display the results.
print("Showing detections and predictions on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    win.clear_overlay()
    win.set_image(img)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shapes = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shapes.part(0),
                                                  shapes.part(1)))
        # Add all facial landmarks one at a time
        win.add_overlay(shapes)

    win.add_overlay(dets)
    raw_input("Hit enter to continue")

# Finally, note that you don't have to use the XML based input to
# train_shape_predictor().  If you have already loaded your training
# images and fll_object_detections for the objects then you can call it with
# the existing objects.
