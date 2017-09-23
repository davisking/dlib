#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool for clustering using chinese_whispers.
#   This is useful when you have a collection of photographs which you know are linked to
#   a particular person, but the person may be photographed with multiple other people.
#   In this example, we assume the largest cluster will contain photos of the common person in the
#   collection of photographs. Then, we save extracted images of the face in the largest cluster in
#   a 150x150 px format which is suitable for jittering and loading to perform metric learning (as shown
#   in the dnn_metric_learning_on_images_ex.cpp example.
#   https://github.com/davisking/dlib/blob/master/examples/dnn_metric_learning_on_images_ex.cpp
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  This code will also use CUDA if you have CUDA and cuDNN
#   installed.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io

if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "   ./face_clustering.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces output_folder\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]
output_folder_path = sys.argv[4]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

descriptors = []
images = []

# Now find all the faces and compute 128D face descriptors for each face.
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))

# Now let's cluster the faces.  
labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
num_classes = len(set(labels))
print("Number of clusters: {}".format(num_classes))

# Find biggest class
biggest_class = None
biggest_class_length = 0
for i in range(0, num_classes):
    class_length = len([label for label in labels if label == i])
    if class_length > biggest_class_length:
        biggest_class_length = class_length
        biggest_class = i

print("Biggest cluster id number: {}".format(biggest_class))
print("Number of faces in biggest cluster: {}".format(biggest_class_length))

# Find the indices for the biggest class
indices = []
for i, label in enumerate(labels):
    if label == biggest_class:
        indices.append(i)

print("Indices of images in the biggest cluster: {}".format(str(indices)))

# Ensure output directory exists
if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

# Save the extracted faces
print("Saving faces in largest cluster to output folder...")
for i, index in enumerate(indices):
    img, shape = images[index]
    file_path = os.path.join(output_folder_path, "face_" + str(i))
    # The size and padding arguments are optional with default size=150x150 and padding=0.25
    dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
    
    


