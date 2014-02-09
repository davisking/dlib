#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# COMPILING THE DLIB PYTHON INTERFACE
#   Dlib comes with a compiled python interface for python 2.7 on MS Windows.  If
#   you are using another python version or operating system then you need to
#   compile the dlib python interface before you can use this file.  To do this,
#   run compile_dlib_python_module.bat.  This should work on any operating system
#   so long as you have CMake and boost-python installed.  On Ubuntu, this can be
#   done easily by running the command:  sudo apt-get install libboost-python-dev cmake

import dlib, sys
from skimage import io


detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

for f in sys.argv[1:]:
    print "processing file: ", f
    img = io.imread(f)
    dets = detector(img,1)
    print "number of faces detected: ", len(dets)

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)

    raw_input("Hit enter to continue")


