#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# 
# This is an example illustrating the use of the global optimization routine,
# find_min_global(), from the dlib C++ Library.  This is a tool for finding the
# inputs to a function that result in the function giving its minimal output.
# This is a very useful tool for hyper parameter search when applying machine
# learning methods.  There are also many other applications for this kind of
# general derivative free optimization.  However, in this example program, we
# simply show how to call the method.  For that, we use a common global
# optimization test function, as you can see below.
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

import dlib
from math import sin,cos,pi,exp,sqrt

# This is a standard test function for these kinds of optimization problems.
# It has a bunch of local minima, with the global minimum resulting in
# holder_table()==-19.2085025679. 
def holder_table(x0,x1):
    return -abs(sin(x0)*cos(x1)*exp(abs(1-sqrt(x0*x0+x1*x1)/pi)))

# Find the optimal inputs to holder_table().  The print statements that follow
# show that find_min_global() finds the optimal settings to high precision.
x,y = dlib.find_min_global(holder_table, 
                           [-10,-10],  # Lower bound constraints on x0 and x1 respectively
                           [10,10],    # Upper bound constraints on x0 and x1 respectively
                           80)         # The number of times find_min_global() will call holder_table()

print("optimal inputs: {}".format(x));
print("optimal output: {}".format(y));

