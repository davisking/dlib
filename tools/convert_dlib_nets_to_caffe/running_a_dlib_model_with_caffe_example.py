#!/usr/bin/env python

# This script takes the dlib lenet model trained by the
# examples/dnn_introduction_ex.cpp example program and runs it using caffe. 

import caffe
import numpy as np

# Before you run this program, you need to run dnn_introduction_ex.cpp to get a
# dlib lenet model.  Then you need to convert that model into a "dlib to caffe
# model" python script.  You can do this using the command line program
# included with dlib: tools/convert_dlib_nets_to_caffe.  That program will
# output a lenet_dlib_to_caffe_model.py file.  You run that program like this:
#    ./dtoc lenet.xml 1 1 28 28
# and it will create the lenet_dlib_to_caffe_model.py file, which we import
# with the next line:
import lenet_dlib_to_caffe_model as dlib_model

# lenet_dlib_to_caffe_model defines a function, save_as_caffe_model() that does
# the work of converting dlib's DNN model to a caffe model and saves it to disk
# in two files.  These files are all you need to run the model with caffe.
dlib_model.save_as_caffe_model('dlib_model_def.prototxt', 'dlib_model.proto')

# Now that we created the caffe model files, we can load them into a caffe Net object.
net = caffe.Net('dlib_model_def.prototxt', 'dlib_model.proto', caffe.TEST);


# Now lets do a test, we will run one of the MNIST images through the network.

# An MNIST image of a 7, it is the very first testing image in MNIST (i.e. wrt dnn_introduction_ex.cpp, it is testing_images[0])
data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,84,185,159,151,60,36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,17,66,14,67,67,67,59,21,236,254,106, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,83,253,209,18, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,22,233,255,83, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,129,254,238,44, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,59,249,254,62, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,133,254,187,5, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,9,205,248,58, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,126,254,182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,75,251,240,57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,19,221,254,166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,3,203,254,219,35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,38,254,254,77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,31,224,254,115,1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,133,254,254,52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,61,242,254,254,52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,121,254,254,219,40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,121,254,207,18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float32');
data.shape = (dlib_model.input_batch_size, dlib_model.input_num_channels, dlib_model.input_num_rows, dlib_model.input_num_cols);

# labels isn't logically needed but there doesn't seem to be a way to use
# caffe's Net interface without providing a superfluous input array.  So we do
# that here.
labels = np.ones((dlib_model.input_batch_size), dtype='float32')
# Give the image to caffe
net.set_input_arrays(data/256, labels)
# Run the data through the network and get the results.
out = net.forward()

# Print outputs, looping over minibatch.  You should see that the network
# correctly classifies the image (it's the number 7).
for i in xrange(dlib_model.input_batch_size):
    print i, 'net final layer = ', out['fc1'][i]
    print i, 'predicted number =', np.argmax(out['fc1'][i])



