#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# 
# This example program shows how to use the dlib sequence segmentation tools from within a
# python program.  In particular, we will create a simple training dataset, learn a
# sequence segmentation model, and then test it on some sequences.  
#
# COMPILING THE DLIB PYTHON INTERFACE
#   You need to compile the dlib python interface before you can use this file.  To do
#   this, run compile_dlib_python_module.bat.  This should work on any operating system so
#   long as you have CMake and boost-python installed.  On Ubuntu, this can be done easily
#   by running the command: sudo apt-get install libboost-python-dev cmake


import dlib


# In a sequence segmentation task we are given a sequence of objects (e.g. words in a
# sentence) and we are supposed to detect certain subsequences (e.g. named entities).  In
# the code below we create some very simple sequence/segmentation training pairs.  In
# particular, each element of a sequence is represented by a vector which describes
# important properties of the element.  The idea is to use vectors that contain information
# useful for detecting whatever kind of subsequences you are interested in detecting.  

# To keep this example simple we will use very simple vectors.  Specifically, each vector
# is 2D and is either the vector [0 1] or [1 0].  Moreover, we will say that the
# subsequences we want to detect are any runs of the [0 1] vector.  Note that the code
# works with both dense and sparse vectors.  The following if statement constructs either
# kind depending on the value in use_sparse_vects.
use_sparse_vects = False 
if use_sparse_vects:
    training_sequences = dlib.sparse_vectorss()
    inside = dlib.sparse_vector()
    outside = dlib.sparse_vector()
    # Add index/value pairs to each sparse vector.  Any index not mentioned in a sparse
    # vector is implicitly associated with a value of zero.
    inside.append(dlib.pair(0,1))
    outside.append(dlib.pair(1,1))
else:
    training_sequences = dlib.vectorss()
    inside = dlib.vector([0, 1])
    outside = dlib.vector([1, 0])

# Here we make our training sequences and their annotated subsegments.  We create two
# training sequences. 
segments = dlib.rangess()
training_sequences.resize(2)
segments.resize(2)

# training_sequences[0] starts out empty and we append vectors onto it.  Note that we wish
# to detect the subsequence of "inside" vectors within the sequence.  So the output should
# be the range (2,5).  Note that this is a "half open" range meaning that it starts with
# the element with index 2 and ends just before the element with index 5.
training_sequences[0].append(outside) # index 0
training_sequences[0].append(outside) # index 1
training_sequences[0].append(inside)  # index 2
training_sequences[0].append(inside)  # index 3
training_sequences[0].append(inside)  # index 4
training_sequences[0].append(outside) # index 5
training_sequences[0].append(outside) # index 6
training_sequences[0].append(outside) # index 7
segments[0].append(dlib.range(2,5))

# Add another training sequence.  This one is a little longer and has two "inside" segments
# which should be detected.
training_sequences[1].append(outside) # index 0
training_sequences[1].append(outside) # index 1
training_sequences[1].append(inside)  # index 2
training_sequences[1].append(inside)  # index 3
training_sequences[1].append(inside)  # index 4
training_sequences[1].append(inside)  # index 5
training_sequences[1].append(outside) # index 6
training_sequences[1].append(outside) # index 7
training_sequences[1].append(outside) # index 8
training_sequences[1].append(inside)  # index 9
training_sequences[1].append(inside)  # index 10 
segments[1].append(dlib.range(2,6))
segments[1].append(dlib.range(9,11))


# Now that we have a simple training set we can train a sequence segmenter.  However, the
# sequence segmentation trainer has some optional parameters we can set.  These parameters
# determine properties of the segmentation model we will learn.  See the dlib documentation
# for the sequence_segmenter object for a full discussion of their meanings.
params = dlib.segmenter_params()
params.window_size = 1
params.use_high_order_features = False
params.use_BIO_model = True
params.C = 1 

# Train a model
model = dlib.train_sequence_segmenter(training_sequences, segments, params)

# A segmenter model takes a sequence of vectors and returns an array of detected ranges.
# So for example, we can give it the first training sequence and it will predict the
# locations of the subsequences.  This statement will correctly print 2,5.
print model.segment_sequence(training_sequences[0])[0]

# We can also measure the accuracy of a model relative to some labeled data.  This
# statement prints the precision, recall, and F1-score of the model relative to the data in
# training_sequences/segments.
print "Test on training data:", dlib.test_sequence_segmenter(model, training_sequences, segments)

# We can also do n-fold cross-validation and print the resulting precision, recall, and
# F1-score.
num_folds = 2
print "cross validation:", dlib.cross_validate_sequence_segmenter(training_sequences, segments, num_folds, params)


