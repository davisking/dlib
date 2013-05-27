#!/usr/bin/python
#
# You need to compile the dlib python interface before you can use this
# file.  To do this, run compile_dlib_python_module.bat.  You also need to
# have the boost-python library installed.  On Ubuntu, this can be done easily by running
# the command: sudo apt-get install libboost-python-dev

#  asfd
import dlib

use_sparse_vects = False 

if use_sparse_vects:
    samples = dlib.sparse_vectorss()
else:
    samples = dlib.vectorss()

segments = dlib.rangess()

if use_sparse_vects:
    inside = dlib.sparse_vector()
    outside = dlib.sparse_vector()
    inside.append(dlib.pair(0,1))
    outside.append(dlib.pair(1,1))
else:
    inside = dlib.vector([0, 1])
    outside = dlib.vector([1, 0])

samples.resize(2)
segments.resize(2)

samples[0].append(outside)
samples[0].append(outside)
samples[0].append(inside)
samples[0].append(inside)
samples[0].append(inside)
samples[0].append(outside)
samples[0].append(outside)
samples[0].append(outside)
segments[0]
segments[0].append(dlib.range(2,5))


samples[1].append(outside)
samples[1].append(outside)
samples[1].append(inside)
samples[1].append(inside)
samples[1].append(inside)
samples[1].append(inside)
samples[1].append(outside)
samples[1].append(outside)
segments[1].append(dlib.range(2,6))


params = dlib.segmenter_params()
#params.be_verbose = True
params.window_size = 1
params.use_high_order_features = False
params.C = 1
print "params:", params

df = dlib.train_sequence_segmenter(samples, segments, params)

print len(df.segment_sequence(samples[0]))
print df.segment_sequence(samples[0])[0]



print df.weights

#res = dlib.test_sequence_segmenter(df, samples, segments)
res = dlib.cross_validate_sequence_segmenter(samples, segments, 2, params)

print res

