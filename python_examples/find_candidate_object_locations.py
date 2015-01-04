#!/usr/bin/python
# This example shows how to use find_candidate_object_locations()

import dlib
from skimage import io

image_file = '../examples/faces/2009_004587.jpg'
img = io.imread(image_file)

# Locations of candidate objects will be saved into rects
rects = []
dlib.find_candidate_object_locations(img, rects, min_size=500)

windows = []
for d in rects:
    windows.append([d.top(), d.left(), d.bottom(), d.right()])

print len(windows)
print (image_file, windows)
