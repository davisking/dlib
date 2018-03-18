from __future__ import print_function
import dlib
import inspect

def print_element(name, fc, ff):
    isclass = inspect.isclass(eval(name))
    ismodule = inspect.ismodule(eval(name))
    if (isclass):
        print("* :class:`{0}`".format(name), file=fc)
    elif (not ismodule):
        print("* :func:`{0}`".format(name), file=ff)

def make_listing_files():

    fc = open('classes.txt', 'w')
    ff = open('functions.txt', 'w')

    for obj in dir(dlib):
        if obj[0] == '_':
            continue
        print_element('dlib.'+obj, fc, ff)

    for obj in dir(dlib.cuda):
        if obj[0] == '_':
            continue
        print_element('dlib.cuda.'+obj, fc, ff)

    for obj in dir(dlib.image_dataset_metadata):
        if obj[0] == '_':
            continue
        print_element('dlib.image_dataset_metadata.'+obj, fc, ff)

