from __future__ import print_function
import _dlib_pybind11
import inspect

def print_element(name, fc, ff, fconstants):
    isclass = inspect.isclass(eval(name))
    ismodule = inspect.ismodule(eval(name))
    isroutine = inspect.isroutine(eval(name))
    if (isclass):
        print("* :class:`{0}`".format(name), file=fc)
    elif (isroutine):
        print("* :func:`{0}`".format(name), file=ff)
    elif (not ismodule):
        print("* :const:`{0}`".format(name), file=fconstants)

def make_listing_files():

    fc = open('classes.txt', 'w')
    ff = open('functions.txt', 'w')
    fconstants = open('constants.txt', 'w')

    for obj in dir(_dlib_pybind11):
        if obj[0] == '_':
            continue
        print_element('_dlib_pybind11.'+obj, fc, ff, fconstants)

    for obj in dir(_dlib_pybind11.cuda):
        if obj[0] == '_':
            continue
        print_element('_dlib_pybind11.cuda.'+obj, fc, ff, fconstants)

    for obj in dir(_dlib_pybind11.image_dataset_metadata):
        if obj[0] == '_':
            continue
        print_element('_dlib_pybind11.image_dataset_metadata.'+obj, fc, ff, fconstants)

