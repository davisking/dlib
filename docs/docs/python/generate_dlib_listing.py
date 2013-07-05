from __future__ import print_function
import dlib
import inspect

def make_listing_files():

    fc = open('classes.txt', 'w')
    ff = open('functions.txt', 'w')

    for obj in dir(dlib):
        if obj[0] == '_':
            continue
        name = 'dlib.'+obj
        isclass = inspect.isclass(eval(name))
        if (isclass):
            print("* :class:`{}`".format(name), file=fc)
        else:
            print("* :func:`{}`".format(name), file=ff)

