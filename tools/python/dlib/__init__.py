# Copyright (C) 2020  Davis E. King (davis@dlib.net)
# License: Boost Software License   See LICENSE.txt for the full license.

try:
    import os
    # On windows you must call os.add_dll_directory() to allow linking to external DLLs.  See
    # https://docs.python.org/3.8/whatsnew/3.8.html#bpo-36085-whatsnew.
    os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
except (AttributeError,KeyError):
    pass

from _dlib_pybind11 import *
from _dlib_pybind11 import __version__, __time_compiled__
