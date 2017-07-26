// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PYaSSERT_Hh_
#define DLIB_PYaSSERT_Hh_

#include <pybind11/pybind11.h>

#define pyassert(_exp,_message)                                             \
    {if ( !(_exp) )                                                         \
    {                                                                       \
        namespace py = pybind11;                                            \
        PyErr_SetString( PyExc_ValueError, _message );                      \
        throw py::error_already_set();                                      \
    }}

#endif // DLIB_PYaSSERT_Hh_

