// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SERIALIZE_PiCKLE_Hh_
#define DLIB_SERIALIZE_PiCKLE_Hh_

#include <dlib/serialize.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <dlib/vectorstream.h>

template<typename T>
py::tuple getstate(const T& item)
{
    using namespace dlib;
    std::vector<char> buf;
    buf.reserve(5000);
    vectorstream sout(buf);
    serialize(item, sout);
    return py::make_tuple(py::handle(
           PyBytes_FromStringAndSize(buf.size()?&buf[0]:0, buf.size())));
}

template<typename T>
T setstate(py::tuple state)
{
    using namespace dlib;
    if (len(state) != 1)
    {
        PyErr_SetObject(PyExc_ValueError,
                        py::str("expected 1-item tuple in call to __setstate__; got {}").format(state).ptr()
        );
        throw py::error_already_set();
    }

    // We used to serialize by converting to a str but the boost.python routines for
    // doing this don't work in Python 3.  You end up getting an error about invalid
    // UTF-8 encodings.  So instead we access the python C interface directly and use
    // bytes objects.  However, we keep the deserialization code that worked with str
    // for backwards compatibility with previously pickled files.
    T item;
    py::object obj = state[0];
    if (py::isinstance<py::str>(obj))
    {
        py::str data = state[0].cast<py::str>();
        std::string temp = data;
        std::istringstream sin(temp);
        deserialize(item, sin);
    }
    else if(PyBytes_Check(py::object(state[0]).ptr()))
    {
        py::object obj = state[0];
        char* data = PyBytes_AsString(obj.ptr());
        unsigned long num = PyBytes_Size(obj.ptr());
        std::istringstream sin(std::string(data, num));
        deserialize(item, sin);
    }
    else
    {
        throw error("Unable to unpickle, error in input file.");
    }

    return item;
}

#endif // DLIB_SERIALIZE_PiCKLE_Hh_

