// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SERIALIZE_PiCKLE_H__
#define DLIB_SERIALIZE_PiCKLE_H__

#include <dlib/serialize.h>
#include <boost/python.hpp>
#include <sstream>

template <typename T>
struct serialize_pickle : boost::python::pickle_suite
{
    static boost::python::tuple getstate(
        const T& item 
    )
    {
        using namespace dlib;
        std::ostringstream sout;
        serialize(item, sout);
        return boost::python::make_tuple(sout.str());
    }

    static void setstate(
        T& item, 
        boost::python::tuple state
    )
    {
        using namespace dlib;
        using namespace boost::python;
        if (len(state) != 1)
        {
            PyErr_SetObject(PyExc_ValueError,
                ("expected 1-item tuple in call to __setstate__; got %s"
                 % state).ptr()
            );
            throw_error_already_set();
        }

        str data = extract<str>(state[0]);
        std::string temp(extract<const char*>(data), len(data));
        std::istringstream sin(temp);
        deserialize(item, sin);
    }
};

#endif // DLIB_SERIALIZE_PiCKLE_H__

