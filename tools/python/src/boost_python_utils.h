// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BOOST_PYTHON_UtILS_H__
#define DLIB_BOOST_PYTHON_UtILS_H__

#include <boost/python.hpp>

inline bool hasattr(
    boost::python::object obj, 
    const std::string& attr_name
) 
/*!
    ensures
        - if (obj has an attribute named attr_name) then
            - returns true
        - else
            - returns false
!*/
{
     return PyObject_HasAttrString(obj.ptr(), attr_name.c_str());
}

#endif // DLIB_BOOST_PYTHON_UtILS_H__

