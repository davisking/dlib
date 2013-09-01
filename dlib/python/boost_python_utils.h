// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BOOST_PYTHON_UtILS_H__
#define DLIB_BOOST_PYTHON_UtILS_H__

#include <boost/python.hpp>
#include <vector>
#include <string>

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

// ----------------------------------------------------------------------------------------

template <typename T>
std::vector<T> python_list_to_vector (
    const boost::python::object& obj
)
/*!
    ensures
        - converts a python object into a std::vector<T> and returns it.
!*/
{
    std::vector<T> vect(len(obj));
    for (unsigned long i = 0; i < vect.size(); ++i)
    {
        vect[i] = boost::python::extract<T>(obj[i]);
    }
    return vect;
}

template <typename T>
boost::python::list vector_to_python_list (
    const std::vector<T>& vect
)
/*!
    ensures
        - converts a std::vector<T> into a python list object.
!*/
{
    boost::python::list obj;
    for (unsigned long i = 0; i < vect.size(); ++i)
        obj.append(vect[i]);
    return obj;
}

// ----------------------------------------------------------------------------------------


#endif // DLIB_BOOST_PYTHON_UtILS_H__

