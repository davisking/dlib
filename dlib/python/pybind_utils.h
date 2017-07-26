// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PYBIND_UtILS_Hh_
#define DLIB_PYBIND_UtILS_Hh_

#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <dlib/serialize.h>

namespace py = pybind11;

template <typename T>
std::vector<T> python_list_to_vector (
    const py::list& obj
)
/*!
    ensures
        - converts a python object into a std::vector<T> and returns it.
!*/
{
    std::vector<T> vect(len(obj));
    for (unsigned long i = 0; i < vect.size(); ++i)
    {
        vect[i] = obj[i].cast<T>();
    }
    return vect;
}

template <typename T>
py::list vector_to_python_list (
    const std::vector<T>& vect
)
/*!
    ensures
        - converts a std::vector<T> into a python list object.
!*/
{
    py::list obj;
    for (unsigned long i = 0; i < vect.size(); ++i)
        obj.append(vect[i]);
    return obj;
}

template <typename T>
void extend_vector_with_python_list (
    std::vector<T> &v,
    const py::list &l
)
/*!
    ensures
        - appends items from a python list to the end of std::vector<T>.
!*/
{
    for (const auto &item : l)
        v.push_back(item.cast<T>());
}

// ----------------------------------------------------------------------------------------

template <typename T>
std::shared_ptr<T> load_object_from_file (
    const std::string& filename
)
/*!
    ensures
        - deserializes an object of type T from the given file and returns it.
!*/
{
    std::ifstream fin(filename.c_str(), std::ios::binary);
    if (!fin)
        throw dlib::error("Unable to open " + filename);
    auto obj = std::make_shared<T>();
    deserialize(*obj, fin);
    return obj;
}

// ----------------------------------------------------------------------------------------


#endif // DLIB_PYBIND_UtILS_Hh_

