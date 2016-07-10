// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PYTHON_NuMPY_Hh_
#define DLIB_PYTHON_NuMPY_Hh_

#include <boost/python.hpp>
#include <dlib/error.h>
#include <dlib/algs.h>
#include <dlib/string.h>
#include <dlib/array.h>
#include <dlib/pixel.h>

// ----------------------------------------------------------------------------------------

template <typename T>
void validate_numpy_array_type (
    const boost::python::object& obj
)
{
    using namespace boost::python;
    const char ch = extract<char>(obj.attr("dtype").attr("char"));

    if (dlib::is_same_type<T,double>::value && ch != 'd')
        throw dlib::error("Expected numpy.ndarray of float64");
    if (dlib::is_same_type<T,float>::value && ch != 'f')
        throw dlib::error("Expected numpy.ndarray of float32");
    if (dlib::is_same_type<T,dlib::int32>::value && ch != 'i')
        throw dlib::error("Expected numpy.ndarray of int32");
    if (dlib::is_same_type<T,unsigned char>::value && ch != 'B')
        throw dlib::error("Expected numpy.ndarray of uint8");
}

// ----------------------------------------------------------------------------------------

template <int dims>
void get_numpy_ndarray_shape (
    const boost::python::object& obj,
    long (&shape)[dims]
)
/*!
    ensures
        - stores the shape of the array into #shape.
        - the dimension of the given numpy array is not greater than #dims.
!*/
{
    Py_buffer pybuf;
    if (PyObject_GetBuffer(obj.ptr(), &pybuf, PyBUF_STRIDES ))
        throw dlib::error("Expected numpy.ndarray with shape set.");

    try
    {

        if (pybuf.ndim > dims)
            throw dlib::error("Expected array with " + dlib::cast_to_string(dims) + " dimensions.");

        for (int i = 0; i < dims; ++i)
        {
            if (i < pybuf.ndim)
                shape[i] = pybuf.shape[i];
            else
                shape[i] = 1;
        }
    }
    catch(...)
    {
        PyBuffer_Release(&pybuf);
        throw;
    }
    PyBuffer_Release(&pybuf);
}

// ----------------------------------------------------------------------------------------

template <typename T, int dims>
void get_numpy_ndarray_parts (
    boost::python::object& obj,
    T*& data,
    dlib::array<T>& contig_buf,
    long (&shape)[dims]
)
/*!
    ensures
        - extracts the pointer to the data from the given numpy ndarray.  Stores the shape
          of the array into #shape.
        - the dimension of the given numpy array is not greater than #dims.
        - #shape[#dims-1] == pixel_traits<T>::num when #dims is greater than 2
!*/
{
    Py_buffer pybuf;
    if (PyObject_GetBuffer(obj.ptr(), &pybuf, PyBUF_STRIDES | PyBUF_WRITABLE ))
        throw dlib::error("Expected writable numpy.ndarray with shape set.");

    try
    {
        validate_numpy_array_type<T>(obj);

        if (pybuf.ndim > dims)
            throw dlib::error("Expected array with " + dlib::cast_to_string(dims) + " dimensions.");
        get_numpy_ndarray_shape(obj, shape);

        if (dlib::pixel_traits<T>::num > 1 && dlib::pixel_traits<T>::num != shape[dims-1])
            throw dlib::error("Expected numpy.ndarray with " + dlib::cast_to_string(dlib::pixel_traits<T>::num) + " channels.");

        if (PyBuffer_IsContiguous(&pybuf, 'C'))
            data = (T*)pybuf.buf;
        else
        {
            contig_buf.resize(pybuf.len);
            if (PyBuffer_ToContiguous(&contig_buf[0], &pybuf, pybuf.len, 'C'))
                throw dlib::error("Can't copy numpy.ndarray to a contiguous buffer.");
            data = &contig_buf[0];
        }
    }
    catch(...)
    {
        PyBuffer_Release(&pybuf);
        throw;
    }
    PyBuffer_Release(&pybuf);
}

// ----------------------------------------------------------------------------------------

template <typename T, int dims>
void get_numpy_ndarray_parts (
    const boost::python::object& obj,
    const T*& data,
    dlib::array<T>& contig_buf,
    long (&shape)[dims]
)
/*!
    ensures
        - extracts the pointer to the data from the given numpy ndarray.  Stores the shape
          of the array into #shape.
        - the dimension of the given numpy array is not greater than #dims.
        - #shape[#dims-1] == pixel_traits<T>::num when #dims is greater than 2
!*/
{
    Py_buffer pybuf;
    if (PyObject_GetBuffer(obj.ptr(), &pybuf, PyBUF_STRIDES ))
        throw dlib::error("Expected numpy.ndarray with shape set.");

    try
    {
        validate_numpy_array_type<T>(obj);

        if (pybuf.ndim > dims)
            throw dlib::error("Expected array with " + dlib::cast_to_string(dims) + " dimensions.");
        get_numpy_ndarray_shape(obj, shape);

        if (dlib::pixel_traits<T>::num > 1 && dlib::pixel_traits<T>::num != shape[dims-1])
            throw dlib::error("Expected numpy.ndarray with " + dlib::cast_to_string(dlib::pixel_traits<T>::num) + " channels.");

        if (PyBuffer_IsContiguous(&pybuf, 'C'))
            data = (const T*)pybuf.buf;
        else
        {
            contig_buf.resize(pybuf.len);
            if (PyBuffer_ToContiguous(&contig_buf[0], &pybuf, pybuf.len, 'C'))
                throw dlib::error("Can't copy numpy.ndarray to a contiguous buffer.");
            data = &contig_buf[0];
        }
    }
    catch(...)
    {
        PyBuffer_Release(&pybuf);
        throw;
    }
    PyBuffer_Release(&pybuf);
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_PYTHON_NuMPY_Hh_

