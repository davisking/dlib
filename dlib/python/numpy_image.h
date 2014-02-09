// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PYTHON_NuMPY_IMAGE_H__
#define DLIB_PYTHON_NuMPY_IMAGE_H__

#include "numpy.h"
#include <dlib/pixel.h>
#include <dlib/matrix.h>


// ----------------------------------------------------------------------------------------

class numpy_gray_image
{
public:
    typedef unsigned char type;
    typedef dlib::default_memory_manager mem_manager_type;

    numpy_gray_image() : _data(0), _nr(0), _nc(0) {}
    numpy_gray_image (boost::python::object& img) 
    {
        long shape[2];
        get_numpy_ndarray_parts(img, _data, shape);
        _nr = shape[0];
        _nc = shape[1];
    }

    unsigned long size () const { return static_cast<unsigned long>(_nr*_nc); }

    inline type* operator[](const long row ) 
    { return _data + _nc*row; }

    inline const type* operator[](const long row ) const
    { return _data + _nc*row; }

    long nr() const { return _nr; }
    long nc() const { return _nc; }
    long width_step() const { return nc()*sizeof(type); }

private:

    type* _data;
    long _nr;
    long _nc;
};

// ----------------------------------------------------------------------------------------

inline const dlib::matrix_op<dlib::op_array2d_to_mat<numpy_gray_image> > mat (
    const numpy_gray_image& m 
)
{
    using namespace dlib;
    typedef op_array2d_to_mat<numpy_gray_image> op;
    return matrix_op<op>(op(m));
}

// ----------------------------------------------------------------------------------------

inline bool is_gray_python_image (boost::python::object& img)
{
    try
    {
        numpy_gray_image temp(img);
        return true;
    }
    catch (dlib::error&)
    {
        return false;
    }
}

// ----------------------------------------------------------------------------------------

class numpy_rgb_image
{
public:
    typedef dlib::rgb_pixel type;
    typedef dlib::default_memory_manager mem_manager_type;

    numpy_rgb_image() : _data(0), _nr(0), _nc(0) {}
    numpy_rgb_image (boost::python::object& img) 
    {
        long shape[3];
        get_numpy_ndarray_parts(img, _data, shape);
        _nr = shape[0];
        _nc = shape[1];
        if (shape[2] != 3)
            throw dlib::error("Error, python object is not a three band image and therefore can't be a RGB image.");
    }

    unsigned long size () const { return static_cast<unsigned long>(_nr*_nc); }

    inline type* operator[](const long row ) 
    { return _data + _nc*row; }

    inline const type* operator[](const long row ) const
    { return _data + _nc*row; }

    long nr() const { return _nr; }
    long nc() const { return _nc; }
    long width_step() const { return nc()*sizeof(type); }

private:

    type* _data;
    long _nr;
    long _nc;
};

// ----------------------------------------------------------------------------------------

inline const dlib::matrix_op<dlib::op_array2d_to_mat<numpy_rgb_image> > mat (
    const numpy_rgb_image& m 
)
{
    using namespace dlib;
    typedef op_array2d_to_mat<numpy_rgb_image> op;
    return matrix_op<op>(op(m));
}

// ----------------------------------------------------------------------------------------

inline bool is_rgb_python_image (boost::python::object& img)
{
    try
    {
        numpy_rgb_image temp(img);
        return true;
    }
    catch (dlib::error&)
    {
        return false;
    }
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_PYTHON_NuMPY_IMAGE_H__

