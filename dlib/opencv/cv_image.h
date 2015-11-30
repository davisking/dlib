// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CvIMAGE_H_
#define DLIB_CvIMAGE_H_

#include <opencv2/core/core.hpp>
#include "cv_image_abstract.h"
#include "../algs.h"
#include "../pixel.h"
#include "../matrix/matrix_mat.h"
#include "../image_processing/generic_image.h"

namespace dlib
{

    template <
        typename pixel_type
        >
    class cv_image
    {
    public:
        typedef pixel_type type;
        typedef default_memory_manager mem_manager_type;

        cv_image (const cv::Mat img) 
        {
            DLIB_CASSERT(img.depth() == cv::DataType<typename pixel_traits<pixel_type>::basic_pixel_type>::depth &&
                         img.channels() == pixel_traits<pixel_type>::num,
                         "The pixel type you gave doesn't match pixel used by the open cv Mat object."
                         << "\n\t img.depth():    " << img.depth() 
                         << "\n\t img.cv::DataType<typename pixel_traits<pixel_type>::basic_pixel_type>::depth: " 
                            << cv::DataType<typename pixel_traits<pixel_type>::basic_pixel_type>::depth 
                         << "\n\t img.channels(): " << img.channels() 
                         << "\n\t img.pixel_traits<pixel_type>::num: " << pixel_traits<pixel_type>::num 
                         );
            init(&img);
        }

        cv_image() : _data(0), _widthStep(0), _nr(0), _nc(0) {}

        unsigned long size () const { return static_cast<unsigned long>(_nr*_nc); }

        inline pixel_type* operator[](const long row ) 
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(0 <= row && row < nr(),
                "\tpixel_type* cv_image::operator[](row)"
                << "\n\t you have asked for an out of bounds row " 
                << "\n\t row:  " << row
                << "\n\t nr(): " << nr() 
                << "\n\t this:  " << this
                );

            return reinterpret_cast<pixel_type*>( _data + _widthStep*row);
        }

        inline const pixel_type* operator[](const long row ) const
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(0 <= row && row < nr(),
                "\tconst pixel_type* cv_image::operator[](row)"
                << "\n\t you have asked for an out of bounds row " 
                << "\n\t row:  " << row
                << "\n\t nr(): " << nr() 
                << "\n\t this:  " << this
                );

            return reinterpret_cast<const pixel_type*>( _data + _widthStep*row);
        }

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long width_step() const { return _widthStep; }

        cv_image& operator=( const cv_image& item)
        {
            _data = item._data;
            _widthStep = item._widthStep;
            _nr = item._nr;
            _nc = item._nc;
            return *this;
        }

        cv_image& operator=( const cv::Mat img)
        {
            init(&img);
            return *this;
        }

    private:

        void init (const cv::Mat* img)
        {
            _data = (char*) img->data;
            _widthStep = img->step;
            _nr = img->rows;
            _nc = img->cols;

        }

        char* _data;
        long _widthStep;
        long _nr;
        long _nc;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    const matrix_op<op_array2d_to_mat<cv_image<T> > > mat (
        const cv_image<T>& m 
    )
    {
        typedef op_array2d_to_mat<cv_image<T> > op;
        return matrix_op<op>(op(m));
    }

// ----------------------------------------------------------------------------------------

// Define the global functions that make cv_image a proper "generic image" according to
// ../image_processing/generic_image.h
    template <typename T>
    struct image_traits<cv_image<T> >
    {
        typedef T pixel_type;
    };

    template <typename T>
    inline long num_rows( const cv_image<T>& img) { return img.nr(); }
    template <typename T>
    inline long num_columns( const cv_image<T>& img) { return img.nc(); }

    template <typename T>
    inline void* image_data(
        cv_image<T>& img
    )
    {
        if (img.size() != 0)
            return &img[0][0];
        else
            return 0;
    }

    template <typename T>
    inline const void* image_data(
        const cv_image<T>& img
    )
    {
        if (img.size() != 0)
            return &img[0][0];
        else
            return 0;
    }

    template <typename T>
    inline long width_step(
        const cv_image<T>& img
    ) 
    { 
        return img.width_step(); 
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CvIMAGE_H_

