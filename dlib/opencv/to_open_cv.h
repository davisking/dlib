// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TO_OPEN_Cv_H__
#define DLIB_TO_OPEN_Cv_H__

#include "to_open_cv_abstract.h"
#include "../pixel.h"
#include "../matrix/matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    cv::Mat toMat (
        image_type& img
    )
    {
        if (img.size() == 0)
            return cv::Mat();

        typedef typename image_type::type type;
        if (pixel_traits<type>::num == 1)
        {
            return cv::Mat(img.nr(), img.nc(), cv::DataType<type>::type, (void*)&img[0][0], img.width_step());
        }
        else
        {
            int depth = sizeof(typename pixel_traits<type>::basic_pixel_type)*8;
            int channels = pixel_traits<type>::num;
            int thetype = CV_MAKETYPE(depth, channels);
            return cv::Mat(img.nr(), img.nc(), thetype, (void*)&img[0][0], img.width_step());
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        long NR,
        long NC,
        typename MM
        >
    cv::Mat toMat (
        matrix<T,NR,NC,MM>& img
    )
    {
        if (img.size() == 0)
            return cv::Mat();

        typedef T type;
        if (pixel_traits<type>::num == 1)
        {
            return cv::Mat(img.nr(), img.nc(), cv::DataType<type>::type, (void*)&img(0,0), img.nc()*sizeof(type));
        }
        else
        {
            int depth = sizeof(typename pixel_traits<type>::basic_pixel_type)*8;
            int channels = pixel_traits<type>::num;
            int thetype = CV_MAKETYPE(depth, channels);
            return cv::Mat(img.nr(), img.nc(), thetype, (void*)&img(0,0), img.nc()*sizeof(type));
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TO_OPEN_Cv_H__

