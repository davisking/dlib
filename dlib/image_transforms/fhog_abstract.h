// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_fHOG_ABSTRACT_H__
#ifdef DLIB_fHOG_ABSTRACT_H__

#include "../matrix/matrix_abstract.h"
#include "../array2d/array2d_kernel_abstract.h"
#include "../array/array_kernel_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename T, 
        typename mm1, 
        typename mm2
        >
    void extract_fhog_features(
        const image_type& img, 
        dlib::array<array2d<T,mm1>,mm2>& hog, 
        int bin_size = 8
    );

    template <
        typename image_type, 
        typename T, 
        typename mm
        >
    void extract_fhog_features(
        const image_type& img, 
        array2d<matrix<T,31,1>,mm>& hog, 
        int bin_size = 8
    );

// ----------------------------------------------------------------------------------------

    inline point image_to_fhog (
        point p,
        int bin_size 
    );

// ----------------------------------------------------------------------------------------

    inline point fhog_to_image (
        point p,
        int bin_size 
    );

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename mm1, 
        typename mm2
        >
    matrix<unsigned char> draw_fhog(
        const dlib::array<array2d<T,mm1>,mm2>& hog,
        const long w = 15
    );

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename mm
        >
    matrix<unsigned char> draw_fhog(
        const array2d<matrix<T,31,1>,mm>& hog,
        const long w = 15
    );

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_fHOG_ABSTRACT_H__


