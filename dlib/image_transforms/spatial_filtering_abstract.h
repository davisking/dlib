// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SPATIAL_FILTERINg_ABSTRACT_
#ifdef DLIB_SPATIAL_FILTERINg_ABSTRACT_

#include "../pixel.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type,
        typename EXP,
        typename T
        >
    void spatially_filter_image (
        const in_image_type& in_img,
        out_image_type& out_img,
        const matrix_exp<EXP>& filter,
        T scale = 1,
        bool use_abs = false
    );
    /*!
        requires
            - in_image_type == is an implementation of array2d/array2d_kernel_abstract.h
            - out_image_type == is an implementation of array2d/array2d_kernel_abstract.h
            - pixel_traits<typename in_image_type::type>::has_alpha == false
            - pixel_traits<typename out_image_type::type>::has_alpha == false 
            - is_same_object(in_img, out_img) == false 
            - T must be some scalar type
            - scale != 0
            - filter.nr() % 2 == 1  (i.e. must be odd)
            - filter.nc() % 2 == 1  (i.e. must be odd)
        ensures
            - Applies the given spatial filter to in_img and stores the result in out_img.  Also 
              divides each resulting pixel by scale.  
            - Pixel values are stored into out_img using the assign_pixel() function and therefore
              any applicable color space conversion or value saturation is performed.
            - if (pixel_traits<typename in_image_type::type>::grayscale == false) then
                - the pixel values are converted to the HSI color space and the filtering
                  is done on the intensity channel only.
            - if (use_abs == true) then
                - pixel values after filtering that are < 0 are converted to their absolute values.
            - Pixels close enough to the edge of in_img to not have the filter still fit 
              inside the image are set to zero.
            - #out_img.nc() == in_img.nc()
            - #out_img.nr() == in_img.nr()
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type,
        typename EXP1,
        typename EXP2,
        typename T
        >
    void spatially_filter_image_separable (
        const in_image_type& in_img,
        out_image_type& out_img,
        const matrix_exp<EXP1>& row_filter,
        const matrix_exp<EXP2>& col_filter,
        T scale = 1,
        bool use_abs = false
    );
    /*!
        requires
            - in_image_type == is an implementation of array2d/array2d_kernel_abstract.h
            - out_image_type == is an implementation of array2d/array2d_kernel_abstract.h
            - pixel_traits<typename in_image_type::type>::has_alpha == false
            - pixel_traits<typename out_image_type::type>::has_alpha == false 
            - is_same_object(in_img, out_img) == false 
            - T must be some scalar type
            - scale != 0
            - is_vector(row_filter) == true
            - is_vector(col_filter) == true
            - row_filter.size() % 2 == 1  (i.e. must be odd)
            - col_filter.size() % 2 == 1  (i.e. must be odd)
        ensures
            - Applies the given separable spatial filter to in_img and stores the result in out_img.  
              Also divides each resulting pixel by scale.  Calling this function has the same
              effect as calling the regular spatially_filter_image() routine with a filter,
              FILT, defined as follows: 
                - FILT(r,c) == col_filter(r)*row_filter(c)
            - Pixel values are stored into out_img using the assign_pixel() function and therefore
              any applicable color space conversion or value saturation is performed.
            - if (pixel_traits<typename in_image_type::type>::grayscale == false) then
                - the pixel values are converted to the HSI color space and the filtering
                  is done on the intensity channel only.
            - if (use_abs == true) then
                - pixel values after filtering that are < 0 are converted to their absolute values
            - Pixels close enough to the edge of in_img to not have the filter still fit 
              inside the image are set to zero.
            - #out_img.nc() == in_img.nc()
            - #out_img.nr() == in_img.nr()
    !*/

// ----------------------------------------------------------------------------------------

    template <
        long NR,
        long NC,
        typename T,
        typename in_image_type
        >
    inline void separable_3x3_filter_block_grayscale (
        T (&block)[NR][NC],
        const in_image_type& img,
        const long& r,
        const long& c,
        const T& fe1, 
        const T& fm,  
        const T& fe2 
    );
    /*!
        requires
            - in_image_type == is an implementation of array2d/array2d_kernel_abstract.h
            - pixel_traits<typename in_image_type::type> must be defined 
            - T should be a scalar type
            - shrink_rect(get_rect(img),1).contains(c,r)
            - shrink_rect(get_rect(img),1).contains(c+NC-1,r+NR-1)
        ensures
            - Filters the image in the sub-window of img defined by a rectangle 
              with its upper left corner at (c,r) and lower right at (c+NC-1,r+NR-1).
            - The output of the filter is stored in #block.  Note that img will be 
              interpreted as a grayscale image.
            - The filter used is defined by the separable filter [fe1 fm fe2].  So the
              spatial filter is thus:
                fe1*fe1  fe1*fm  fe2*fe1
                fe1*fm   fm*fm   fe2*fm
                fe1*fe2  fe2*fm  fe2*fe2
    !*/

// ----------------------------------------------------------------------------------------

    template <
        long NR,
        long NC,
        typename T,
        typename U,
        typename in_image_type
        >
    inline void separable_3x3_filter_block_rgb (
        T (&block)[NR][NC],
        const in_image_type& img,
        const long& r,
        const long& c,
        const U& fe1, 
        const U& fm, 
        const U& fe2
    );
    /*!
        requires
            - in_image_type == is an implementation of array2d/array2d_kernel_abstract.h
            - pixel_traits<typename in_image_type::type>::rgb == true
            - T should be a struct with .red .green and .blue members.
            - U should be a scalar type
            - shrink_rect(get_rect(img),1).contains(c,r)
            - shrink_rect(get_rect(img),1).contains(c+NC-1,r+NR-1)
        ensures
            - Filters the image in the sub-window of img defined by a rectangle 
              with its upper left corner at (c,r) and lower right at (c+NC-1,r+NR-1).
            - The output of the filter is stored in #block.  Note that the filter is applied
              to each color component independently.
            - The filter used is defined by the separable filter [fe1 fm fe2].  So the
              spatial filter is thus:
                fe1*fe1  fe1*fm  fe2*fe1
                fe1*fm   fm*fm   fe2*fm
                fe1*fe2  fe2*fm  fe2*fe2
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SPATIAL_FILTERINg_ABSTRACT_

