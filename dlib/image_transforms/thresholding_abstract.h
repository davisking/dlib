// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_THRESHOLDINg_ABSTRACT_
#ifdef DLIB_THRESHOLDINg_ABSTRACT_ 

#include "../pixel.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    const unsigned char on_pixel = 255;
    const unsigned char off_pixel = 0;

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type 
    partition_pixels (
        const image_type& img
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - pixel_traits<typename image_traits<image_type>::pixel_type>::max() <= 65535 
            - pixel_traits<typename image_traits<image_type>::pixel_type>::has_alpha   == false
            - pixel_traits<typename image_traits<image_type>::pixel_type>::is_unsigned == true 
        ensures
            - Finds a threshold value that would be reasonable to use with
              threshold_image(img, threshold).  It does this by finding the threshold that
              partitions the pixels in img into two groups such that the sum of absolute
              deviations between each pixel and the mean of its group is minimized.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void threshold_image (
        const in_image_type& in_img,
        out_image_type& out_img,
        typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type thresh
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - pixel_traits<typename image_traits<out_image_type>::pixel_type>::grayscale == true  
            - pixel_traits<typename image_traits<in_image_type>::pixel_type>::has_alpha == false
            - pixel_traits<typename image_traits<out_image_type>::pixel_type>::has_alpha == false 
        ensures
            - #out_img == the thresholded version of in_img (in_img is converted to a grayscale
              intensity image if it is color).  Pixels in in_img with grayscale values >= thresh 
              have an output value of on_pixel and all others have a value of off_pixel.
            - #out_img.nc() == in_img.nc()
            - #out_img.nr() == in_img.nr()
    !*/

    template <
        typename image_type
        >
    void threshold_image (
        image_type& img,
        typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type thresh
    );
    /*!
        requires
            - it is valid to call threshold_image(img,img,thresh);
        ensures
            - calls threshold_image(img,img,thresh);
    !*/

    template <
        typename image_type
        >
    void threshold_image (
        image_type& img
    );
    /*!
        requires
            - it is valid to call threshold_image(img,img,thresh);
        ensures
            - calls threshold_image(img,img,partition_pixels(img));
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void hysteresis_threshold (
        const in_image_type& in_img,
        out_image_type& out_img,
        typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type lower_thresh,
        typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type upper_thresh
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - pixel_traits<typename image_traits<out_image_type>::pixel_type>::grayscale == true  
            - pixel_traits<typename image_traits<in_image_type>::pixel_type>::has_alpha == false
            - pixel_traits<typename image_traits<out_image_type>::pixel_type>::has_alpha == false 
            - lower_thresh <= upper_thresh
            - is_same_object(in_img, out_img) == false
        ensures
            - #out_img == the hysteresis thresholded version of in_img (in_img is converted to a 
              grayscale intensity image if it is color). Pixels in in_img with grayscale 
              values >= upper_thresh have an output value of on_pixel and all others have a 
              value of off_pixel unless they are >= lower_thresh and are connected to a pixel
              with a value >= upper_thresh, in which case they have a value of on_pixel.  Here
              pixels are connected if there is a path between them composed of pixels that 
              would receive an output of on_pixel.
            - #out_img.nc() == in_img.nc()
            - #out_img.nr() == in_img.nr()
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_THRESHOLDINg_ABSTRACT_ 


