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
            - pixel_traits<typename image_traits<image_type>::pixel_type>::has_alpha == false
        ensures
            - Finds a threshold value that would be reasonable to use with
              threshold_image(img, threshold).  It does this by finding the threshold that
              partitions the pixels in img into two groups such that the sum of absolute
              deviations between each pixel and the mean of its group is minimized.
    !*/

    template <
        typename image_type,
        typename ...T
        >
    void partition_pixels (
        const image_type& img,
        typename pixel_traits<typename image_traits<image_type>::pixel_type>::basic_pixel_type& pix_thresh,
        T&& ...more_thresholds
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - pixel_traits<typename image_traits<image_type>::pixel_type>::has_alpha == false
            - more_thresholds == a bunch of parameters of the same type as pix_thresh.
        ensures
            - This version of partition_pixels() finds multiple partitions rather than just
              one partition.  It does this by first partitioning the pixels just as the
              above partition_pixels(img) does.  Then it forms a new image with only pixels
              >= that first partition value and recursively partitions this new image.
              However, the recursion is implemented in an efficient way which is faster than
              explicitly forming these images and calling partition_pixels(), but the
              output is the same as if you did.  For example, suppose you called
              partition_pixels(img, t1, t2, t3).  Then we would have:
                - t1 == partition_pixels(img)
                - t2 == partition_pixels(an image with only pixels with values >= t1 in it)
                - t3 == partition_pixels(an image with only pixels with values >= t2 in it)
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
        typename in_image_type,
        typename out_image_type
        >
    void threshold_image (
        const in_image_type& in_img,
        out_image_type& out_img
    );
    /*!
        requires
            - it is valid to call threshold_image(in_img,out_img,partition_pixels(in_img));
        ensures
            - calls threshold_image(in_img,out_img,partition_pixels(in_img));
    !*/

    template <
        typename image_type
        >
    void threshold_image (
        image_type& img
    );
    /*!
        requires
            - it is valid to call threshold_image(img,img,partition_pixels(img));
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

    template <
        typename in_image_type,
        typename out_image_type
        >
    void hysteresis_threshold (
        const in_image_type& in_img,
        out_image_type& out_img
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
            - is_same_object(in_img, out_img) == false
        ensures
            - performs: hysteresis_threshold(in_img, out_img, t1, t2) where the thresholds
              are first obtained by calling partition_pixels(in_img, t1, t2).
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_THRESHOLDINg_ABSTRACT_ 


