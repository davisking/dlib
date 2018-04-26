// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_EDGE_DETECTOr_ABSTRACT_
#ifdef DLIB_EDGE_DETECTOr_ABSTRACT_

#include "../pixel.h"
#include "../image_processing/generic_image.h"
#include "../geometry.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    inline char edge_orientation (
        const T& x,
        const T& y
    );
    /*!
        ensures
            - returns the orientation of the line drawn from the origin to the point (x,y).
              The orientation is represented pictorially using the four ascii 
              characters /,|,\, and -.
            - if (the line is horizontal) then 
                returns '-' 
            - if (the line is vertical) then 
                returns '|' 
            - if (the line is diagonal with a positive slope) then 
                returns '/' 
            - if (the line is diagonal with a negative slope) then 
                returns '\\' 
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void sobel_edge_detector (
        const in_image_type& in_img,
        out_image_type& horz,
        out_image_type& vert
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type must use signed grayscale pixels
            - is_same_object(in_img,horz) == false
            - is_same_object(in_img,vert) == false
            - is_same_object(horz,vert) == false
        ensures
            - Applies the sobel edge detector to the given input image and stores the resulting
              edge detections in the horz and vert images
            - #horz.nr() == in_img.nr()
            - #horz.nc() == in_img.nc()
            - #vert.nr() == in_img.nr()
            - #vert.nc() == in_img.nc()
            - for all valid r and c:    
                - #horz[r][c] == the magnitude of the horizontal gradient at the point in_img[r][c]
                - #vert[r][c] == the magnitude of the vertical gradient at the point in_img[r][c]
                - edge_orientation(#vert[r][c], #horz[r][c]) == the edge direction at this point in 
                  the image
    !*/
    
// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void suppress_non_maximum_edges (
        const in_image_type& horz,
        const in_image_type& vert,
        out_image_type& out_img
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - horz.nr() == vert.nr()
            - horz.nc() == vert.nc()
            - is_same_object(out_img, horz) == false
            - is_same_object(out_img, vert) == false
            - image_traits<in_image_type>::pixel_type == A signed scalar type (e.g. int, double, etc.) 
        ensures
            - #out_img.nr() = horz.nr()
            - #out_img.nc() = horz.nc()
            - let edge_strength(r,c) == sqrt(pow(horz[r][c],2) + pow(vert[r][c],2))
              (i.e. The Euclidean norm of the gradient)
            - for all valid r and c:
                - if (edge_strength(r,c) is at a maximum with respect to its 2 neighboring
                  pixels along the line given by edge_orientation(vert[r][c],horz[r][c])) then
                    - performs assign_pixel(#out_img[r][c], edge_strength(r,c))
                - else
                    - performs assign_pixel(#out_img[r][c], 0)
    !*/
    
// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void normalize_image_gradients (
        image_type& img1,
        image_type& img2 
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - image_type contains float, double, or long double pixels.
            - img1.nr() == img2.nr()
            - img1.nc() == img2.nc()
        ensures
            - #out_img.nr() = img1.nr()
            - #out_img.nc() = img1.nc()
            - This function assumes img1 and img2 are the two gradient images produced by a
              function like sobel_edge_detector().  It then unit normalizes the gradient
              vectors. That is, for all valid r and c, this function ensures that:
                - #img1[r][c]*img1[r][c] + #img2[r][c]*img2[r][c] == 1 
                  unless both img1[r][c] and img2[r][c] were 0 initially, then they stay zero.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    std::vector<point> remove_incoherent_edge_pixels (
        const std::vector<point>& line,
        const image_type& horz_gradient,
        const image_type& vert_gradient,
        const double angle_threshold
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - image_type contains float, double, or long double pixels.
            - horz_gradient.nr() == vert_gradient.nr()
            - horz_gradient.nc() == vert_gradient.nc()
            - horz_gradient and vert_gradient represent unit normalized vectors.  That is,
              you should have called normalize_image_gradients(horz_gradient,vert_gradient)
              or otherwise caused all the gradients to have unit norm.
            - for all valid i:
                get_rect(horz_gradient).contains(line[i])
        ensures
            - This routine looks at all the points in the given line and discards the ones that
              have outlying gradient directions.  To be specific, this routine returns a set
              of points PTS such that: 
                - for all valid i,j:
                    - The difference in angle between the gradients for PTS[i] and PTS[j] is 
                      less than angle_threshold degrees.  
                - PTS.size() <= line.size()
                - PTS is just line with some elements removed.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_EDGE_DETECTOr_ABSTRACT_


