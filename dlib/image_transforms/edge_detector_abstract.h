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
                - img1[r][c]*img1[r][c] + img2[r][c]*img2[r][c] == 1 
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

    template <
        typename image_type
        >
    std::vector<std::vector<point>> remove_incoherent_edge_pixels (
        const std::vector<std::vector<point>>& lines,
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
            - for all valid i,j:
                get_rect(horz_gradient).contains(lines[i][j])
        ensures
            - Returns a vector LINES where:
                - LINES.size() == lines.size()
                - LINES[i] == remove_incoherent_edge_pixels(lines[i], horz_gradient, vert_gradient, angle_threshold)
    !*/

// ----------------------------------------------------------------------------------------

    class image_gradients
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class is a tool for computing first and second derivatives of an
                image.  It does this by fitting a quadratic surface around each pixel and
                then computing the gradients of that quadratic surface.  For the details
                see the paper:
                    Quadratic models for curved line detection in SAR CCD by Davis E. King
                    and Rhonda D. Phillips

                This technique gives very accurate gradient estimates and is also very fast
                since the entire gradient estimation procedure, for each type of gradient,
                is accomplished by cross-correlating the image with a single separable
                filter.  This means you can compute gradients at very large scales (e.g. by
                fitting the quadratic to a large window, like a 99x99 window) and it still
                runs very quickly.
        !*/

    public:

        image_gradients (
        );
        /*!
            ensures
                - #get_scale() == 1
        !*/

        image_gradients (
            long scale
        ); 
        /*!
            requires
                - scale >= 1
            ensures
                - #get_scale() == scale
        !*/

        long get_scale(
        ) const;
        /*!
            ensures
                - When we estimate a gradient we do so by fitting a quadratic filter to a
                  window of size get_scale()*2+1 centered on each pixel.  Therefore, the
                  scale parameter controls the size of gradients we will find.  For
                  example, a very large scale will cause the gradient_xx() to be
                  insensitive to high frequency noise in the image while smaller scales
                  would be more sensitive to such fluctuations in the image.
        !*/

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_x(
            const in_image_type& img,
            out_image_type& out
        ) const;
        /*!
            requires
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - out_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - img and out do not contain pixels with an alpha channel.  That is,
                  pixel_traits::has_alpha is false for the pixels in these objects.
                - is_same_object(img, out) == false 
                - out_image_type must use signed grayscale pixels
            ensures
                - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()).
                - Computes the 1st order gradient of img in the x direction at each
                  location in VALID_AREA.  The gradients are stored in out.  All pixels in
                  #out that are outside VALID_AREA are set to 0.
                - #num_rows(out) == num_rows(img)
                - #num_columns(out) == num_columns(img)
                - While not a requirement, it is a good idea if the output image contains
                  float or double pixels.  If get_scale() is small then this is less of an
                  issue, but at large scales the gradient can easily be a small number
                  which is not well represented by integer pixel type such as short. Also,
                  if you use float pixels in the input and output images then this routine
                  will use SIMD instructions and is particularly fast.
                - returns VALID_AREA.  That is, returns the part of the output image which
                  contains actual valid gradient values.
        !*/

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_y(
            const in_image_type& img,
            out_image_type& out
        ) const;
        /*!
            This routine is identical to gradient_x() (defined above) except that it
            computes the 1st order y gradient.
        !*/

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_xx(
            const in_image_type& img,
            out_image_type& out
        )  const;
        /*!
            This routine is identical to gradient_x() (defined above) except that it
            computes the 2nd order x gradient. 
        !*/

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_xy(
            const in_image_type& img,
            out_image_type& out
        ) const;
        /*!
            This routine is identical to gradient_x() (defined above) except that it
            computes the partial derivative with respect to x and y.
        !*/

        template <
            typename in_image_type,
            typename out_image_type
            >
        rectangle gradient_yy(
            const in_image_type& img,
            out_image_type& out
        ) const;
        /*!
            This routine is identical to gradient_x() (defined above) except that it
            computes the 2nd order y gradient. 
        !*/

        matrix<float> get_x_filter(
        )  const; 
        /*!
            ensures
                - Returns the filter used by gradient_x() to compute the image gradient.
                  That is, the output of gradient_x() is found by cross correlating the
                  filter get_x_filter() with the image.
                - The returned filter has get_scale()*2+1 rows and columns.
        !*/

        matrix<float> get_y_filter(
        )  const; 
        /*!
            ensures
                - Returns the filter used by gradient_y() to compute the image gradient.
                  That is, the output of gradient_y() is found by cross correlating the
                  filter get_y_filter() with the image.
                - The returned filter has get_scale()*2+1 rows and columns.
        !*/

        matrix<float> get_xx_filter(
        ) const; 
        /*!
            ensures
                - Returns the filter used by gradient_xx() to compute the image gradient.
                  That is, the output of gradient_xx() is found by cross correlating the
                  filter get_xx_filter() with the image.
                - The returned filter has get_scale()*2+1 rows and columns.
        !*/


        matrix<float> get_xy_filter(
        ) const; 
        /*!
            ensures
                - Returns the filter used by gradient_xy() to compute the image gradient.
                  That is, the output of gradient_xy() is found by cross correlating the
                  filter get_xy_filter() with the image.
                - The returned filter has get_scale()*2+1 rows and columns.
        !*/


        matrix<float> get_yy_filter(
        ) const; 
        /*!
            ensures
                - Returns the filter used by gradient_yy() to compute the image gradient.
                  That is, the output of gradient_yy() is found by cross correlating the
                  filter get_yy_filter() with the image.
                - The returned filter has get_scale()*2+1 rows and columns.
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_bright_lines(
        const in_image_type& xx,
        const in_image_type& xy,
        const in_image_type& yy,
        out_image_type& horz,
        out_image_type& vert
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - All images are grayscale and the horz and vert images must contain float or
              double pixel types.
            - num_rows(xx) == num_rows(xy) == num_rows(yy)
            - num_columns(xx) == num_columns(xy) == num_columns(yy)
        ensures
            - This routine is similar to sobel_edge_detector(), except instead of finding
              an edge it finds a bright/white line.  For example, the border between a
              black piece of paper and a white table is an edge, but a curve drawn with a
              pencil on a piece of paper makes a line.  Therefore, the output of this
              routine is a vector field encoded in the horz and vert images.  The vector
              obtains a large magnitude when centered on a bright line in an image and the
              direction of the vector is perpendicular to the line.  To be very precise,
              each vector points in the direction of greatest change in second derivative
              and the magnitude of the vector encodes the derivative magnitude in that
              direction.  Moreover, if the second derivative is positive then the output
              vector is zero.  This zeroing if positive gradients causes the output to be
              sensitive only to bright lines surrounded by darker pixels.
            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output images will have the same sizes as the input images, that is:
                - #num_rows(horz) == #num_rows(vert) == num_rows(xx)
                - #num_columns(horz) == #num_columns(vert) == num_columns(xx)
    !*/

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_dark_lines(
        const in_image_type& xx,
        const in_image_type& xy,
        const in_image_type& yy,
        out_image_type& horz,
        out_image_type& vert
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - All images are grayscale and the horz and vert images must contain float or
              double pixel types.
            - num_rows(xx) == num_rows(xy) == num_rows(yy)
            - num_columns(xx) == num_columns(xy) == num_columns(yy)
        ensures
            - This routine is similar to sobel_edge_detector(), except instead of finding
              an edge it finds a dark/black line.  For example, the border between a
              black piece of paper and a white table is an edge, but a curve drawn with a
              pencil on a piece of paper makes a line.  Therefore, the output of this
              routine is a vector field encoded in the horz and vert images.  The vector
              obtains a large magnitude when centered on a dark line in an image and the
              direction of the vector is perpendicular to the line.  To be very precise,
              each vector points in the direction of greatest change in second derivative
              and the magnitude of the vector encodes the derivative magnitude in that
              direction.  Moreover, if the second derivative is negative then the output
              vector is zero.  This zeroing if negative gradients causes the output to be
              sensitive only to dark lines surrounded by light pixels.
            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output images will have the same sizes as the input images, that is:
                - #num_rows(horz) == #num_rows(vert) == num_rows(xx)
                - #num_columns(horz) == #num_columns(vert) == num_columns(xx)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_bright_keypoints(
        const in_image_type& xx,
        const in_image_type& xy,
        const in_image_type& yy,
        out_image_type& saliency
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - All images are grayscale and the saliency image must contain float or double
              pixel types.
            - num_rows(xx) == num_rows(xy) == num_rows(yy)
            - num_columns(xx) == num_columns(xy) == num_columns(yy)
        ensures
            - This routine finds bright "keypoints" in an image.  In general, these are
              bright/white localized blobs.  It does this by computing the determinant of
              the image Hessian at each location and storing this value into the output
              saliency image if both eigenvalues of the Hessian are negative.  If either
              eigenvalue is positive then the saliency for that pixel is 0.  I.e.
                - for all valid r,c:
                    - #saliency[r][c] == a number >= 0 and larger values indicate the
                      presence of a keypoint at this pixel location.
            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output image will have the same size as the input images, that is:
                - #num_rows(saliency) == num_rows(xx)
                - #num_columns(saliency) == num_columns(xx)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    void find_dark_keypoints(
        const in_image_type& xx,
        const in_image_type& xy,
        const in_image_type& yy,
        out_image_type& saliency
    );
    /*!
        requires
            - in_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - out_image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - All images are grayscale and the saliency image must contain float or double
              pixel types.
            - num_rows(xx) == num_rows(xy) == num_rows(yy)
            - num_columns(xx) == num_columns(xy) == num_columns(yy)
        ensures
            - This routine finds dark "keypoints" in an image.  In general, these are dark
              localized blobs.  It does this by computing the determinant of the image
              Hessian at each location and storing this value into the output saliency
              image if both eigenvalues of the Hessian are positive.  If either eigenvalue
              is negative then the saliency for that pixel is 0.  I.e.
                - for all valid r,c:
                    - #saliency[r][c] == a number >= 0 and larger values indicate the
                      presence of a keypoint at this pixel location.
            - We assume that xx, xy, and yy are the 3 second order gradients of the image
              in question.  You can obtain these gradients using the image_gradients class.
            - The output image will have the same size as the input images, that is:
                - #num_rows(saliency) == num_rows(xx)
                - #num_columns(saliency) == num_columns(xx)
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_EDGE_DETECTOr_ABSTRACT_


