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
        typename mm
        >
    void extract_fhog_features(
        const image_type& img, 
        array2d<matrix<T,31,1>,mm>& hog, 
        int cell_size = 8
    );
    /*!
        requires
            - cell_size > 0
            - in_image_type  == is an implementation of array2d/array2d_kernel_abstract.h
            - img contains some kind of pixel type. 
              (i.e. pixel_traits<typename in_image_type::type> is defined)
            - T should be float or double
        ensures
            - This function implements the HOG feature extraction method described in 
              the paper:
                Object Detection with Discriminatively Trained Part Based Models by
                P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan
                IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010
              This means that it takes an input image img and outputs Felzenszwalb's
              31 dimensional version of HOG features, which are stored into #hog.
            - The input image is broken into cells that are cell_size by cell_size pixels
              and within each cell we compute a 31 dimensional FHOG vector.  This vector
              describes the gradient structure within the cell.  
            - #hog.nr() is approximately equal to img.nr()/cell_size.
            - #hog.nc() is approximately equal to img.nc()/cell_size.
            - for all valid r and c:
                - #hog[r][c] == the FHOG vector describing the cell centered at the pixel
                  location fhog_to_image(point(c,r),cell_size) in img.
    !*/

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
        int cell_size = 8
    );
    /*!
        requires
            - cell_size > 0
            - in_image_type  == is an implementation of array2d/array2d_kernel_abstract.h
            - img contains some kind of pixel type. 
              (i.e. pixel_traits<typename in_image_type::type> is defined)
            - T should be float or double
        ensures
            - This function is identical to the above extract_fhog_features() routine
              except that it outputs the results in a planar format rather than the
              interlaced format used above.  That is, each element of the hog vector is
              placed into one of 31 images inside #hog.  To be precise, if vhog is the
              output of the above interlaced version of extract_fhog_features() then we
              will have, for all valid r and c:
                - #hog[i][r][c] == vhog[r][c](i)
                  (where 0 <= i < 31)
            - #hog.size() == 31
    !*/

// ----------------------------------------------------------------------------------------

    inline point image_to_fhog (
        point p,
        int cell_size = 8
    );
    /*!
        requires
            - cell_size > 0
        ensures
            - When using extract_fhog_features(), each FHOG cell is extracted from a
              certain region in the input image.  image_to_fhog() returns the identity of
              the FHOG cell containing the image pixel at location p.  Or in other words,
              let P == image_to_fhog(p) and hog be a FHOG feature map output by
              extract_fhog_features(), then hog[P.y()][P.x()] == the FHOG vector/cell
              containing the point p in the input image.  Note that some image points
              might not have corresponding feature locations.  E.g. border points or points
              outside the image.  In these cases the returned point will be outside the
              input image.
    !*/

// ----------------------------------------------------------------------------------------

    inline point fhog_to_image (
        point p,
        int cell_size = 8
    );
    /*!
        requires
            - cell_size > 0
        ensures
            - Maps a pixel in a FHOG image (produced by extract_fhog_features()) back to the
              corresponding original input pixel.  Note that since FHOG images are
              spatially downsampled by aggregation into cells the mapping is not totally
              invertible.  Therefore, the returned location will be the center of the cell
              in the original image that contained the FHOG vector at position p.  Moreover,
              cell_size should be set to the value used by the call to extract_fhog_features().
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename mm1, 
        typename mm2
        >
    matrix<unsigned char> draw_fhog(
        const dlib::array<array2d<T,mm1>,mm2>& hog,
        const long cell_draw_size = 15
    );
    /*!
        requires
            - cell_draw_size > 0
            - hog.size() == 31
        ensures
            - Interprets hog as a FHOG feature map output by extract_fhog_features() and
              converts it into an image suitable for display on the screen.  In particular,
              we draw all the hog cells into a grayscale image in a way that shows the
              magnitude and orientation of the gradient energy in each cell.  The result is
              then returned.
            - The size of the cells in the output image will be rendered as cell_draw_size 
              pixels wide and tall.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        typename mm
        >
    matrix<unsigned char> draw_fhog(
        const array2d<matrix<T,31,1>,mm>& hog,
        const long cell_draw_size = 15
    );
    /*!
        requires
            - cell_draw_size > 0
        ensures
            - Interprets hog as a FHOG feature map output by extract_fhog_features() and
              converts it into an image suitable for display on the screen.  In particular,
              we draw all the hog cells into a grayscale image in a way that shows the
              magnitude and orientation of the gradient energy in each cell.  The result is 
              then returned.
            - The size of the cells in the output image will be rendered as cell_draw_size 
              pixels wide and tall.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_fHOG_ABSTRACT_H__


