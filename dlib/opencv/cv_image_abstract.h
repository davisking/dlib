// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OPENCV_IMAGE_AbSTRACT_H_
#ifdef DLIB_OPENCV_IMAGE_AbSTRACT_H_

#include "../algs.h"

namespace dlib
{

    template <
        typename pixel_type
        >
    class cv_image
    {
        /*!
            REQUIREMENTS ON pixel_type
                pixel_type just needs to be something that matches the pixel memory
                layout of whatever open cv image you are going to use this object
                with.  For example, you might use unsigned char or bgr_pixel depending
                on what you needed.

            WHAT THIS OBJECT REPRESENTS
                This object is meant to be used as a simple wrapper around the OpenCV
                IplImage struct.  Using this class template you can turn an IplImage
                object into something that looks like a normal dlib style image object.

                So you should be able to use cv_image objects with many of the image
                processing functions in dlib as well as the GUI tools for displaying
                images on the screen.

                Note that this object does NOT take ownership of the IplImage pointer
                you give to it.  This means you must still remember to free this pointer
                yourself.
        !*/

    public:
        typedef pixel_type type;

        cv_image (
            const IplImage* img
        );
        /*!
            requires
                - img->dataOrder == 0
                  (i.e. Only interleaved color channels are supported with cv_image)
                - (img->depth&0xFF)/8*img->nChannels == sizeof(pixel_type)
                  (i.e. The size of the pixel_type needs to match the size of the pixels 
                  inside the open cv image)
            ensures
                - #nr() == img->height
                  #nc() == img->width
                - using the operator[] on this object you will be able to access the pixels
                  inside this open cv image.
        !*/

        cv_image(
        ); 
        /*!
            ensures
                - #nr() == 0
                - #nc() == 0
        !*/

        ~cv_image (
        );
        /*!
            ensures
                - This function does nothing.  e.g. It doesn't delete the IplImage open cv 
                  image used by this cv_image object
        !*/

        long nr(
        ) const; 
        /*!
            ensures
                - returns the number of rows in this image
        !*/

        long nc(
        ) const;
        /*!
            ensures
                - returns the number of columns in this image
        !*/

        unsigned long size (
        ) const; 
        /*!
            ensures
                - returns nr()*nc()
                  (i.e. returns the number of pixels in this image)
        !*/

        inline pixel_type* operator[] (
            const long row 
        );
        /*!
            requires
                - 0 <= row < nr()
            ensures
                - returns a pointer to the first pixel in the given row
                  of this image
        !*/

        inline const pixel_type* operator[] (
            const long row 
        ) const;
        /*!
            requires
                - 0 <= row < nr()
            ensures
                - returns a pointer to the first pixel in the given row
                  of this image
        !*/

        cv_image& operator= ( 
            const cv_image& item
        );
        /*!
            ensures
                - #*this is an identical copy of item
                - returns #*this
        !*/

        cv_image& operator=( 
            const IplImage* img
        );
        /*!
            requires
                - img->dataOrder == 0
                  (i.e. Only interleaved color channels are supported with cv_image)
                - (img->depth&0xFF)/8*img->nChannels == sizeof(pixel_type)
                  (i.e. The size of the pixel_type needs to match the size of the pixels 
                  inside the open cv image)
            ensures
                - #nr() == img->height
                  #nc() == img->width
                - using the operator[] on this object you will be able to access the pixels
                  inside this open cv image.
                - returns #*this
        !*/

    };


}

#endif // DLIB_OPENCV_IMAGE_AbSTRACT_H_

