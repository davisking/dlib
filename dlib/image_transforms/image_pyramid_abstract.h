// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_IMAGE_PYRaMID_ABSTRACT_H__
#ifdef DLIB_IMAGE_PYRaMID_ABSTRACT_H__

#include "../pixel.h"
#include "../array2d.h"

namespace dlib
{

    class pyramid_down : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a simple functor to help create image pyramids.
        !*/
    public:

        template <
            typename in_image_type,
            typename out_image_type
            >
        void operator() (
            const in_image_type& original,
            out_image_type& down
        );
        /*!
            requires
                - original.nr() > 10
                - original.nc() > 10
                - is_same_object(original, down) == false
                - in_image_type == is an implementation of array2d/array2d_kernel_abstract.h
                - out_image_type == is an implementation of array2d/array2d_kernel_abstract.h
                - pixel_traits<typename in_image_type::type>::has_alpha == false
                - pixel_traits<typename out_image_type::type>::has_alpha == false
            ensures
                - #down will contain an image that is roughly half the size of the original
                  image.  To be specific, this function performs the following steps:
                    - 1. Applies a 5x5 Gaussian filter to the original image to smooth it a little.
                    - 2. Every other row and column is discarded to create an image half the size
                         of the original.  This smaller image is stored in #down.
        !*/
    };

}

#endif // DLIB_IMAGE_PYRaMID_ABSTRACT_H__


