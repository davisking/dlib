// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RANDOMLY_COlOR_IMAGE_ABSTRACT_H__
#ifdef DLIB_RANDOMLY_COlOR_IMAGE_ABSTRACT_H__

#include "randomly_color_image_abstract.h"
#include "../hash.h"
#include "../pixel.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    const matrix_exp randomly_color_image (
        const image_type& img
    );
    /*!
        requires
            - image_type is an implementation of array2d/array2d_kernel_abstract.h
            - pixel_traits<image_type::type> must be defined
        ensures
            - randomly generates a mapping from gray level pixel values
              to the RGB pixel space and then uses this mapping to create
              a colored version of img.  Returns a matrix which represents
              this colored version of img.
            - black pixels in img will remain black in the output image.  
            - The returned matrix will have the same dimensions as img.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RANDOMLY_COlOR_IMAGE_ABSTRACT_H__


