// Copyright (C) 2024  Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SAVE_JXL_ABSTRACT_Hh_
#ifdef DLIB_SAVE_JXL_ABSTRACT_Hh_

#include "../image_processing/generic_image.h"
#include "../pixel.h"
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void save_jxl (
        const image_type& img,
        const std::string& filename,
        float quality = 90
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h or a matrix expression
            - image.size() != 0
            - quality >= 0
        ensures
            - writes the image to the file indicated by filename in the JPEG XL format.
            - image[0][0] will be in the upper left corner of the image.
            - image[image.nr()-1][image.nc()-1] will be in the lower right corner of the
              image.
            - This routine can save images containing any type of pixel.  However,
              save_jxl() can only natively store rgb_pixel, rgb_alpha_pixel and unsigned
              char pixel types.  All other pixel types will be converted into one of
              these types as appropriate before being saved to disk.
            - The quality value determines how lossy the compression is.  Larger quality
              values result in larger output images but the images will look better.
              Although it can range from 0 to 100, the recommended range is between
              68 and 96.  A value of 90 means visually lossless, while a value of 100
              means mathematically lossless.
        throws
            - image_save_error
                This exception is thrown if there is an error that prevents us from saving
                the image.
            - std::bad_alloc
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SAVE_JXL_ABSTRACT_Hh_

