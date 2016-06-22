// Copyright (C) 2011  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_LOAd_IMAGE_ABSTRACT_
#ifdef DLIB_LOAd_IMAGE_ABSTRACT_

#include "../image_processing/generic_image.h"

namespace dlib
{
    template <typename image_type>
    void load_image (
        image_type& image,
        const std::string& file_name
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - This function loads an image from disk, in the indicated file file_name, and
              writes it to the indicated image object.
            - It is capable of reading the PNG, JPEG, BMP, GIF, and DNG image formats.  It
              is always capable of reading BMP and DNG images.  However, for PNG, JPEG, and
              GIF you must #define DLIB_PNG_SUPPORT, DLIB_JPEG_SUPPORT, and
              DLIB_GIF_SUPPORT respectively and link your program to libpng, libjpeg, and
              libgif respectively.
        throws
            - image_load_error
                This exception is thrown if there is some error that prevents
                us from loading the given image file.
    !*/

}

#endif // DLIB_LOAd_IMAGE_ABSTRACT_ 

 
