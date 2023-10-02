// Copyright (C) 2008  Davis E. King (davis@dlib.net), Nils Labugt
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PNG_IMPORT_ABSTRACT
#ifdef DLIB_PNG_IMPORT_ABSTRACT

#include "image_loader_abstract.h"
#include "../algs.h"
#include "../pixel.h"
#include "../image_processing/generic_image.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void load_png (
        image_type& image,
        const std::string& file_name
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - file_name points to a PNG file
        ensures
            - Reads and decodes the PNG file located at file_name
    !*/

    template <
        typename image_type,
        typeame Byte
        >
    void load_png (
        image_type& image,
        const Byte* image_buffer,
        size_t buffer_size
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - Byte is either char, uint8_t, int8_t, std::byte
            - image_buffer is a memory buffer containing a complete PNG encoded image
        ensures
            - Reads and ecodes the PNG file located in memory
    !*/

// ----------------------------------------------------------------------------------------

    template <
      class image_type
    >
    void load_png (
        image_type& img,
        std::istream& in
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - in is an input stream containing a complete PNG encoded image
        ensures
            - Reads and ecodes the PNG file located in stream
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PNG_IMPORT_ABSTRACT