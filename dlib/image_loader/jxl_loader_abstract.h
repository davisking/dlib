// Copyright (C) 2024  Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_JXL_IMPORT_ABSTRACT
#ifdef DLIB_JXL_IMPORT_ABSTRACT

#include "image_loader_abstract.h"
#include "../algs.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "../image_processing/generic_image.h"

namespace dlib
{

    class jxl_loader : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a class capable of loading JPEG XL image files.
                Once an instance of it is created to contain a JPEG XL file from
                disk you can obtain the image stored in it via get_image().
        !*/

    public:

        jxl_loader( 
            const char* filename 
        );
        /*!
            ensures
                - loads the JPEG XL file with the given file name into this object
            throws
                - std::bad_alloc
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given JPEG XL file.
        !*/

        jxl_loader( 
            const std::string& filename 
        );
        /*!
            ensures
                - loads the JPEG XL file with the given file name into this object
            throws
                - std::bad_alloc
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given JPEG XL file.
        !*/

        jxl_loader( 
            const dlib::file& f 
        );
        /*!
            ensures
                - loads the JPEG XL file with the given file name into this object
            throws
                - std::bad_alloc
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given JPEG XL file.
        !*/

        jxl_loader( 
            const unsigned char* imgbuffer,
            size_t buffersize
        );
        /*!
            ensures
                - loads the JPEG XL from memory imgbuffer of size buffersize into
                  this object
            throws
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given JPEG XL buffer.
        !*/

        ~jxl_loader(
        );
        /*!
            ensures
                - all resources associated with *this has been released
        !*/

        template<
            typename image_type 
            >
        void get_image( 
            image_type& img
        ) const;
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
            ensures
                - loads the JPEG XL image stored in this object into img
        !*/

    };

// ----------------------------------------------------------------------------------------

        bool is_gray(
        ) const;
        /*!
            ensures
                - if (this object contains a grayscale image without an alpha channel)
                  then
                    - returns true
                - else
                    - returns false
        !*/

        bool is_graya(
        ) const;
        /*!
            ensures
                - if (this object contains a grayscale image with an alpha channel) then
                    - returns true
                - else
                    - returns false
        !*/

        bool is_rgb(
        ) const;
        /*!
            ensures
                - if (this object contains a 3 channel RGB image) then
                    - returns true
                - else
                    - returns false
        !*/

        bool is_rgba(
        ) const;
        /*!
            ensures
                - if (this object contains a 4 channel RGB alpha image) then
                    - returns true
                - else
                    - returns false
        !*/

        unsigned int bit_depth (
        ) const;
        /*!
            ensures
                - returns the number of bits per channel in the image contained by this
                  object.
        !*/

        long nr (
        ) const;
        /*!
            ensures
                - returns the number of rows (height) of the image contained by this
                  object.
        !*/

        long nc (
        ) const;
        /*!
            ensures
                - returns the number of colums (width) of the image contained by this
                  object.
        !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void load_jxl (
        image_type& image,
        const std::string& file_name
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - performs: jxl_loader(file_name).get_image(image);
    !*/

    template <
        typename image_type
        >
    void load_jxl (
        image_type& image,
        const unsigned char* imgbuff,
        size_t imgbuffsize
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - performs: jxl_loader(imgbuff, imgbuffsize).get_image(image);
    !*/

    template <
        typename image_type
        >
    void load_jxl (
        image_type& image,
        const char* imgbuff,
        size_t imgbuffsize
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - performs: jxl_loader((unsigned char*)imgbuff, imgbuffsize).get_image(image);
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_JXL_IMPORT_ABSTRACT

