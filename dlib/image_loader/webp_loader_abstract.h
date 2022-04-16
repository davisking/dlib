// Copyright (C) 2022  Davis E. King (davis@dlib.net), Martin Sandsmark, Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_WEBP_IMPORT_ABSTRACT
#ifdef DLIB_WEBP_IMPORT_ABSTRACT

#include "image_loader_abstract.h"
#include "../algs.h"
#include "../pixel.h"
#include "../dir_nav.h"
#include "../image_processing/generic_image.h"

namespace dlib
{

    class webp_loader : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a class capable of loading WEBP image files.
                Once an instance of it is created to contain a WEBP file from
                disk you can obtain the image stored in it via get_image().
        !*/

    public:

        webp_loader( 
            const char* filename 
        );
        /*!
            ensures
                - loads the WEBP file with the given file name into this object
            throws
                - std::bad_alloc
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given WEBP file.
        !*/

        webp_loader( 
            const std::string& filename 
        );
        /*!
            ensures
                - loads the WEBP file with the given file name into this object
            throws
                - std::bad_alloc
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given WEBP file.
        !*/

        webp_loader( 
            const dlib::file& f 
        );
        /*!
            ensures
                - loads the WEBP file with the given file name into this object
            throws
                - std::bad_alloc
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given WEBP file.
        !*/

        webp_loader( 
            const unsigned char* imgbuffer,
            size_t buffersize
        );
        /*!
            ensures
                - loads the WEBP from memory imgbuffer of size buffersize into this object
            throws
                - image_load_error
                  This exception is thrown if there is some error that prevents
                  us from loading the given WEBP buffer.
        !*/

        ~webp_loader(
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
                - loads the WEBP image stored in this object into img
        !*/

    };

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    void load_webp (
        image_type& image,
        const std::string& file_name
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - performs: webp_loader(file_name).get_image(image);
    !*/

    template <
        typename image_type
        >
    void load_webp (
        image_type& image,
        const unsigned char* imgbuff,
        size_t imgbuffsize
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - performs: webp_loader(imgbuff, imgbuffsize).get_image(image);
    !*/

    template <
        typename image_type
        >
    void load_webp (
        image_type& image,
        const char* imgbuff,
        size_t imgbuffsize
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - performs: webp_loader((unsigned char*)imgbuff, imgbuffsize).get_image(image);
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WEBP_IMPORT_ABSTRACT

