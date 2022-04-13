// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_WEBP_SAVER_CPp_
#define DLIB_WEBP_SAVER_CPp_

// only do anything with this file if DLIB_WEBP_SUPPORT is defined
#ifdef DLIB_WEBP_SUPPORT

#include "save_webp.h"
#include "image_saver.h"
#include <sstream>

#include <webp/encode.h>

namespace dlib {


// ----------------------------------------------------------------------------------------

    void save_webp (
        const array2d<rgb_pixel>& img,
        const std::string& filename,
        float quality
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT(img.size() != 0,
            "\t save_webp()"
            << "\n\t You can't save an empty image as a WEBP."
            );
        DLIB_CASSERT(0 <= quality && quality <= 100,
            "\t save_webp()"
            << "\n\t Invalid quality value."
            << "\n\t quality: " << quality
            );

        auto fp = fopen(filename.c_str(), "wb");
        if (fp == NULL)
            throw image_save_error("Unable to open " + filename + " for writing.");

        auto data = reinterpret_cast<const uint8_t*>(image_data(img));
        std::cout << "nr: " << img.nr() << ", nc: " << img.nc() << ", stride: "<< width_step(img) << '\n';
        uint8_t* encoded;
        const auto size = WebPEncodeRGB(data, img.nc(), img.nr(), width_step(img), quality, &encoded);
        std::cout << "encoded size: " << size << '\n';
        if (size > 0)
        {
            if (fwrite(encoded, size, 1, fp) == 0)
                throw image_save_error("Unable to write image to " + filename + ".");
        }
        else
        {
            throw image_save_error("Error while encoding image to " + filename + ".");
        }
    }

// ----------------------------------------------------------------------------------------

    void save_webp (
        const array2d<bgr_pixel>& img,
        const std::string& filename,
        float quality
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT(img.size() != 0,
            "\t save_webp()"
            << "\n\t You can't save an empty image as a WEBP."
            );
        DLIB_CASSERT(0 <= quality && quality <= 100,
            "\t save_webp()"
            << "\n\t Invalid quality value."
            << "\n\t quality: " << quality
            );

        auto fp = fopen(filename.c_str(), "wb");
        if (fp == NULL)
            throw image_save_error("Unable to open " + filename + " for writing.");

        auto data = reinterpret_cast<const uint8_t*>(image_data(img));
        std::cout << "nr: " << img.nr() << ", nc: " << img.nc() << ", stride: "<< width_step(img) << '\n';
        uint8_t* encoded;
        const auto size = WebPEncodeBGR(data, img.nc(), img.nr(), width_step(img), quality, &encoded);
        std::cout << "encoded size: " << size << '\n';
        if (size > 0)
        {
            if (fwrite(encoded, size, 1, fp) == 0)
                throw image_save_error("Unable to write image to " + filename + ".");
        }
        else
        {
            throw image_save_error("Error while encoding image to " + filename + ".");
        }
    }


// ----------------------------------------------------------------------------------------

    void save_webp (
        const array2d<rgb_alpha_pixel>& img,
        const std::string& filename,
        float quality
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT(img.size() != 0,
            "\t save_webp()"
            << "\n\t You can't save an empty image as a WEBP."
            );
        DLIB_CASSERT(0 <= quality && quality <= 100,
            "\t save_webp()"
            << "\n\t Invalid quality value."
            << "\n\t quality: " << quality
            );

        auto fp = fopen(filename.c_str(), "wb");
        if (fp == NULL)
            throw image_save_error("Unable to open " + filename + " for writing.");

        auto data = reinterpret_cast<const uint8_t*>(image_data(img));
        std::cout << "nr: " << img.nr() << ", nc: " << img.nc() << ", stride: "<< width_step(img) << '\n';
        uint8_t* encoded;
        const auto size = WebPEncodeRGBA(data, img.nc(), img.nr(), width_step(img), quality, &encoded);
        std::cout << "encoded size: " << size << '\n';
        if (size > 0)
        {
            if(fwrite(encoded, size, 1, fp) == 0)
                throw image_save_error("Unable to write image to " + filename + ".");
        }
        else
        {
            throw image_save_error("Error while encoding image to " + filename + ".");
        }
    }

// ----------------------------------------------------------------------------------------

    void save_webp (
        const array2d<bgr_alpha_pixel>& img,
        const std::string& filename,
        float quality
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT(img.size() != 0,
            "\t save_webp()"
            << "\n\t You can't save an empty image as a WEBP."
            );
        DLIB_CASSERT(0 <= quality && quality <= 100,
            "\t save_webp()"
            << "\n\t Invalid quality value."
            << "\n\t quality: " << quality
            );

        auto fp = fopen(filename.c_str(), "wb");
        if (fp == NULL)
            throw image_save_error("Unable to open " + filename + " for writing.");

        auto data = reinterpret_cast<const uint8_t*>(image_data(img));
        std::cout << "nr: " << img.nr() << ", nc: " << img.nc() << ", stride: "<< width_step(img) << '\n';
        uint8_t* encoded;
        const auto size = WebPEncodeBGRA(data, img.nc(), img.nr(), width_step(img), quality, &encoded);
        std::cout << "encoded size: " << size << '\n';
        if (size > 0)
        {
            if (fwrite(encoded, size, 1, fp) == 0)
                throw image_save_error("Unable to write image to " + filename + ".");
        }
        else
        {
            throw image_save_error("Error while encoding image to " + filename + ".");
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WEBP_SUPPORT

#endif // DLIB_WEBP_SAVER_CPp_
