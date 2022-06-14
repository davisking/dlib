// Copyright (C) 2022  Davis E. King (davis@dlib.net), Adrià Arrufat
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

    namespace impl
    {
        void impl_save_webp (
            const std::string& filename,
            const uint8_t* data,
            const int width,
            const int height,
            const int stride,
            const float quality,
            const webp_type type
        )
        {
            if (width > WEBP_MAX_DIMENSION || height > WEBP_MAX_DIMENSION)
                throw image_save_error("Error while encoding " + filename + ". Bad picture dimensions: "
                + std::to_string(width) + "x" + std::to_string(height)
                + ". Maximum WebP width and height allowed is "
                + std::to_string(WEBP_MAX_DIMENSION) + " pixels");

            std::ofstream fout(filename, std::ios::binary);
            if (!fout.good())
                throw image_save_error("Unable to open " + filename + " for writing.");

            uint8_t* output;
            size_t output_size = 0;
            switch (type)
            {
            case webp_type::rgb:
                if (quality > 100)
                    output_size = WebPEncodeLosslessRGB(data, width, height, stride, &output);
                else
                    output_size = WebPEncodeRGB(data, width, height, stride, quality, &output);
                break;
            case webp_type::rgba:
                if (quality > 100)
                    output_size = WebPEncodeLosslessRGBA(data, width, height, stride, &output);
                else
                    output_size = WebPEncodeRGBA(data, width, height, stride, quality, &output);
                break;
            case webp_type::bgr:
                if (quality > 100)
                    output_size = WebPEncodeLosslessBGR(data, width, height, stride, &output);
                else
                    output_size = WebPEncodeBGR(data, width, height, stride, quality, &output);
                break;
            case webp_type::bgra:
                if (quality > 100)
                    output_size = WebPEncodeLosslessBGRA(data, width, height, stride, &output);
                else
                    output_size = WebPEncodeBGRA(data, width, height, stride, quality, &output);
                break;
            default:
                throw image_save_error("Invalid WebP color type");
            }

            if (output_size > 0)
            {
                fout.write(reinterpret_cast<char*>(output), output_size);
                if (!fout.good())
                {
                    WebPFree(output);
                    throw image_save_error("Error while writing WebP image to " + filename + ".");
                }
            }
            else
            {
                throw image_save_error("Error while encoding WebP image to " + filename + ".");
            }
            WebPFree(output);
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WEBP_SUPPORT

#endif // DLIB_WEBP_SAVER_CPp_

