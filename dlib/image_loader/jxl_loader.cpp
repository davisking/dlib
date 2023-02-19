// Copyright (C) 2022  Davis E. King (davis@dlib.net), Martin Sandsmark, Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_JXL_LOADER_CPp_
#define DLIB_JXL_LOADER_CPp_

// only do anything with this file if DLIB_JPEGXL_SUPPORT is defined
#ifdef DLIB_JPEGXL_SUPPORT

#include "jxl_loader.h"

#include <jxl/decode.h>
#include <fstream>

namespace dlib
{

    static std::vector<unsigned char> load_contents(const std::string& filename)
    {
        std::ifstream stream(filename, std::ios::binary);
        stream.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
        stream.seekg(0, std::ios_base::end);
        std::vector<unsigned char> buffer(stream.tellg());
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        return buffer;
    }

// ----------------------------------------------------------------------------------------

    jxl_loader::
    jxl_loader(const char* filename) : height_(0), width_(0)
    {
        data_ = load_contents(filename);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    jxl_loader::
    jxl_loader(const std::string& filename) : height_(0), width_(0)
    {
        data_ = load_contents(filename);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    jxl_loader::
    jxl_loader(const dlib::file& f) : height_(0), width_(0)
    {
        data_ = load_contents(f.full_name());
        get_info();
    }

// ----------------------------------------------------------------------------------------

    jxl_loader::
    jxl_loader(const unsigned char* imgbuffer, size_t imgbuffersize) : height_(0), width_(0)
    {
        data_.resize(imgbuffersize);
        memcpy(data_.data(), imgbuffer, imgbuffersize);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    void jxl_loader::get_info()
    {
        JxlSignature signature = JxlSignatureCheck(data_.data(), data_.size());
        if (signature != JXL_SIG_CODESTREAM && signature != JXL_SIG_CONTAINER)
        {
            throw image_load_error("jxl_loader: Invalid header");
        }
    }

/* ----------------------------------------------------------------------------------------

    void jxl_loader::read_argb(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeARGBInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("jxl_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void jxl_loader::read_rgba(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeRGBAInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("jxl_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void jxl_loader::read_bgra(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeBGRAInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("jxl_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void jxl_loader::read_rgb(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeRGBInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("jxl_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void jxl_loader::read_bgr(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeBGRInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("jxl_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------
*/

}

#endif // DLIB_JPEG_SUPPORT

#endif // DLIB_JXL_LOADER_CPp_

