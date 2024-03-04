// Copyright (C) 2022  Davis E. King (davis@dlib.net), Martin Sandsmark, Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_WEBP_LOADER_CPp_
#define DLIB_WEBP_LOADER_CPp_

// only do anything with this file if DLIB_WEBP_SUPPORT is defined
#ifdef DLIB_WEBP_SUPPORT

#include "webp_loader.h"

#include <webp/decode.h>
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

    webp_loader::
    webp_loader(const char* filename) : height_(0), width_(0)
    {
        data_ = load_contents(filename);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    webp_loader::
    webp_loader(const std::string& filename) : height_(0), width_(0)
    {
        data_ = load_contents(filename);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    webp_loader::
    webp_loader(const dlib::file& f) : height_(0), width_(0)
    {
        data_ = load_contents(f.full_name());
        get_info();
    }

// ----------------------------------------------------------------------------------------

    webp_loader::
    webp_loader(const unsigned char* imgbuffer, size_t imgbuffersize) : height_(0), width_(0)
    {
        data_.resize(imgbuffersize);
        memcpy(data_.data(), imgbuffer, imgbuffersize);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    long webp_loader::nr() const { return height_; }
    long webp_loader::nc() const { return width_; }

// ----------------------------------------------------------------------------------------

    void webp_loader::get_info()
    {
        if (!WebPGetInfo(data_.data(), data_.size(), &width_, &height_))
        {
            throw image_load_error("webp_loader: Invalid header");
        }
    }

// ----------------------------------------------------------------------------------------

    void webp_loader::read_argb(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeARGBInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("webp_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void webp_loader::read_rgba(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeRGBAInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("webp_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void webp_loader::read_bgra(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeBGRAInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("webp_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void webp_loader::read_rgb(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeRGBInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("webp_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

    void webp_loader::read_bgr(unsigned char *out, const size_t out_size, const int out_stride) const
    {
        if (!WebPDecodeBGRInto(data_.data(), data_.size(), out, out_size, out_stride))
        {
            throw image_load_error("webp_loader: decoding failed");
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_WEBP_SUPPORT

#endif // DLIB_WEBP_LOADER_CPp_

