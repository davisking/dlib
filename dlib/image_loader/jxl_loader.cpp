// Copyright (C) 2022  Davis E. King (davis@dlib.net), Martin Sandsmark, Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_JXL_LOADER_CPp_
#define DLIB_JXL_LOADER_CPp_

// only do anything with this file if DLIB_JPEGXL_SUPPORT is defined
#ifdef DLIB_JPEGXL_SUPPORT
#include "jxl_loader.h"
#include <jxl/decode_cxx.h>
#include <jxl/resizable_parallel_runner_cxx.h>
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
    jxl_loader(const char* filename) : height(0), width(0)
    {
        data = load_contents(filename);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    jxl_loader::
    jxl_loader(const std::string& filename) : height(0), width(0)
    {
        data = load_contents(filename);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    jxl_loader::
    jxl_loader(const dlib::file& f) : height(0), width(0)
    {
        data = load_contents(f.full_name());
        get_info();
    }

// ----------------------------------------------------------------------------------------

    jxl_loader::
    jxl_loader(const unsigned char* imgbuffer, size_t imgbuffersize) : height(0), width(0)
    {
        data.resize(imgbuffersize);
        memcpy(data.data(), imgbuffer, imgbuffersize);
        get_info();
    }

// ----------------------------------------------------------------------------------------

    void jxl_loader::get_info()
    {
        JxlSignature signature = JxlSignatureCheck(data.data(), data.size());
        if (signature != JXL_SIG_CODESTREAM && signature != JXL_SIG_CONTAINER)
        {
            throw image_load_error("jxl_loader: JxlSignatureCheck failed");
        }

        auto dec = JxlDecoderMake(nullptr);
        if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO))
        {
            throw image_load_error("jxl_loader: JxlDecoderSubscribeEvents failed");
        }

        JxlDecoderSetInput(dec.get(), data.data(), data.size());
        JxlDecoderCloseInput(dec.get());
        if (JXL_DEC_BASIC_INFO != JxlDecoderProcessInput(dec.get())) {
            throw image_load_error("jxl_loader: JxlDecoderProcessInput failed");
        }

        JxlBasicInfo basic_info;
        if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &basic_info))
        {
            throw image_load_error("jxl_loader: JxlDecoderGetBasicInfo failed");
        }
        width = basic_info.xsize;
        height = basic_info.ysize;
    }
// ----------------------------------------------------------------------------------------

    void jxl_loader::decode(unsigned char* out, const size_t out_size, const long num_channels) const
    {
        auto runner = JxlResizableParallelRunnerMake(nullptr);
        auto dec = JxlDecoderMake(nullptr);
        if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_FULL_IMAGE))
        {
            throw image_load_error("jxl_loader: JxlDecoderSubscribeEvents failed");
        }

        if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner, runner.get()))
        {
            throw image_load_error("jxl_loader: JxlDecoderSetParallelRunner failed");
        }

        if (JXL_DEC_SUCCESS != JxlDecoderSetInput(dec.get(), data.data(), data.size()))
        {
            throw image_load_error("jxl_loader: JxlDecoderSetInput failed");
        }
        JxlDecoderCloseInput(dec.get());

        JxlPixelFormat format = {
            .num_channels = static_cast<uint32_t>(num_channels),
            .data_type = JXL_TYPE_UINT8,
            .endianness = JXL_NATIVE_ENDIAN,
            .align=0
        };
        for (;;)
        {
            JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
            if (status == JXL_DEC_ERROR)
            {
                throw image_load_error("jxl_loader: JxlDecoderProcessInput failed");
            }
            else if (status == JXL_DEC_NEED_MORE_INPUT)
            {
                throw image_load_error("jxl_loader: Error, expected more input");
            }
            else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER)
            {
                JxlResizableParallelRunnerSetThreads(runner.get(), JxlResizableParallelRunnerSuggestThreads(width, height));
                size_t buffer_size;
                if (JXL_DEC_SUCCESS != JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size))
                {
                    throw image_load_error("jxl_loader: JxlDecoderImageOutBufferSize failed");
                }
                if (buffer_size != width * height * num_channels)
                {
                    throw image_load_error("jxl_loader: invalid output buffer size");
                }
                if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec.get(), &format, out, out_size))
                {
                    throw image_load_error("jxl_loader: JxlDecoderSetImageOutBuffer failed");
                }
            }
            else if (status == JXL_DEC_FULL_IMAGE)
            {
                // If the image is an animation, more full frames may be decoded.
                // This loader only decodes the first one.
                return;
            }
            else if (status == JXL_DEC_SUCCESS)
            {
                return;
            }
            else
            {
                throw image_load_error("jxl_loder: Unknown decoder status");
            }
        }
    }
}

#endif // DLIB_JPEG_SUPPORT

#endif // DLIB_JXL_LOADER_CPp_

