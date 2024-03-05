// Copyright (C) 2024  Davis E. King (davis@dlib.net), Adrià Arrufat
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_JXL_SAVER_CPp_
#define DLIB_JXL_SAVER_CPp_

// only do anything with this file if DLIB_JXL_SUPPORT is defined
#ifdef DLIB_JXL_SUPPORT

#include "save_jxl.h"
#include "image_saver.h"
#include <sstream>
#include <jxl/encode_cxx.h>
#include <jxl/resizable_parallel_runner_cxx.h>

namespace dlib {

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        void impl_save_jxl (
            const std::string& filename,
            const uint8_t* pixels,
            const uint32_t width,
            const uint32_t height,
            const uint32_t num_channels,
            const float quality
        )
        {
            std::ofstream fout(filename, std::ios::binary);
            if (!fout.good())
            {
                throw image_save_error("Unable to open " + filename + " for writing.");
            }

            auto enc = JxlEncoderMake(nullptr);
            auto runner = JxlResizableParallelRunnerMake(nullptr);
            JxlResizableParallelRunnerSetThreads(runner.get(), JxlResizableParallelRunnerSuggestThreads(width, height));

            if (JXL_ENC_SUCCESS != JxlEncoderSetParallelRunner(enc.get(), JxlResizableParallelRunner, runner.get()))
            {
                throw image_save_error("jxl_saver: JxlResizableParallelRunner failed");
            }

            JxlPixelFormat pixel_format{
                .num_channels = num_channels,
                .data_type = JXL_TYPE_UINT8,
                .endianness = JXL_NATIVE_ENDIAN,
                .align = 0
            };
            JxlBasicInfo basic_info;
            JxlEncoderInitBasicInfo(&basic_info);
            basic_info.xsize = width;
            basic_info.ysize = height;
            basic_info.bits_per_sample = 8;
            basic_info.uses_original_profile = quality == 100;
            switch (num_channels)
            {
            case 1:
                basic_info.num_color_channels = 1;
                basic_info.num_extra_channels = 0;
                basic_info.alpha_bits = 0;
                basic_info.alpha_exponent_bits = 0;
                break;
            case 3:
                basic_info.num_color_channels = 3;
                basic_info.num_extra_channels = 0;
                basic_info.alpha_bits = 0;
                basic_info.alpha_exponent_bits = 0;
                break;
            case 4:
                basic_info.num_color_channels = 3;
                basic_info.num_extra_channels = 1;
                basic_info.alpha_bits = basic_info.bits_per_sample;
                basic_info.alpha_exponent_bits = 0;
                break;
            default:
                throw ("jxl_saver: unsupported number of channels");
            }

            if (JXL_ENC_SUCCESS != JxlEncoderSetBasicInfo(enc.get(), &basic_info))
            {
                throw image_save_error("jxl_saver: JxlEncoderSetBasicInfo failed");
            }

            JxlColorEncoding color_encoding = {};
            JxlColorEncodingSetToSRGB(&color_encoding, /* is_gray = */ num_channels < 3);
            if (JXL_ENC_SUCCESS != JxlEncoderSetColorEncoding(enc.get(), &color_encoding))
            {
                throw image_save_error("jxl_saver: JxlEncoderSetColorEncoding failed");
            }

            JxlEncoderFrameSettings* frame_settings = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
            JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_DECODING_SPEED, 0);

            const float distance = JxlEncoderDistanceFromQuality(quality);
            if (JXL_ENC_SUCCESS != JxlEncoderSetFrameDistance(frame_settings, distance))
            {
                throw image_save_error("jxl_saver: JxlEncoderSetFrameDistance failed");
            }
            if (basic_info.alpha_bits > 0)
            {
                if (JXL_ENC_SUCCESS != JxlEncoderSetExtraChannelDistance(frame_settings, 0, distance))
                {
                    throw image_save_error("jxl_saver: JxlEncoderSetExtraChannelDistance failed");
                }
            }

            // explictly enable lossless mode
            if (distance == 0)
            {
                if (JXL_ENC_SUCCESS != JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE))
                {
                    throw image_save_error("jxl_saver: JxlEncoderSetFrameLossless failed");
                }
            }

            void* pixels_data = reinterpret_cast<void*>(const_cast<uint8_t*>(pixels));
            const size_t pixels_size = width * height * num_channels;
            if (JXL_ENC_SUCCESS != JxlEncoderAddImageFrame(frame_settings, &pixel_format, pixels_data, pixels_size))
            {
                throw image_save_error("jxl_saver: JxlEncoderAddImageFrame failed");
            }
            JxlEncoderCloseInput(enc.get());

            std::vector<uint8_t> compressed;
            compressed.resize(64);
            uint8_t* next_out = compressed.data();
            size_t avail_out = compressed.size() - (next_out - compressed.data());
            JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
            while (process_result == JXL_ENC_NEED_MORE_OUTPUT)
            {
                process_result = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
                if (process_result == JXL_ENC_NEED_MORE_OUTPUT)
                {
                    size_t offset = next_out - compressed.data();
                    compressed.resize(compressed.size() * 2);
                    next_out = compressed.data() + offset;
                    avail_out = compressed.size() - offset;
                }
            }
            compressed.resize(next_out - compressed.data());
            if (JXL_ENC_SUCCESS != process_result)
            {
                throw image_save_error("jxl_saver: JxlEncoderProcessOutput failed");
            }
            fout.write(reinterpret_cast<char*>(compressed.data()), compressed.size());
            if (!fout.good())
            {
                throw image_save_error("Error while writing JPEG XL image to " + filename + ".");
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_JXL_SUPPORT

#endif // DLIB_JXL_SAVER_CPp_

