// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_USE_FFMPEG

#include <dlib/dir_nav.h>
#include <dlib/config_reader.h>
#include <dlib/video_io.h>
#include "tester.h"

#ifndef DLIB_FFMPEG_DATA
static_assert(false, "Build is faulty. DLIB_VIDEOS_FILEPATH should be defined by cmake");
#endif

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.video");
        
    void test_demux_video(
        const std::string& filepath,
        const dlib::config_reader& cfg
    )
    {
        const int nframes   = dlib::get_option(cfg, "nframes", 0);
        const int height    = dlib::get_option(cfg, "height", 0);
        const int width     = dlib::get_option(cfg, "width", 0);
        const int rate      = dlib::get_option(cfg, "sample_rate", 0);

        dlib::demuxer_ffmpeg::args args;
        args.filepath = filepath;
        dlib::demuxer_ffmpeg cap(args);

        DLIB_TEST(cap.is_open());
        DLIB_TEST(cap.height() == height);
        DLIB_TEST(cap.width() == width);
        DLIB_TEST(cap.sample_rate() == rate);

        int counter_frames = 0;
        int counter_samples = 0;

        type_safe_union<array2d<rgb_pixel>, audio_frame> frame;
        uint64_t timestamp_us = 0;

        while (cap.read(frame, timestamp_us))
        {
            frame.visit(overloaded(
                [&](const array2d<rgb_pixel>& frame) {
                    DLIB_TEST(frame.nc() == width);
                    DLIB_TEST(frame.nr() == height);
                    counter_frames++;
                },
                [&](const audio_frame& frame) {
                    DLIB_TEST(frame.sample_rate == rate);
                    counter_samples += frame.samples.size();
                }
            ));
        }

        DLIB_TEST(counter_frames == nframes);
        DLIB_TEST(counter_samples >= cap.estimated_total_samples() - cap.sample_rate()); //within 1 second
        DLIB_TEST(counter_samples <= cap.estimated_total_samples() + cap.sample_rate()); //within 1 second
    }

    void test_decode_video(
        const std::string& filepath,
        const dlib::config_reader& cfg
    )
    {
        const int nframes       = dlib::get_option(cfg, "nframes", 0);
        const int height        = dlib::get_option(cfg, "height", 0);
        const int width         = dlib::get_option(cfg, "width", 0);
        const int rate          = dlib::get_option(cfg, "sample_rate", 0);
        const std::string codec = cfg["codec"];

        std::ifstream file(filepath, std::ios::binary);
        std::vector<uint8_t> buffer(1024);

        dlib::decoder_ffmpeg::args args;
        args.base.codec_name = codec;
        args.options = dlib::decoder_ffmpeg::args::image_args{};
        dlib::decoder_ffmpeg cap(args);

        DLIB_TEST(cap.is_open());

        /*! Just decode everything first !*/
        while (file)
        {
            file.read((char*)buffer.data(), buffer.size());
            int ret = file.gcount();

            if (ret > 0)
                cap.push_encoded(buffer.data(), ret);
        }
        cap.flush();

        /*! Now read everything !*/
        DLIB_TEST(cap.is_open());

        int counter_frames = 0;
        int counter_samples = 0;

        type_safe_union<array2d<rgb_pixel>, audio_frame> frame;
        uint64_t timestamp_us = 0;

        while (cap.read(frame, timestamp_us) == decoder_ffmpeg::FRAME_AVAILABLE)
        {
            frame.visit(overloaded(
                    [&](const array2d<rgb_pixel>& frame) {
                        DLIB_TEST(frame.nc() == width);
                        DLIB_TEST(frame.nr() == height);
                        counter_frames++;
                    },
                    [&](const audio_frame& frame) {
                        DLIB_TEST(frame.sample_rate == rate);
                        counter_samples += frame.samples.size();
                    }
            ));
        }

        DLIB_TEST(!cap.is_open());
        DLIB_TEST(cap.height() == height);
        DLIB_TEST(cap.width() == width);
        DLIB_TEST(counter_frames == nframes);
    }

    class video_tester : public tester
    {
    public:
        video_tester (
        ) :
            tester ("test_video",
                    "Runs tests on video IO.")
        {}

        void perform_test (
        )
        {
            dlib::file f(DLIB_FFMPEG_DATA);
            dlib::config_reader cfg(f.full_name());
            std::vector<string> blocks;
            cfg.get_blocks(blocks);

            for (const auto& block : blocks)
            {
                const auto& sublock = cfg.block(block);
                const std::string filepath  = get_parent_directory(f).full_name() + "/" + sublock["file"];
                const int type              = dlib::get_option(sublock, "type", 0);

                if (type == 0)
                {
                    test_demux_video(filepath, sublock);
                }
                else if (type == 1)
                {
                    test_decode_video(filepath, sublock);
                }
            }
        }
    } a;
}

#endif
