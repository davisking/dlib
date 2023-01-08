#// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_USE_FFMPEG

#include <fstream>
#include <dlib/dir_nav.h>
#include <dlib/config_reader.h>
#include <dlib/media.h>
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

    void test_decoder(
        const std::string& filepath,
        const dlib::config_reader& cfg
    )
    {
        const std::string codec = dlib::get_option(cfg, "codec", "");
        const int nframes       = dlib::get_option(cfg, "nframes", 0);
        const int height        = dlib::get_option(cfg, "height", 0);
        const int width         = dlib::get_option(cfg, "width", 0);
        const int sample_rate   = dlib::get_option(cfg, "sample_rate", 0);
        const bool is_audio     = sample_rate > 0;

        ffmpeg_decoder::args args;
        args.args_codec.codec_name  = codec;
        args.args_image.fmt         = AV_PIX_FMT_RGB24;
        args.args_audio.fmt         = AV_SAMPLE_FMT_S16;

        dlib::ffmpeg_decoder decoder(args);
        DLIB_TEST(decoder.is_open());
        DLIB_TEST(decoder.get_codec_name() == codec);
        if (is_audio)
            DLIB_TEST(decoder.is_audio_decoder());
        else
            DLIB_TEST(decoder.is_image_decoder());

        type_safe_union<array2d<rgb_pixel>, audio_frame> obj;
        dlib::Frame             frame;
        int                     count{0};
        int                     nsamples{0};
        int                     iteration{0};
        dlib::decoder_status    status{DECODER_EAGAIN};

        ifstream fin{filepath, std::ios::binary};
        std::vector<char> buf(1024);

        const auto pull = [&]
        {
            while ((status = decoder.read(frame)) == DECODER_FRAME_AVAILABLE)
            {
                if (is_audio)
                {
                    DLIB_TEST(frame.is_audio());
                    DLIB_TEST(frame.sample_rate() == sample_rate);
                    DLIB_TEST(frame.samplefmt() == AV_SAMPLE_FMT_S16);
                }
                else
                {
                    DLIB_TEST(frame.is_image());
                    DLIB_TEST(frame.height() == height);
                    DLIB_TEST(frame.width() == width);
                    DLIB_TEST(frame.pixfmt() == AV_PIX_FMT_RGB24);
                }
                
                convert(frame, obj);

                if (is_audio)
                {
                    DLIB_TEST(obj.contains<audio_frame>());
                    DLIB_TEST(obj.get<audio_frame>().sample_rate == sample_rate);
                    nsamples += obj.get<audio_frame>().samples.size();
                }
                else
                {
                    DLIB_TEST(obj.contains<array2d<rgb_pixel>>());
                    DLIB_TEST(obj.get<array2d<rgb_pixel>>().nr() == height);
                    DLIB_TEST(obj.get<array2d<rgb_pixel>>().nc() == width);
                    ++count;
                    DLIB_TEST(decoder.height() == height);
                    DLIB_TEST(decoder.width() == width);
                }

                ++iteration;
                
                if (iteration % 10 == 0)
                    print_spinner();
            }
        };

        while (fin && status != DECODER_CLOSED)
        {
            fin.read(buf.data(), buf.size());
            size_t ret = fin.gcount();

            DLIB_TEST(decoder.push_encoded((const uint8_t*)buf.data(), ret));
            pull();
        }

        decoder.flush();
        pull();
        DLIB_TEST(count == nframes);
        DLIB_TEST(!decoder.is_open());
    }

    void test_demuxer (
        const std::string& filepath,
        const dlib::config_reader& cfg
    )
    {
        const int nframes   = dlib::get_option(cfg, "nframes", 0);
        const int height    = dlib::get_option(cfg, "height", 0);
        const int width     = dlib::get_option(cfg, "width", 0);

        dlib::ffmpeg_demuxer::args args;
        args.filepath           = filepath;
        args.image_options.fmt  = AV_PIX_FMT_RGB24;

        dlib::ffmpeg_demuxer cap(args);
        DLIB_TEST(cap.is_open());
        DLIB_TEST(cap.video_enabled());
        DLIB_TEST(cap.height() == height);
        DLIB_TEST(cap.width() == width);
        DLIB_TEST(cap.pixel_fmt() == AV_PIX_FMT_RGB24);

        type_safe_union<array2d<rgb_pixel>, audio_frame> obj;
        dlib::Frame frame;
        int         count{0};

        while (cap.read(frame))
        {
            DLIB_TEST(frame.is_image());
            DLIB_TEST(frame.height() == height);
            DLIB_TEST(frame.width() == width);
            DLIB_TEST(frame.pixfmt() == AV_PIX_FMT_RGB24);
            convert(frame, obj);
            DLIB_TEST(obj.contains<array2d<rgb_pixel>>());
            DLIB_TEST(obj.get<array2d<rgb_pixel>>().nr() == height);
            DLIB_TEST(obj.get<array2d<rgb_pixel>>().nc() == width);
            ++count;

            if (count % 10 == 0)
                print_spinner();
        }

        DLIB_TEST(count == nframes);
        DLIB_TEST(!cap.is_open());
    }

    class video_tester : public tester
    {
    public:
        video_tester (
        ) :
            tester ("test_ffmpeg",
                    "Runs tests on video IO.")
        {}

        void perform_test (
        )
        {
            dlib::file f(DLIB_FFMPEG_DATA);
            dlib::config_reader cfg(f.full_name());

            {
                const auto& video_raw_block = cfg.block("decoding");
                std::vector<string> blocks;
                video_raw_block.get_blocks(blocks);

                for (const auto& block : blocks)
                {
                    const auto& sublock = video_raw_block.block(block);
                    const std::string filepath = get_parent_directory(f).full_name() + "/" + sublock["file"];

                    test_decoder(filepath, sublock);
                }
            }

            {
                const auto& video_file_block = cfg.block("demuxing");
                std::vector<string> blocks;
                video_file_block.get_blocks(blocks);

                for (const auto& block : blocks)
                {
                    const auto& sublock = video_file_block.block(block);
                    const std::string filepath = get_parent_directory(f).full_name() + "/" + sublock["file"];

                    test_demuxer(filepath, sublock);
                }
            }
        }
    } a;
}

#endif