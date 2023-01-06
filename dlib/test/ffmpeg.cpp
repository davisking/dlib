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
        const std::string codec     = dlib::get_option(cfg, "codec", "");
        const int nframes           = dlib::get_option(cfg, "nframes", 0);
        const int height            = dlib::get_option(cfg, "height", 0);
        const int width             = dlib::get_option(cfg, "width", 0);

        ffmpeg_decoder::args args;
        args.args_codec.codec_name = codec;
        dlib::ffmpeg_decoder decoder(args);
        DLIB_TEST(decoder.is_open());
        DLIB_TEST(decoder.is_image_decoder());
        DLIB_TEST(decoder.get_codec_name() == codec);

        dlib::Frame             frame;
        int                     count{0};
        dlib::decoder_status    status{DECODER_EAGAIN};

        ifstream fin{filepath, std::ios::binary};
        std::vector<char> buf(1024);

        auto pull = [&]
        {
            while ((status = decoder.read(frame)) == DECODER_FRAME_AVAILABLE)
            {
                DLIB_TEST(frame.height() == height);
                DLIB_TEST(frame.width() == width);
                DLIB_TEST(frame.is_image());
                ++count;
                DLIB_TEST(decoder.height() == height);
                DLIB_TEST(decoder.width() == width);
            }
        };

        while (fin && status != DECODER_CLOSED)
        {
            fin.read(buf.data(), buf.size());
            size_t ret = fin.gcount();
            print_spinner();

            DLIB_TEST(decoder.push_encoded((const uint8_t*)buf.data(), ret));
            pull();
        }

        decoder.flush();
        pull();
        DLIB_TEST(count == nframes);
        DLIB_TEST(!decoder.is_open());
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
                const auto& video_raw_block = cfg.block("video_raw");
                std::vector<string> blocks;
                video_raw_block.get_blocks(blocks);

                for (const auto& block : blocks)
                {
                    const auto& sublock = video_raw_block.block(block);
                    const std::string filepath = get_parent_directory(f).full_name() + "/" + sublock["file"];

                    test_decoder(filepath, sublock);
                }
            }
        }
    } a;
}

#endif