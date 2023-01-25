#// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_USE_FFMPEG

#include <fstream>
#include <dlib/dir_nav.h>
#include <dlib/config_reader.h>
#include <dlib/media.h>
#include <dlib/array2d.h>
#include "tester.h"

#ifndef DLIB_FFMPEG_DATA
static_assert(false, "Build is faulty. DLIB_VIDEOS_FILEPATH should be defined by cmake");
#endif

namespace  
{
    using namespace std;
    using namespace test;
    using namespace dlib;
    using namespace dlib::ffmpeg;
    
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

        decoder::args args;
        args.args_codec.codec_name      = codec;
        args.args_image.fmt             = AV_PIX_FMT_RGB24;
        args.args_audio.fmt             = AV_SAMPLE_FMT_S16;
        args.args_audio.channel_layout  = AV_CH_LAYOUT_MONO;

        decoder dec(args);
        DLIB_TEST(dec.is_open());
        DLIB_TEST(dec.get_codec_name() == codec);
        if (is_audio)
            DLIB_TEST(dec.is_audio_decoder());
        else
            DLIB_TEST(dec.is_image_decoder());

        array2d<rgb_pixel>          img;
        ffmpeg::audio<int16_t, 1>   audio;
        dlib::ffmpeg::frame         frame, frame_copy;
        int                         count{0};
        int                         nsamples{0};
        int                         iteration{0};
        decoder_status              status{DECODER_EAGAIN};

        ifstream fin{filepath, std::ios::binary};
        std::vector<char> buf(1024);

        const auto pull = [&]
        {
            while ((status = dec.read(frame)) == DECODER_FRAME_AVAILABLE)
            {
                if (is_audio)
                {
                    convert(frame, audio);
                    convert(audio, frame_copy);
                    DLIB_TEST(frame.is_audio());
                    DLIB_TEST(frame.sample_rate() == sample_rate);
                    DLIB_TEST(frame.samplefmt() == AV_SAMPLE_FMT_S16);
                    DLIB_TEST(frame_copy.is_audio());
                    DLIB_TEST(frame_copy.sample_rate() == frame.sample_rate());
                    DLIB_TEST(frame_copy.samplefmt() == frame.samplefmt());
                    DLIB_TEST(frame_copy.nsamples() == frame.nsamples());

                    DLIB_TEST(audio.sample_rate == sample_rate);
                    DLIB_TEST(audio.samples.size() == frame.nsamples());
                    nsamples += audio.samples.size();

                    DLIB_TEST(dec.sample_rate() == sample_rate);
                    DLIB_TEST(dec.sample_fmt() == AV_SAMPLE_FMT_S16);
                }
                else
                {
                    convert(frame, img);
                    convert(img, frame_copy);
                    DLIB_TEST(frame.is_image());
                    DLIB_TEST(frame.height() == height);
                    DLIB_TEST(frame.width() == width);
                    DLIB_TEST(frame.pixfmt() == AV_PIX_FMT_RGB24);
                    DLIB_TEST(frame_copy.is_image());
                    DLIB_TEST(frame_copy.height() == frame.height());
                    DLIB_TEST(frame_copy.width() == frame.width());
                    DLIB_TEST(frame_copy.pixfmt() == frame.pixfmt());

                    DLIB_TEST(img.nr() == height);
                    DLIB_TEST(img.nc() == width);
                    ++count;

                    DLIB_TEST(dec.height() == height);
                    DLIB_TEST(dec.width() == width);
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

            DLIB_TEST(dec.push_encoded((const uint8_t*)buf.data(), ret));
            pull();
        }

        dec.flush();
        pull();
        DLIB_TEST(count == nframes);
        DLIB_TEST(!dec.is_open());
    }

    void test_demuxer (
        const std::string& filepath,
        const dlib::config_reader& cfg
    )
    {
        const int nframes       = dlib::get_option(cfg, "nframes", 0);
        const int height        = dlib::get_option(cfg, "height", 0);
        const int width         = dlib::get_option(cfg, "width", 0);
        const int sample_rate   = dlib::get_option(cfg, "sample_rate", 0);
        const bool has_video    = height > 0 && width > 0 && nframes > 0;
        const bool has_audio    = sample_rate > 0;

        demuxer::args args;
        args.filepath           = filepath;
        args.image_options.fmt  = AV_PIX_FMT_RGB24;
        args.audio_options.fmt  = AV_SAMPLE_FMT_S16;

        demuxer cap(args);
        DLIB_TEST(cap.is_open());
        DLIB_TEST(cap.video_enabled() == has_video);
        DLIB_TEST(cap.audio_enabled() == has_audio);
        DLIB_TEST(cap.height() == height);
        DLIB_TEST(cap.width() == width);
        DLIB_TEST(cap.sample_rate() == sample_rate);

        if (has_video)
        {
            DLIB_TEST(cap.pixel_fmt() == AV_PIX_FMT_RGB24);
        }
        if (has_audio)
        {
            DLIB_TEST(cap.sample_fmt() == AV_SAMPLE_FMT_S16);
        }
        
        dlib::ffmpeg::frame frame, frame_copy;
        array2d<rgb_pixel>  img;
        audio<int16_t, 1>   audio1;
        audio<int16_t, 2>   audio2;
        int                 count{0};
        int                 nsamples{0};
        int                 iteration{0};

        while (cap.read(frame))
        {
            if (frame.is_image())
            {
                DLIB_TEST(frame.height() == height);
                DLIB_TEST(frame.width() == width);
                DLIB_TEST(frame.pixfmt() == AV_PIX_FMT_RGB24);
                convert(frame, img);

                DLIB_TEST(img.nr() == height);
                DLIB_TEST(img.nc() == width);
                convert(img, frame_copy);

                DLIB_TEST(frame_copy.height() == frame.height());
                DLIB_TEST(frame_copy.width() == frame.width());
                DLIB_TEST(frame_copy.pixfmt() == frame.pixfmt());
                
                ++count;
            }

            if (frame.is_audio())
            {
                DLIB_TEST(frame.sample_rate() == sample_rate);
                DLIB_TEST(frame.samplefmt() == AV_SAMPLE_FMT_S16);

                if (frame.nchannels() == 1)
                {
                    convert(frame, audio1);
                    convert(audio1, frame_copy);
                }
                else if (frame.nchannels() == 2)
                {
                    convert(frame, audio2);
                    convert(audio2, frame_copy);
                }

                DLIB_TEST(frame.sample_rate() == sample_rate);
                nsamples += frame.nsamples();
                DLIB_TEST(frame_copy.is_audio());
                DLIB_TEST(frame_copy.sample_rate() == frame.sample_rate());
                DLIB_TEST(frame_copy.samplefmt() == frame.samplefmt());
                DLIB_TEST(frame_copy.nsamples() == frame.nsamples());
            }

            ++iteration;
            if (iteration % 10 == 0)
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