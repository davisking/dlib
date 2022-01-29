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

        dlib::demuxer_ffmpeg cap(dlib::demuxer_ffmpeg::args{filepath});

        DLIB_TEST(cap.is_open());
        DLIB_TEST(cap.height() == height);
        DLIB_TEST(cap.width() == width);
        DLIB_TEST(cap.sample_rate() == rate);
        print_spinner();

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
            print_spinner();
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
        args.args_common.codec_name = codec;
        dlib::decoder_ffmpeg cap(args);

        DLIB_TEST(cap.is_open());
        print_spinner();

        /*! Just decode everything first !*/
        while (file)
        {
            file.read((char*)buffer.data(), buffer.size());
            int ret = file.gcount();

            if (ret > 0)
            {
                DLIB_TEST(cap.push_encoded(buffer.data(), ret));
            }
            print_spinner();
        }
        cap.flush();

        /*! Now read everything !*/
        DLIB_TEST(cap.is_open());
        print_spinner();

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
            print_spinner();
        }

        DLIB_TEST(!cap.is_open());
        DLIB_TEST(cap.height() == height);
        DLIB_TEST(cap.width() == width);
        DLIB_TEST(counter_frames == (nframes-1)); // I don't know why, but you always get 1 less frame
    }

    void test_demux_to_encode_to_decode(
        const std::string& filepath,
        AVCodecID audio_codec
    )
    {
        dlib::demuxer_ffmpeg::args args;
        args.filepath = filepath;
        dlib::demuxer_ffmpeg cap(args);

        DLIB_TEST(cap.is_open());
        print_spinner();

        dlib::encoder_ffmpeg enc_image, enc_audio;
        dlib::decoder_ffmpeg dec_image, dec_audio;

        if (cap.video_enabled())
        {
            {
                dlib::encoder_ffmpeg::args args2;
                args2.args_common.codec = AV_CODEC_ID_MPEG4;
                args2.args_image.h      = cap.height();
                args2.args_image.w      = cap.width();
                args2.args_image.fmt    = AV_PIX_FMT_YUV420P;

                enc_image = dlib::encoder_ffmpeg(args2, std::unique_ptr<std::ostream>(new std::stringstream()));
                DLIB_TEST(enc_image.is_open());
                print_spinner();
            }

            {
                dlib::decoder_ffmpeg::args args2;
                args2.args_common.codec = AV_CODEC_ID_MPEG4;
                args2.args_image.h   = cap.height();
                args2.args_image.w   = cap.width();
                args2.args_image.fmt = cap.fmt();

                dec_image = dlib::decoder_ffmpeg(args2);
                DLIB_TEST(dec_image.is_open());
                print_spinner();
            }
        }

        if (cap.audio_enabled())
        {
            {
                dlib::encoder_ffmpeg::args args2;
                args2.args_common.codec = audio_codec;
                args2.args_audio.sample_rate    = cap.sample_rate();
                args2.args_audio.channel_layout = cap.channel_layout();
                args2.args_audio.fmt            = cap.sample_fmt();

                enc_audio = dlib::encoder_ffmpeg(args2, std::unique_ptr<std::ostream>(new std::stringstream()));
                DLIB_TEST(enc_audio.is_open());
                print_spinner();
            }

            {
                dlib::decoder_ffmpeg::args args2;
                args2.args_common.codec = audio_codec;
                args2.args_audio.sample_rate    = cap.sample_rate();
                args2.args_audio.channel_layout = cap.channel_layout();
                args2.args_audio.fmt            = cap.sample_fmt();

                dec_audio = dlib::decoder_ffmpeg(args2);
                DLIB_TEST(dec_audio.is_open());
                print_spinner();
            }
        }

        type_safe_union<array2d<rgb_pixel>, audio_frame> frame;
        uint64_t timestamp_us = 0;
        int counter_images = 0;
        int counter_audio  = 0;

        while (cap.read(frame, timestamp_us))
        {
            frame.visit(overloaded(
                    [&](const array2d<rgb_pixel>& frame) {
                        DLIB_TEST(enc_image.push(frame, timestamp_us));
                        counter_images++;
                    },
                    [&](const audio_frame& frame) {
                        DLIB_TEST(enc_audio.push(frame, timestamp_us));
                        counter_audio += frame.samples.size();
                    }
            ));
            print_spinner();
        }

        enc_audio.flush();
        enc_image.flush();

        auto populate_encoder_and_decoder = [](encoder_ffmpeg& enc, decoder_ffmpeg& dec)
        {
            std::shared_ptr<std::ostream> encoded = enc.get_encoded_stream();
            const std::string encoded_str = dynamic_cast<std::stringstream&>(*encoded).str();
            DLIB_TEST(dec.push_encoded((const uint8_t*)encoded_str.c_str(), encoded_str.size()));
            dec.flush();
            print_spinner();
        };

        if (cap.video_enabled())
            populate_encoder_and_decoder(enc_image, dec_image);
        print_spinner();

        if (cap.audio_enabled())
            populate_encoder_and_decoder(enc_audio, dec_audio);
        print_spinner();

        frame.clear();

        auto run_decoder = [&cap](decoder_ffmpeg& dec, int counter_image, int counter_audio)
        {
            type_safe_union<array2d<rgb_pixel>, audio_frame> frame;
            uint64_t timestamp_us = 0;
            int counter_image_actual = 0;
            int counter_audio_actual = 0;

            while (dec.read(frame, timestamp_us) == decoder_ffmpeg::FRAME_AVAILABLE)
            {
                frame.visit(overloaded(
                        [&](const array2d<rgb_pixel>& frame) {
                            DLIB_TEST(frame.nc() == cap.width());
                            DLIB_TEST(frame.nr() == cap.height());
                            counter_image_actual++;
                        },
                        [&](const audio_frame& frame) {
                            DLIB_TEST(frame.sample_rate == cap.sample_rate());
                            counter_audio_actual += frame.samples.size();
                        }
                ));
                print_spinner();
            }

            DLIB_TEST(counter_image_actual == counter_image);
            DLIB_TEST(counter_audio_actual >= counter_audio - dec.sample_rate()); //within 1 second
            DLIB_TEST(counter_audio_actual <= counter_audio + dec.sample_rate()); //within 1 second
        };

        /*push all frames*/
        if (dec_image.is_open())
            run_decoder(dec_image, counter_images-1, 0); // I don't know why, but you always get 1 less frame
        print_spinner();
        if (dec_audio.is_open())
            run_decoder(dec_audio, 0, counter_audio);
        print_spinner();
    }

//    void test_decode_to_encode_to_decode(
//        const std::string& filepath,
//        const dlib::config_reader& cfg
//    )
//    {
//
//    }
//
//    void test_demux_to_mux_to_demux(
//        const std::string& filepath,
//        const dlib::config_reader& cfg
//    )
//    {
//
//    }
//
//    void test_decode_to_mux_to_demux(
//        const std::string& filepath,
//        const dlib::config_reader& cfg
//    )
//    {
//
//    }

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
                    test_demux_to_encode_to_decode(filepath, AV_CODEC_ID_MP2);
                    test_demux_to_encode_to_decode(filepath, AV_CODEC_ID_AC3);
                    test_demux_to_encode_to_decode(filepath, AV_CODEC_ID_PCM_S16LE);
//                    test_demux_to_mux_to_demux(filepath, sublock);
                }
                else if (type == 1)
                {
                    test_decode_video(filepath, sublock);
//                    test_decode_to_encode_to_decode(filepath, sublock);
//                    test_decode_to_mux_to_demux(filepath, sublock);
                }
            }
        }
    } a;
}

#endif
