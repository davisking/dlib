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

    void test_demux_to_mux_to_demux(
        const dlib::config_reader& cfg,
        const std::string& filepath,
        AVCodecID image_codec,
        AVCodecID audio_codec
    )
    {
        const std::string muxed_file = "dummy.avi";
        const int nframes   = dlib::get_option(cfg, "nframes", 0);
        const int height    = dlib::get_option(cfg, "height", 0);
        const int width     = dlib::get_option(cfg, "width", 0);
        const int rate      = dlib::get_option(cfg, "sample_rate", 0);

        dlib::demuxer_ffmpeg cap(filepath);
        DLIB_TEST(cap.is_open());
        print_spinner();

        int encoded_sample_rate = 0;

        {
            dlib::muxer_ffmpeg::args args;
            args.filepath       = muxed_file;
            args.enable_image   = cap.video_enabled();
            args.enable_audio   = cap.audio_enabled();

            if (cap.video_enabled())
            {
                args.args_image.codec   = image_codec;
                args.args_image.h       = cap.height();
                args.args_image.w       = cap.width();
                args.args_image.fmt     = AV_PIX_FMT_YUV420P;
            }

            if (cap.audio_enabled())
            {
                args.args_audio.codec           = audio_codec;
                args.args_audio.sample_rate     = cap.sample_rate();
                args.args_audio.channel_layout  = cap.channel_layout();
                args.args_audio.fmt             = cap.sample_fmt();
            }

            dlib::muxer_ffmpeg mux(args);
            DLIB_TEST(mux.is_open());
            DLIB_TEST(mux.video_enabled() == cap.video_enabled());
            DLIB_TEST(mux.audio_enabled() == cap.audio_enabled());
            encoded_sample_rate = mux.sample_rate();

            if (mux.video_enabled())
            {
                DLIB_TEST(mux.height() == cap.height());
                DLIB_TEST(mux.width() == cap.width());
            }

            print_spinner();

            type_safe_union<array2d<rgb_pixel>, audio_frame> frame;
            std::chrono::system_clock::time_point timestamp{};
            int counter_images  = 0;
            int counter_samples = 0;

            while (cap.read(frame, timestamp)) {
                visit(overloaded(
                        [&](const array2d<rgb_pixel> &frame) {
                            DLIB_TEST(frame.nc() == width);
                            DLIB_TEST(frame.nr() == height);
                            DLIB_TEST(mux.push(frame, timestamp));
                            ++counter_images;
                        },
                        [&](const audio_frame &frame) {
                            DLIB_TEST(frame.sample_rate == rate);
                            DLIB_TEST(mux.push(frame, timestamp));
                            counter_samples += frame.samples.size();
                        }
                ), frame);
                print_spinner();
            }

            DLIB_TEST(counter_images == nframes);
            DLIB_TEST(counter_samples >= cap.estimated_total_samples() - cap.sample_rate()); //within 1 second
            DLIB_TEST(counter_samples <= cap.estimated_total_samples() + cap.sample_rate()); //within 1 second
        }

        dlib::demuxer_ffmpeg cap2(muxed_file);
        DLIB_TEST(cap2.is_open());
        DLIB_TEST(cap2.height() == height);
        DLIB_TEST(cap2.width() == width);
        DLIB_TEST(cap2.sample_rate() == encoded_sample_rate);
        print_spinner();

        int counter_images  = 0;
        int counter_samples = 0;

        {
            type_safe_union<array2d<rgb_pixel>, audio_frame> frame;
            std::chrono::system_clock::time_point timestamp{};

            while (cap2.read(frame, timestamp)) {
                visit(overloaded(
                        [&](const array2d<rgb_pixel> &frame) {
                            DLIB_TEST(frame.nc() == width);
                            DLIB_TEST(frame.nr() == height);
                            ++counter_images;
                        },
                        [&](const audio_frame &frame) {
                            DLIB_TEST(frame.sample_rate == encoded_sample_rate);
                            counter_samples += frame.samples.size();
                        }
                ), frame);
                print_spinner();
            }
        }

        DLIB_TEST(counter_images == nframes);

        ::remove(muxed_file.c_str());
    }

    void test_demux_to_encode_to_decode(
        const dlib::config_reader& cfg,
        const std::string& filepath,
        AVCodecID image_codec,
        AVCodecID audio_codec
    )
    {
        const int nframes               = dlib::get_option(cfg, "nframes", 0);
        const int height                = dlib::get_option(cfg, "height", 0);
        const int width                 = dlib::get_option(cfg, "width", 0);
        const int rate                  = dlib::get_option(cfg, "sample_rate", 0);

        dlib::demuxer_ffmpeg cap(filepath);
        DLIB_TEST(cap.is_open());
        print_spinner();

        dlib::encoder_ffmpeg enc_image, enc_audio;
        dlib::decoder_ffmpeg dec_image, dec_audio;

        if (cap.video_enabled())
        {
            {
                dlib::encoder_ffmpeg::args args2;
                args2.args_common.codec = image_codec;
                args2.args_image.h      = cap.height();
                args2.args_image.w      = cap.width();
                args2.args_image.fmt    = AV_PIX_FMT_YUV420P;

                enc_image = dlib::encoder_ffmpeg(args2,  std::make_shared<std::stringstream>());
                DLIB_TEST(enc_image.is_open());
                DLIB_TEST(enc_image.is_image_encoder());
                DLIB_TEST(enc_image.get_codec_id() == args2.args_common.codec);
                DLIB_TEST(enc_image.height() == cap.height());
                DLIB_TEST(enc_image.width() == cap.width());
                print_spinner();
            }

            {
                dlib::decoder_ffmpeg::args args2;
                args2.args_common.codec = enc_image.get_codec_id();
                args2.args_image.h      = cap.height();
                args2.args_image.w      = cap.width();
                args2.args_image.fmt    = cap.pixel_fmt();

                dec_image = dlib::decoder_ffmpeg(args2);
                DLIB_TEST(dec_image.is_open());
                DLIB_TEST(dec_image.is_image_decoder());
                DLIB_TEST(dec_image.get_codec_id() == args2.args_common.codec);
                print_spinner();
            }
        }

        if (cap.audio_enabled())
        {
            {
                dlib::encoder_ffmpeg::args args2;
                args2.args_common.codec         = audio_codec;
                args2.args_audio.sample_rate    = cap.sample_rate();
                args2.args_audio.channel_layout = cap.channel_layout();
                args2.args_audio.fmt            = cap.sample_fmt();

                enc_audio = dlib::encoder_ffmpeg(args2, std::make_shared<std::stringstream>());
                DLIB_TEST(enc_audio.is_open());
                DLIB_TEST(enc_audio.is_audio_encoder());
                DLIB_TEST(enc_audio.get_codec_id() == args2.args_common.codec);
                //You can't guarantee that the requested sample rate or sample format are supported.
                //In which case, the object changes them to values that ARE supported. So we can't add
                //tests that check the sample rate is set to what we asked for.
                print_spinner();
            }

            {
                dlib::decoder_ffmpeg::args args2;
                args2.args_common.codec         = enc_audio.get_codec_id();
                args2.args_audio.sample_rate    = cap.sample_rate();
                args2.args_audio.channel_layout = cap.channel_layout();
                args2.args_audio.fmt            = cap.sample_fmt();

                dec_audio = dlib::decoder_ffmpeg(args2);
                DLIB_TEST(dec_audio.is_open());
                DLIB_TEST(dec_audio.is_audio_decoder());
                DLIB_TEST(dec_audio.get_codec_id() == args2.args_common.codec);
                print_spinner();
            }
        }

        type_safe_union<array2d<rgb_pixel>, audio_frame> frame;
        std::chrono::system_clock::time_point timestamp{};
        int counter_images  = 0;
        int counter_samples = 0;

        while (cap.read(frame, timestamp))
        {
            visit(overloaded(
                    [&](const array2d<rgb_pixel>& frame) {
                        DLIB_TEST(frame.nr() == height);
                        DLIB_TEST(frame.nc() == width);
                        DLIB_TEST(enc_image.push(frame, timestamp));
                        ++counter_images;
                    },
                    [&](const audio_frame& frame) {
                        DLIB_TEST(frame.sample_rate == rate);
                        DLIB_TEST(enc_audio.push(frame, timestamp));
                        counter_samples += frame.samples.size();
                    }
            ), frame);
            print_spinner();
        }

        DLIB_TEST(counter_images == nframes);
        DLIB_TEST(counter_samples >= cap.estimated_total_samples() - cap.sample_rate()); //within 1 second
        DLIB_TEST(counter_samples <= cap.estimated_total_samples() + cap.sample_rate()); //within 1 second

        enc_audio.flush();
        enc_image.flush();

        auto push_to_decoder = [](const encoder_ffmpeg& enc, decoder_ffmpeg& dec)
        {
            auto encoded = std::dynamic_pointer_cast<std::stringstream>(enc.get_encoded_stream());
            const std::string encoded_str = encoded->str();
            DLIB_TEST(dec.push_encoded((const uint8_t*)encoded_str.c_str(), encoded_str.size()));
            dec.flush();
            print_spinner();
        };

        if (cap.video_enabled())
            push_to_decoder(enc_image, dec_image);

        if (cap.audio_enabled())
            push_to_decoder(enc_audio, dec_audio);

        auto read_from_decoder = [&](decoder_ffmpeg& dec, const int counter_image, const int counter_audio)
        {
            frame.clear();
            timestamp = {};
            counter_images  = 0;
            counter_samples = 0;

            while (dec.read(frame, timestamp) == decoder_ffmpeg::FRAME_AVAILABLE)
            {
                visit(overloaded(
                        [&](const array2d<rgb_pixel>& frame) {
                            DLIB_TEST(frame.nc() == width);
                            DLIB_TEST(frame.nr() == height);
                            counter_images++;
                        },
                        [&](const audio_frame& frame) {
                            DLIB_TEST(frame.sample_rate == cap.sample_rate());
                            counter_samples += frame.samples.size();
                        }
                ), frame);
                print_spinner();
            }

            DLIB_TEST(counter_images == counter_image);
            DLIB_TEST(counter_samples >= counter_audio - dec.sample_rate()); //within 1 second
            DLIB_TEST(counter_samples <= counter_audio + dec.sample_rate()); //within 1 second
        };

        /*push all frames*/
        if (dec_image.is_open())
            read_from_decoder(dec_image, counter_images, 0);

        if (dec_audio.is_open())
            read_from_decoder(dec_audio, 0, cap.estimated_total_samples());
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

                test_demux_to_encode_to_decode(sublock, filepath, AV_CODEC_ID_MPEG4, AV_CODEC_ID_AC3);
                test_demux_to_mux_to_demux(sublock, filepath, AV_CODEC_ID_MPEG4, AV_CODEC_ID_AC3);
            }
        }
    } a;
}

#endif
