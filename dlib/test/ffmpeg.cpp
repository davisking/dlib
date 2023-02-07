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

        decoder dec([&] {
            decoder::args args;
            args.args_codec.codec_name      = codec;
            args.args_image.fmt             = AV_PIX_FMT_RGB24;
            args.args_audio.fmt             = AV_SAMPLE_FMT_S16;
            args.args_audio.channel_layout  = AV_CH_LAYOUT_MONO;
            return args;
        }());

        DLIB_TEST(dec.is_open());
        DLIB_TEST(dec.get_codec_name() == codec);
        DLIB_TEST(is_audio ? dec.is_audio_decoder() : dec.is_image_decoder());

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

    void test_demuxer_encoder_decoder (
        const std::string& filepath,
        const dlib::config_reader& cfg,
        AVCodecID image_codec,
        AVCodecID audio_codec
    )
    {
        const int nframes       = dlib::get_option(cfg, "nframes", 0);
        const int height        = dlib::get_option(cfg, "height", 0);
        const int width         = dlib::get_option(cfg, "width", 0);
        const int sample_rate   = dlib::get_option(cfg, "sample_rate", 0);
        const bool has_video    = height > 0 && width > 0 && nframes > 0;
        const bool has_audio    = sample_rate > 0;

        demuxer cap{[&] {
            demuxer::args args;
            args.filepath           = filepath;
            args.image_options.fmt  = AV_PIX_FMT_RGB24;
            args.audio_options.fmt  = AV_SAMPLE_FMT_S16;
            return args;
        }()};
        
        DLIB_TEST(cap.is_open());
        DLIB_TEST(cap.video_enabled()       == has_video);
        DLIB_TEST(cap.audio_enabled()       == has_audio);
        DLIB_TEST(cap.height()              == height);
        DLIB_TEST(cap.width()               == width);
        DLIB_TEST(cap.sample_rate()         == sample_rate);
        DLIB_TEST(cap.estimated_nframes()   == nframes);
        const int estimated_samples_min = cap.estimated_total_samples() - cap.sample_rate(); // - 1s
        const int estimated_samples_max = cap.estimated_total_samples() + cap.sample_rate(); // + 1s

        if (has_video)
        {
            DLIB_TEST(cap.pixel_fmt() == AV_PIX_FMT_RGB24);
        }
        if (has_audio)
        {
            DLIB_TEST(cap.sample_fmt() == AV_SAMPLE_FMT_S16);
        }

        encoder enc_image, enc_audio;
        decoder dec_image, dec_audio;
        std::vector<uint8_t> buf_image, buf_audio;

        if (has_video)
        {
            {
                enc_image = encoder([&]{
                    encoder::args args;
                    args.args_codec.codec       = image_codec;
                    args.args_image.h           = cap.height();
                    args.args_image.w           = cap.width();
                    args.args_image.framerate   = cap.fps();
                    args.args_image.fmt         = AV_PIX_FMT_YUV420P;
                    return args;
                }(), sink(buf_image));

                DLIB_TEST(enc_image.is_open());
                DLIB_TEST(enc_image.is_image_encoder());
                DLIB_TEST(enc_image.get_codec_id()  == image_codec);
                DLIB_TEST(enc_image.height()        == cap.height());
                DLIB_TEST(enc_image.width()         == cap.width());
                print_spinner();
            }

            {
                dec_image = decoder{[&]{
                    decoder::args args;
                    args.args_codec.codec  = enc_image.get_codec_id();
                    args.args_image.h      = cap.height();
                    args.args_image.w      = cap.width();
                    args.args_image.fmt    = cap.pixel_fmt();
                    return args;
                }()};

                DLIB_TEST(dec_image.is_open());
                DLIB_TEST(dec_image.is_image_decoder());
                DLIB_TEST(dec_image.get_codec_id() == enc_image.get_codec_id());
                print_spinner();
            }
        }

        if (has_audio)
        {
            {
                enc_audio = encoder([&]{
                    encoder::args args;
                    args.args_codec.codec           = audio_codec;
                    args.args_audio.sample_rate     = cap.sample_rate();
                    args.args_audio.channel_layout  = cap.channel_layout();
                    args.args_audio.fmt             = cap.sample_fmt();
                    return args;
                }(), sink(buf_audio));

                DLIB_TEST(enc_audio.is_open());
                DLIB_TEST(enc_audio.is_audio_encoder());
                DLIB_TEST(enc_audio.get_codec_id()      == audio_codec);
                //You can't guarantee that the requested sample rate or sample format are supported.
                //In which case, the object changes them to values that ARE supported. So we can't add
                //tests that check the sample rate is set to what we asked for.
                print_spinner();
            }

            {
                dec_audio = decoder{[&]{
                    decoder::args args;
                    args.args_codec.codec           = enc_audio.get_codec_id();
                    args.args_audio.sample_rate     = cap.sample_rate();
                    args.args_audio.channel_layout  = cap.channel_layout();
                    args.args_audio.fmt             = cap.sample_fmt();
                    return args;
                }()};

                DLIB_TEST(dec_audio.is_open());
                DLIB_TEST(dec_audio.is_audio_decoder());
                DLIB_TEST(dec_audio.get_codec_id() == enc_audio.get_codec_id());
                print_spinner();
            }
        }
        
        dlib::ffmpeg::frame frame, frame_copy;
        array2d<rgb_pixel>  img;
        audio<int16_t, 1>   audio1;
        audio<int16_t, 2>   audio2;
        int                 counter_images{0};
        int                 counter_samples{0};
        int                 iteration{0};

        while (cap.read(frame))
        {
            if (frame.is_image())
            {
                // Test frame 
                DLIB_TEST(frame.height()    == height);
                DLIB_TEST(frame.width()     == width);
                DLIB_TEST(frame.pixfmt()    == AV_PIX_FMT_RGB24);
                convert(frame, img);

                // Test frame -> dlib array
                DLIB_TEST(img.nr() == height);
                DLIB_TEST(img.nc() == width);
                convert(img, frame_copy);

                // Test dlib array -> frame
                DLIB_TEST(frame_copy.height()   == frame.height());
                DLIB_TEST(frame_copy.width()    == frame.width());
                DLIB_TEST(frame_copy.pixfmt()   == frame.pixfmt());

                // Push to encoder
                DLIB_TEST(enc_image.push(std::move(frame)));
                
                ++counter_images;
            }

            if (frame.is_audio())
            {
                // Test frame 
                DLIB_TEST(frame.sample_rate() == sample_rate);
                DLIB_TEST(frame.samplefmt()   == AV_SAMPLE_FMT_S16);

                // Test frame -> dlib array
                // Test dlib array -> frame
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
                DLIB_TEST(frame_copy.is_audio());
                DLIB_TEST(frame_copy.sample_rate()  == frame.sample_rate());
                DLIB_TEST(frame_copy.samplefmt()    == frame.samplefmt());
                DLIB_TEST(frame_copy.nsamples()     == frame.nsamples());
                DLIB_TEST(frame_copy.nchannels()    == frame.nchannels());

                counter_samples += frame.nsamples();

                // Push to encoder
                DLIB_TEST(enc_audio.push(std::move(frame))); 
            }

            ++iteration;
            if (iteration % 10 == 0)
                print_spinner();
        }

        DLIB_TEST(counter_images == nframes);
        DLIB_TEST(counter_samples >= estimated_samples_min); //within 1 second
        DLIB_TEST(counter_samples <= estimated_samples_max); //within 1 second
        DLIB_TEST(!cap.is_open());

        enc_audio.flush();
        enc_image.flush();

        print_spinner();

        if (has_video)
        {
            DLIB_TEST(dec_image.push_encoded(buf_image.data(), buf_image.size()));
            print_spinner();
            dec_image.flush();
            
            counter_images = 0;
            decoder_status status;

            while ((status = dec_image.read(frame)) == DECODER_FRAME_AVAILABLE)
            {
                ++counter_images;
                DLIB_TEST(frame.height()    == height);
                DLIB_TEST(frame.width()     == width);
                DLIB_TEST(frame.pixfmt()    == AV_PIX_FMT_RGB24);
            }

            DLIB_TEST(counter_images == nframes);
        }

        if (has_audio)
        {
            DLIB_TEST(dec_audio.push_encoded(buf_audio.data(), buf_audio.size()));
            print_spinner();
            dec_audio.flush();
            
            counter_samples = 0;
            decoder_status status;

            while ((status = dec_audio.read(frame)) == DECODER_FRAME_AVAILABLE)
            {
                counter_samples += frame.nsamples();
                DLIB_TEST(frame.sample_rate() == sample_rate);
                DLIB_TEST(frame.samplefmt()   == AV_SAMPLE_FMT_S16);
            }

            DLIB_TEST(counter_samples >= estimated_samples_min); //within 1 second
            DLIB_TEST(counter_samples <= estimated_samples_max); //within 1 second
        }
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

                    test_demuxer_encoder_decoder(filepath, sublock, AV_CODEC_ID_MPEG4, AV_CODEC_ID_AC3);
                }
            }
        }
    } a;
}

#endif