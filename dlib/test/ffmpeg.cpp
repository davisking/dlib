#// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_USE_FFMPEG

#include <fstream>
#include <vector>
#include <chrono>
#include <dlib/dir_nav.h>
#include <dlib/config_reader.h>
#include <dlib/media.h>
#include <dlib/array2d.h>
#include <dlib/matrix.h>
#include <dlib/rand.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include "tester.h"

#ifndef DLIB_FFMPEG_DATA
static_assert(false, "Build is faulty. DLIB_VIDEOS_FILEPATH should be defined by cmake");
#endif

namespace  
{
    using namespace std;
    using namespace std::chrono;
    using namespace test;
    using namespace dlib;
    using namespace dlib::ffmpeg;
    
    logger dlog("test.video");

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
// UTILS
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
    
    dlib::rand rng(10000);

    template <class pixel>
    matrix<pixel> get_random_image()
    {
        matrix<pixel> img;

        img.set_size(rng.get_integer_in_range(1, 128),
                     rng.get_integer_in_range(1, 128));

        for (long i = 0 ; i < img.nr() ; ++i)
            for (long j = 0 ; j < img.nc() ; ++j)
                assign_pixel(img(i,j), rng.get_random_8bit_number());

        return img;
    }

    template <class pixel>
    void check_image (
        const frame& f,
        const matrix<pixel>& img
    )
    {
        DLIB_TEST(!f.is_empty());
        DLIB_TEST(f.is_image());
        DLIB_TEST(!f.is_audio());
        DLIB_TEST(f.samplefmt()    == AV_SAMPLE_FMT_NONE);
        DLIB_TEST(f.sample_rate()  == 0);
        DLIB_TEST(f.nsamples()     == 0);
        DLIB_TEST(f.layout()       == 0);
        DLIB_TEST(f.nchannels()    == 0);
        DLIB_TEST(f.pixfmt()       == pix_traits<pixel>::fmt);
        DLIB_TEST(f.height()       == img.nr());
        DLIB_TEST(f.width()        == img.nc());

        matrix<pixel> dummy;
        convert(f, dummy);
        DLIB_TEST(dummy.nc() == img.nc());
        DLIB_TEST(dummy.nr() == img.nr());
        DLIB_TEST(img == dummy);
    }

    template <class pixel>
    void test_frame()
    {
        const matrix<pixel> img1 = get_random_image<pixel>();
        const matrix<pixel> img2 = get_random_image<pixel>();

        // Create a frame
        frame f1(img1.nr(), img1.nc(), pix_traits<pixel>::fmt, system_clock::time_point{});
        convert(img1, f1);

        // Check frame is correct
        check_image(f1, img1);

        // Copy
        frame f2(f1);

        // Check it's a deepcopy, i.e. pointers are different but values are the same
        check_image(f2, img1);

        // Check pointers are different
        DLIB_TEST(f1.get_frame().data[0] != f2.get_frame().data[0]);

        // Set f2 and check this doesn't affect f1
        f2 = frame(img2.nr(), img2.nc(), pix_traits<pixel>::fmt, system_clock::time_point{});
        convert(img2, f2);

        check_image(f2, img2);
        check_image(f1, img1);
        DLIB_TEST(f1.get_frame().data[0] != f2.get_frame().data[0]);

        // Move
        f2 = std::move(f1);
        check_image(f2, img1);
        DLIB_TEST(f1.is_empty());

        print_spinner();
    }

    template <typename pixel_type>
    static double psnr(const matrix<pixel_type>& img1, const matrix<pixel_type>& img2)
    {
        DLIB_TEST(have_same_dimensions(img1, img2));
        const long nk           = width_step(img1) / img1.nc();
        const long data_size    = img1.size() * nk;
        auto* data1             = reinterpret_cast<const uint8_t*>(image_data(img1));
        auto* data2             = reinterpret_cast<const uint8_t*>(image_data(img2));

        double mse = 0;
        for (long i = 0; i < data_size; i += nk)
        {
            for (long k = 0; k < nk; ++k)
                mse += std::pow(static_cast<double>(data1[i + k]) - static_cast<double>(data2[i + k]), 2);
        }
        mse /= data_size;
        return 20 * std::log10(pixel_traits<pixel_type>::max()) - 10 * std::log10(mse);
    }

    void test_load_frame(const std::string& filename)
    {
        matrix<rgb_pixel> img1, img2;
        load_image(img1, filename);
        load_frame(img2, filename);
        DLIB_TEST(img1.nr() == img2.nr());
        DLIB_TEST(img1.nc() == img2.nc());
        const double similarity = psnr(img1, img2);
        DLIB_TEST_MSG(similarity > 25.0, "psnr " << similarity);
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
// DECODER
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

    template <
      class image_type
    >
    void test_decoder_images_only(
        const std::string& filepath,
        const std::string& codec,
        const int nframes,
        const int height,
        const int width
    )
    {
        decoder dec([&] {
            decoder::args args;
            args.codec_name = codec;
            return args;
        }());

        DLIB_TEST(dec.is_open());
        DLIB_TEST(dec.get_codec_name() == codec);
        DLIB_TEST(dec.is_image_decoder());
        // You can't test for dec.height() or dec.width() yet because no data has been pushed to the decoder.

        image_type      img;
        int             counter{0};
        decoder_status  status{DECODER_EAGAIN};

        ifstream fin{filepath, std::ios::binary};
        std::vector<char> buf(1024);

        const auto pull = [&]
        {
            while ((status = dec.read(img)) == DECODER_FRAME_AVAILABLE)
            {
                DLIB_TEST(img.nr()      == height);
                DLIB_TEST(img.nc()      == width);
                DLIB_TEST(dec.height()  == height);
                DLIB_TEST(dec.width()   == width);
                ++counter;
                
                if (counter % 10 == 0)
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
        DLIB_TEST(counter == nframes);
        DLIB_TEST(!dec.is_open());
    }

    template <
      class image_type
    >
    void test_decoder_full (
        const std::string&  filepath,
        const std::string&  codec,
        const int           nframes,
        const int           height,
        const int           width,
        const int           sample_rate,
        bool                has_image,
        bool                has_audio
    )
    {
        decoder dec([&] {
            decoder::args args;
            args.codec_name = codec;
            return args;
        }());

        DLIB_TEST(dec.is_open());
        DLIB_TEST(dec.get_codec_name() == codec);
        DLIB_TEST(dec.is_audio_decoder() == has_audio);
        DLIB_TEST(dec.is_image_decoder() == has_image);

        image_type                  img;
        audio<int16_t, 2>           audio_buf;
        dlib::ffmpeg::frame         frame, frame_copy;
        int                         count{0};
        int                         nsamples{0};
        int                         iteration{0};
        decoder_status              status{DECODER_EAGAIN};

        const resizing_args args_image {
            0,
            0,
            pix_traits<pixel_type_t<image_type>>::fmt
        };

        const resampling_args args_audio {
            sample_rate,
            dlib::ffmpeg::details::get_layout_from_channels(decltype(audio_buf)::nchannels),
            sample_traits<decltype(audio_buf)::format>::fmt
        };

        ifstream fin{filepath, std::ios::binary};
        std::vector<char> buf(1024);

        const auto pull = [&]
        {
            while ((status = dec.read(frame, args_image, args_audio)) == DECODER_FRAME_AVAILABLE)
            {
                if (has_audio)
                {
                    convert(frame, audio_buf);
                    convert(audio_buf, frame_copy);
                    DLIB_TEST(frame.is_audio());
                    DLIB_TEST(frame.sample_rate() == sample_rate);
                    DLIB_TEST(frame.samplefmt() == args_audio.fmt);
                    DLIB_TEST(frame_copy.is_audio());
                    DLIB_TEST(frame_copy.sample_rate() == sample_rate);
                    DLIB_TEST(frame_copy.samplefmt() == args_audio.fmt);

                    nsamples += frame.nsamples();

                    DLIB_TEST(dec.sample_rate() == sample_rate);
                    DLIB_TEST(dec.sample_fmt() == args_audio.fmt);
                }
                else
                {
                    convert(frame, img);
                    convert(img, frame_copy);
                    DLIB_TEST(frame.is_image());
                    DLIB_TEST(frame.height() == height);
                    DLIB_TEST(frame.width()  == width);
                    DLIB_TEST(frame.pixfmt() == args_image.fmt);
                    DLIB_TEST(frame_copy.is_image());
                    DLIB_TEST(frame_copy.height() == height);
                    DLIB_TEST(frame_copy.width()  == width);
                    DLIB_TEST(frame_copy.pixfmt() == args_image.fmt);

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

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
// DEMUXER
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

    template <
      class image_type
    >
    void test_demuxer_images_only (
        const std::string& filepath,
        const int nframes,
        const int height,
        const int width
    )
    {
        demuxer cap({filepath, video_enabled, audio_enabled});
        DLIB_TEST(cap.is_open());
        DLIB_TEST(cap.video_enabled());
        DLIB_TEST(cap.height()  == height);
        DLIB_TEST(cap.width()   == width);
        // DLIB_TEST(cap.estimated_nframes() == nframes); // This won't always work with ffmpeg v3. v4 onwards is fine

        image_type img;
        int counter{0};

        while (cap.read(img))
        {
            DLIB_TEST(img.nr() == height);
            DLIB_TEST(img.nc() == width);
            ++counter;

            if (counter % 10 == 0)
                print_spinner();
        }

        DLIB_TEST(counter == nframes);
        DLIB_TEST(!cap.is_open());
    }

    void test_demuxer_full (
        const std::string& filepath,
        const int nframes,
        const int height,
        const int width,
        const int sample_rate,
        bool has_video,
        bool has_audio
    )
    {
        demuxer cap(filepath);
        DLIB_TEST(cap.video_enabled()       == has_video);
        DLIB_TEST(cap.audio_enabled()       == has_audio);
        DLIB_TEST(cap.height()              == height);
        DLIB_TEST(cap.width()               == width);
        DLIB_TEST(cap.sample_rate()         == sample_rate);
        // DLIB_TEST(cap.estimated_nframes()   == nframes); // This won't always work with ffmpeg v3. v4 onwards is fine
        const int estimated_samples_min = cap.estimated_total_samples() - cap.sample_rate(); // - 1s
        const int estimated_samples_max = cap.estimated_total_samples() + cap.sample_rate(); // + 1s

        dlib::ffmpeg::frame frame;
        int                 counter_images{0};
        int                 counter_samples{0};
        int                 iteration{0};

        resizing_args args_image;
        args_image.fmt = AV_PIX_FMT_RGB24;

        resampling_args args_audio;
        args_audio.sample_rate      = sample_rate;
        args_audio.fmt              = AV_SAMPLE_FMT_S16;
        args_audio.channel_layout   = AV_CH_LAYOUT_STEREO;

        while (cap.read(frame, args_image, args_audio))
        {
            if (frame.is_image())
            {
                DLIB_TEST(frame.height() == height);
                DLIB_TEST(frame.width()  == width);
                DLIB_TEST(frame.pixfmt() == args_image.fmt);
                
                ++counter_images;
            }

            if (frame.is_audio())
            {
                DLIB_TEST(frame.sample_rate()   == sample_rate);
                DLIB_TEST(frame.layout()        == args_audio.channel_layout);
                DLIB_TEST(frame.samplefmt()     == args_audio.fmt);
                counter_samples += frame.nsamples();
            }

            ++iteration;
            if (iteration % 10 == 0)
                print_spinner();
        }

        DLIB_TEST(counter_images == nframes);
        DLIB_TEST(counter_samples >= estimated_samples_min); //within 1 second
        DLIB_TEST(counter_samples <= estimated_samples_max); //within 1 second
        DLIB_TEST(!cap.is_open());
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
// ENCODER
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

    void test_encoder (
        const std::string& filepath,
        AVCodecID image_codec,
        AVCodecID audio_codec
    )
    {
        // Load a video/audio as a source of frames
        demuxer cap(filepath);
        DLIB_TEST(cap.is_open());
        const bool has_video = cap.video_enabled();
        const bool has_audio = cap.audio_enabled();
        const int  height    = cap.height();
        const int  width     = cap.width();
        const auto pixfmt    = cap.pixel_fmt();
        const auto fps       = cap.fps();
        const int  rate      = cap.sample_rate();
        const auto samplefmt = cap.sample_fmt();
        const auto layout    = cap.channel_layout();

        std::vector<frame> frames;
        frame f;
        int nimages{0};
        int nsamples{0};

        while (cap.read(f))
        {
            if (f.is_image())
                ++nimages;
            if (f.is_audio())
                nsamples += f.nsamples();
            frames.push_back(std::move(f));
        }
            
        print_spinner();

        // Encoder configured using same parameters as video
        encoder enc_image, enc_audio;
        std::vector<uint8_t> buf_image, buf_audio;

        if (has_video)
        {
            enc_image = encoder([&] {
                encoder::args args;
                args.args_codec.codec       = image_codec;
                args.args_image.h           = height;
                args.args_image.w           = width;
                args.args_image.framerate   = fps;
                args.args_image.fmt         = AV_PIX_FMT_YUV420P;
                return args;
            }(), sink(buf_image));

            DLIB_TEST(enc_image.is_open());
            DLIB_TEST(enc_image.is_image_encoder());
            DLIB_TEST(enc_image.get_codec_id()  == image_codec);
            DLIB_TEST(enc_image.height()        == height);
            DLIB_TEST(enc_image.width()         == width);
            // Can't check for framerate, as this might have been changed from requested value, due to codec availability.
            print_spinner();
        }

        if (has_audio)
        {
            enc_audio = encoder([&] {
                encoder::args args;
                args.args_codec.codec           = audio_codec;
                args.args_audio.sample_rate     = rate;
                args.args_audio.channel_layout  = layout;
                args.args_audio.fmt             = samplefmt;
                return args;
            }(), sink(buf_audio));

            DLIB_TEST(enc_audio.is_open());
            DLIB_TEST(enc_audio.is_audio_encoder());
            DLIB_TEST(enc_audio.get_codec_id() == audio_codec);
            print_spinner();
        }

        int iteration{0};

        for (auto& f : frames)
        {
            if (f.is_image())
                DLIB_TEST(enc_image.push(std::move(f)));
            
            if (f.is_audio())
                DLIB_TEST(enc_audio.push(std::move(f)));

            if ((iteration++ % 10) == 0)
                print_spinner();
        }

        enc_image.flush();
        enc_audio.flush();
        print_spinner();

        // Decoder everything back
        decoder dec_image, dec_audio;

        if (has_video)
        {
            dec_image = decoder([&] {
                decoder::args args;
                args.codec = image_codec;
                return args;
            }());

            DLIB_TEST(dec_image.is_open());
            DLIB_TEST(dec_image.is_image_decoder());
            DLIB_TEST(dec_image.get_codec_id() == image_codec);
            DLIB_TEST(dec_image.push_encoded(buf_image.data(), buf_image.size()));
            dec_image.flush();

            int images = 0;
            decoder_status status;

            while ((status = dec_image.read(f)) == DECODER_FRAME_AVAILABLE)
            {
                ++images;
                DLIB_TEST(f.height()            == height);
                DLIB_TEST(f.width()             == width);
                DLIB_TEST(dec_image.height()    == height);
                DLIB_TEST(dec_image.width()     == width);
                print_spinner();
            }

            DLIB_TEST(images == nimages);
        }

        if (has_audio)
        {
            dec_audio = decoder([&] {
                decoder::args args;
                args.codec = audio_codec;
                return args;
            }());

            DLIB_TEST(dec_audio.is_open());
            DLIB_TEST(dec_audio.is_audio_decoder());
            DLIB_TEST(dec_audio.get_codec_id()  == audio_codec);
            DLIB_TEST(dec_audio.push_encoded(buf_audio.data(), buf_audio.size()));
            dec_audio.flush();

            int samples = 0;
            decoder_status status;

            while ((status = dec_audio.read(f, {}, {rate})) == DECODER_FRAME_AVAILABLE)
            {
                samples += f.nsamples();
                DLIB_TEST(f.sample_rate() == rate);
                print_spinner();
            }

            DLIB_TEST(samples > (nsamples - rate));
            DLIB_TEST(samples < (nsamples + rate));
        }
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
// MUXER
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

    void test_muxer (
        const std::string& filepath,
        AVCodecID image_codec,
        AVCodecID audio_codec
    )
    {
        const std::string tmpfile = "dummy.avi";

        // Load a video/audio as a source of frames
        demuxer cap(filepath);
        DLIB_TEST(cap.is_open());
        const bool has_video = cap.video_enabled();
        const bool has_audio = cap.audio_enabled();
        const int  height    = cap.height();
        const int  width     = cap.width();
        const auto pixfmt    = cap.pixel_fmt();
        const auto fps       = cap.fps();
        const int  rate      = cap.sample_rate();
        const auto samplefmt = cap.sample_fmt();
        const auto layout    = cap.channel_layout();

        std::vector<frame> frames;
        frame f;
        int nimages{0};
        int nsamples{0};

        while (cap.read(f))
        {
            if (f.is_image())
                ++nimages;
            if (f.is_audio())
                nsamples += f.nsamples();
            frames.push_back(std::move(f));
        }
            
        print_spinner();

        // Muxer configured using parameters in video
        {
            muxer writer([&] {
                muxer::args args;
                args.filepath = tmpfile;
                args.enable_image = has_video;
                args.enable_audio = has_audio;

                if (has_video)
                {
                    args.args_image.codec        = image_codec;
                    args.args_image.h            = height;
                    args.args_image.w            = width;
                    args.args_image.framerate    = fps;
                    args.args_image.fmt          = AV_PIX_FMT_YUV420P;
                }

                if (has_audio)
                {
                    args.args_audio.codec            = audio_codec;
                    args.args_audio.sample_rate      = rate;
                    args.args_audio.channel_layout   = layout;
                    args.args_audio.fmt              = samplefmt;
                }

                return args;
            }());

            DLIB_TEST(writer.is_open());
            DLIB_TEST(writer.audio_enabled() == has_audio);
            DLIB_TEST(writer.video_enabled() == has_video);

            if (has_video)
            {
                DLIB_TEST(writer.get_video_codec_id()   == image_codec);
                DLIB_TEST(writer.height()               == height);
                DLIB_TEST(writer.width()                == width);
            }

            if (has_audio)
            {
                DLIB_TEST(writer.get_audio_codec_id() == audio_codec);
                //You can't guarantee that the requested sample rate or sample format are supported.
                //In which case, the object changes them to values that ARE supported. So we can't add
                //tests that check the sample rate is set to what we asked for.
            } 

            for (auto& f : frames)
                writer.push(std::move(f));

            // muxer.flush(); // You don't need to call this since muxer's destructor already does.
        }

        // Demux everything back
        demuxer cap2(tmpfile);
        DLIB_TEST(cap2.is_open());
        DLIB_TEST(cap2.video_enabled()       == has_video);
        DLIB_TEST(cap2.audio_enabled()       == has_audio);
        if (has_video)
            DLIB_TEST(cap2.get_video_codec_id() == image_codec);
        if (has_audio)
            DLIB_TEST(cap2.get_audio_codec_id() == audio_codec);
        DLIB_TEST(cap2.height() == height);
        DLIB_TEST(cap2.width()  == width);
        // Can't test for sample_rate since muxer may have changed it due to codec availability.

        int images{0};
        int samples{0};
        int iteration{0};

        while (cap2.read(f, {}, {rate}))
        {
            if (f.is_image())
                ++images;

            if (f.is_audio())
                samples += f.nsamples();

            ++iteration;
            if (iteration % 10 == 0)
                print_spinner();
        }

        DLIB_TEST(images == nimages);
        DLIB_TEST(samples >= (nsamples - rate));
        DLIB_TEST(samples <= (nsamples + rate));
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
            for (int i = 0 ; i < 10 ; ++i)
            {
                test_frame<rgb_pixel>();
                test_frame<bgr_pixel>();
                test_frame<rgb_alpha_pixel>();
                test_frame<bgr_alpha_pixel>();
            }

            dlib::file f(DLIB_FFMPEG_DATA);
            dlib::config_reader cfg(f.full_name());

            {
                const auto& image_block = cfg.block("images");
                std::vector<string> blocks;
                image_block.get_blocks(blocks);

                for (const auto& block : blocks)
                {
                    const auto& sublock = image_block.block(block);
                    const std::string filepath = get_parent_directory(f).full_name() + "/" + sublock["file"];

                    test_load_frame(filepath);
                }
            }

            {
                const auto& video_raw_block = cfg.block("decoding");
                std::vector<string> blocks;
                video_raw_block.get_blocks(blocks);

                for (const auto& block : blocks)
                {
                    const auto& sublock = video_raw_block.block(block);
                    const std::string filepath  = get_parent_directory(f).full_name() + "/" + sublock["file"];
                    const std::string codec     = dlib::get_option(sublock, "codec", "");
                    const int nframes           = dlib::get_option(sublock, "nframes", 0);
                    const int height            = dlib::get_option(sublock, "height", 0);
                    const int width             = dlib::get_option(sublock, "width", 0);
                    const int sample_rate       = dlib::get_option(sublock, "sample_rate", 0);
                    const bool has_image        = height > 0 && width > 0;
                    const bool has_audio        = sample_rate > 0;   

                    if (has_image)
                    {
                        test_decoder_images_only<array2d<rgb_pixel>>(filepath, codec, nframes, height, width);
                        test_decoder_images_only<array2d<rgb_alpha_pixel>>(filepath, codec, nframes, height, width);
                        test_decoder_images_only<matrix<bgr_pixel>>(filepath, codec, nframes, height, width);
                        test_decoder_images_only<matrix<bgr_alpha_pixel>>(filepath, codec, nframes, height, width);
                    } 

                    test_decoder_full<array2d<rgb_pixel>>(filepath, codec, nframes, height, width, sample_rate, has_image, has_audio);
                    test_decoder_full<matrix<bgr_pixel>>(filepath, codec, nframes, height, width, sample_rate, has_image, has_audio);
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

                    const std::string tmpfile = "dummy.avi";
                    const int nframes       = dlib::get_option(sublock, "nframes", 0);
                    const int height        = dlib::get_option(sublock, "height", 0);
                    const int width         = dlib::get_option(sublock, "width", 0);
                    const int sample_rate   = dlib::get_option(sublock, "sample_rate", 0);
                    const bool has_video    = height > 0 && width > 0 && nframes > 0;
                    const bool has_audio    = sample_rate > 0;

                    if (has_video)
                    {
                        test_demuxer_images_only<array2d<rgb_pixel>>(filepath, nframes, height, width);
                        test_demuxer_images_only<matrix<bgr_pixel>>(filepath, nframes, height, width);
                    }

                    test_demuxer_full(filepath, nframes, height, width, sample_rate, has_video, has_audio);
                    test_encoder(filepath, AV_CODEC_ID_MPEG4, AV_CODEC_ID_AC3);
                    test_muxer(filepath, AV_CODEC_ID_MPEG4, AV_CODEC_ID_AC3);
                }
            }
        }
    } a;
}

#endif
