#include <thread>
#include "video_muxer.h"

namespace dlib
{
    Frame dlib_image_to_frame(
        const array2d<rgb_pixel>& frame,
        uint64_t timestamp_us
    )
    {
        Frame f = Frame::make_image(frame.nr(), frame.nc(), AV_PIX_FMT_RGB24, timestamp_us);

        for (int row = 0 ; row < f.frame->height ; row++)
        {
            memcpy(f.frame->data[0] + row * f.frame->linesize[0],
                   frame.begin() + row * f.frame->width,
                   f.frame->width*3);
        }

        return f;
    }

    Frame dlib_audio_to_frame(
        const audio_frame& frame,
        uint64_t timestamp_us
    )
    {
        Frame f = Frame::make_audio(frame.sample_rate, frame.samples.size(), AV_CH_LAYOUT_STEREO, AV_SAMPLE_FMT_S16, timestamp_us);
        memcpy(f.frame->data[0], frame.samples.data(), frame.samples.size()*sizeof(audio_frame::sample));
        return f;
    }

    encoder_ffmpeg::encoder_ffmpeg(
        const args &a,
        std::unique_ptr<std::ostream> out
    ) : _args(a),
        encoded(std::move(out))
    {
        connected = open();
    }

    bool encoder_ffmpeg::open()
    {
        DLIB_CASSERT(!_args.options.is_empty(), "You must set `options` to either an instance of image_args or audio_args");
        DLIB_CASSERT(encoded != nullptr, "encoded must be set to a non-null pointer");

        packet = make_avpacket();
        AVCodec* pCodec = nullptr;

        if (_args.base.codec != AV_CODEC_ID_NONE)
            pCodec = avcodec_find_encoder(_args.base.codec);
        else if (!_args.base.codec_name.empty())
            pCodec = avcodec_find_encoder_by_name(_args.base.codec_name.c_str());

        if (!pCodec)
        {
            printf("Codec %i : `%s` not found\n", _args.base.codec, _args.base.codec_name.c_str());
            return false;
        }

        pCodecCtx.reset(avcodec_alloc_context3(pCodec));
        if (!pCodecCtx)
        {
            printf("AV : failed to allocate codec context for `%s` : likely ran out of memory", pCodec->name);
            return false;
        }

        if (_args.base.nthreads > 0)
            pCodecCtx->thread_count = _args.base.nthreads;
        if (_args.base.bitrate > 0)
            pCodecCtx->bit_rate = _args.base.bitrate;
        if (_args.base.gop_size > 0)
            pCodecCtx->gop_size = _args.base.gop_size;

        if (pCodec->type == AVMEDIA_TYPE_VIDEO)
        {
            if (!_args.options.contains<args::image_args>())
            {
                printf("You forgot to set encoder_ffmpeg::args::options to image_args and fill the options\n");
                return false;
            }

            const auto& opts = _args.options.cast_to<args::image_args>();
            DLIB_ASSERT(opts.h > 0, "height must be set");
            DLIB_ASSERT(opts.w > 0, "width must be set");
            DLIB_ASSERT(opts.fmt != AV_PIX_FMT_NONE, "pixel format must be set");
            DLIB_ASSERT(opts.fps.num > 0 && opts.fps.den > 0, "FPS must be set");

            pCodecCtx->height       = opts.h;
            pCodecCtx->width        = opts.w;
            pCodecCtx->pix_fmt      = opts.fmt;
            pCodecCtx->time_base    = (AVRational){opts.fps.den, opts.fps.num};
            pCodecCtx->framerate    = (AVRational){opts.fps.num, opts.fps.den};
        }
        else if (pCodec->type == AVMEDIA_TYPE_AUDIO)
        {
            if (!_args.options.contains<args::audio_args>())
            {
                printf("You forgot to set encoder_ffmpeg::args::options to audio_args and fill the options\n");
                return false;
            }

            const auto& opts = _args.options.cast_to<args::audio_args>();

            pCodecCtx->sample_rate      = opts.sample_rate != 0 ? opts.sample_rate : pCodec->supported_samplerates ? pCodec->supported_samplerates[0] : 44100;
            pCodecCtx->sample_fmt       = opts.fmt != AV_SAMPLE_FMT_NONE ? opts.fmt : pCodec->sample_fmts ? pCodec->sample_fmts[0] : AV_SAMPLE_FMT_S16;
            pCodecCtx->channel_layout   = opts.channel_layout != 0 ? opts.channel_layout : pCodec->channel_layouts ? pCodec->channel_layouts[0] : AV_CH_LAYOUT_STEREO;
            pCodecCtx->channels         = av_get_channel_layout_nb_channels(pCodecCtx->channel_layout);
            pCodecCtx->time_base        = (AVRational){ 1, pCodecCtx->sample_rate };

            if (pCodecCtx->codec_id == AV_CODEC_ID_AAC)
                pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

            if ((pCodecCtx->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) == 0) {
                printf("Codec `%s` does not support variable frame size!\n", pCodecCtx->codec->name);
            }

            if (pCodec->supported_samplerates)
            {
                bool sample_rate_supported = false;

                for (int i = 0 ; pCodec->supported_samplerates[i] != 0 ; i++)
                {
                    if (pCodecCtx->sample_rate == pCodec->supported_samplerates[i])
                    {
                        sample_rate_supported = true;
                        break;
                    }
                }

                if (!sample_rate_supported)
                {
                    printf("Requested sample rate %i not supported. Changing to default %i\n",
                             pCodecCtx->sample_rate,
                           pCodec->supported_samplerates[0]);
                    pCodecCtx->sample_rate = pCodec->supported_samplerates[0];
                }
            }

            if (pCodec->sample_fmts)
            {
                bool sample_fmt_supported = false;

                for (int i = 0 ; pCodec->sample_fmts[i] != AV_SAMPLE_FMT_NONE ; i++)
                {
                    if (pCodecCtx->sample_fmt == pCodec->sample_fmts[i])
                    {
                        sample_fmt_supported = true;
                        break;
                    }
                }

                if (!sample_fmt_supported)
                {
                    printf("Requested sample format `%s` not supported. Changing to default `%s`\n",
                             av_get_sample_fmt_name(pCodecCtx->sample_fmt),
                             av_get_sample_fmt_name(pCodec->sample_fmts[0]));
                    pCodecCtx->sample_fmt = pCodec->sample_fmts[0];
                }
            }

            if (pCodec->channel_layouts)
            {
                bool channel_layout_supported= false;

                for (int i = 0 ; pCodec->channel_layouts[i] != 0 ; i++)
                {
                    if (pCodecCtx->channel_layout == pCodec->channel_layouts[i])
                    {
                        channel_layout_supported = true;
                        break;
                    }
                }

                if (!channel_layout_supported)
                {
                    printf("Channel layout `%s` not supported. Changing to default `%s`\n",
                             get_channel_layout_str(pCodecCtx->channel_layout).c_str(),
                             get_channel_layout_str(pCodec->channel_layouts[0]).c_str());
                    pCodecCtx->channel_layout = pCodec->channel_layouts[0];
                }
            }
        }

        av_dict opt = _args.base.codec_options;
        int ret = avcodec_open2(pCodecCtx.get(), pCodec, opt.avdic ? &opt.avdic : nullptr);
        if (ret < 0)
        {
            printf("avcodec_open2() failed : `%s`\n", get_av_error(ret).c_str());
            return false;
        }

        if (pCodec->type == AVMEDIA_TYPE_AUDIO)
        {
            audio_fifo = sw_audio_fifo(pCodecCtx->frame_size,
                                       pCodecCtx->sample_fmt,
                                       pCodecCtx->channels);
        }

        return true;
    }

    bool encoder_ffmpeg::is_open() const
    {
        return connected && pCodecCtx != nullptr && FFMPEG_INITIALIZED;
    }

    bool encoder_ffmpeg::is_image_encoder() const
    {
        return is_open() && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
    }

    bool encoder_ffmpeg::is_audio_encoder() const
    {
        return is_open() && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO;
    }

    void encoder_ffmpeg::swap_encoded_stream(std::unique_ptr<std::ostream> &out)
    {
        std::swap(encoded, out);
    }

    bool encoder_ffmpeg::push(Frame &&frame)
    {
        auto send_frame = [&](const AVFrame* frame)
        {
            int ret = avcodec_send_frame(pCodecCtx.get(), frame);
            int suc = -1; //default to not-ok

            if (ret == AVERROR(EAGAIN))
            {
                suc = 0;
            }
            else if (ret < 0 && ret != AVERROR_EOF)
            {
                printf("avcodec_send_frame() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
            }
            else if (ret >= 0)
            {
                suc = 1;
            }

            return suc;
        };

        auto recv_packet = [&]
        {
            int ret = avcodec_receive_packet(pCodecCtx.get(), packet.get());
            int suc = -1;

            if (ret == AVERROR(EAGAIN))
            {
                suc = 0; //ok but need more input
            }
            else if (ret == AVERROR_EOF)
            {
            }
            else if (ret < 0)
            {
                printf("avcodec_receive_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
            }
            else
            {
                encoded->write((char*)packet->data, packet->size);
                suc = 1;
            }

            return suc;
        };

        auto encode = [&](const AVFrame* frame)
        {
            int suc1 = 0; //-1 == error, 0 == ok but EAGAIN, 1 == ok
            int suc2 = 0; //-1 == error, 0 == ok but EAGAIN, 1 == ok
            bool ok = true;

            while (ok && suc1 == 0)
            {
                suc1 = send_frame(frame);
                ok   = suc1 >= 0;

                if (ok)
                {
                    suc2 = 1;

                    while (suc2 > 0)
                        suc2 = recv_packet();

                    ok = suc2 >= 0;
                }
            }

            return ok;
        };

        std::vector<Frame> frames;

        if (frame.is_image())
        {
            resizer_image.resize(frame,
                                 pCodecCtx->height,
                                 pCodecCtx->width,
                                 pCodecCtx->pix_fmt,
                                 frame);
            frames.push_back(std::move(frame));
        }
        else if (frame.is_audio())
        {
            const auto& opts = _args.options.cast_to<args::audio_args>();
            resizer_audio.resize(frame,
                                 pCodecCtx->sample_rate,
                                 pCodecCtx->channel_layout,
                                 pCodecCtx->sample_fmt,
                                 frame);
            frames = audio_fifo.push_pull(std::move(frame));
        }
        else
        {
            /*! FLUSH !*/
            frames.push_back(std::move(frame));
        }

        for (auto& frame : frames)
        {
            if (frame.timestamp_us > 0)
            {
                const AVRational tb1 = {1,1000000};
                const AVRational tb2 = pCodecCtx->time_base;
                frame.frame->pts = av_rescale_q(frame.timestamp_us, tb1, tb2);
            }
            else if (frame.frame)
            {
                frame.frame->pts = next_pts;
            }

            if (frame.frame)
                next_pts = frame.frame->pts + (frame.is_image() ? 1 : frame.frame->nb_samples);
        }

        bool ok = true;
        for(const auto& frame : frames)
            ok = ok && encode(frame.frame.get());

        return ok;
    }

    bool encoder_ffmpeg::push(
        const array2d<rgb_pixel> &frame,
        uint64_t timestamp_us
    )
    {
        DLIB_CASSERT(is_image_encoder(), "This object is either empty or doesn't represent an image/video encoder");
        return push(dlib_image_to_frame(frame, timestamp_us));
    }

    bool encoder_ffmpeg::push(
        const audio_frame &frame,
        uint64_t timestamp_us
    )
    {
        DLIB_CASSERT(is_audio_encoder(), "This object is either empty or doesn't represent an audio encoder");
        return push(dlib_audio_to_frame(frame, timestamp_us));
    }

    bool encoder_ffmpeg::flush()
    {
        return push(Frame{});
    }
}