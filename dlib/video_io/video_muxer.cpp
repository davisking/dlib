#include <thread>
#include <chrono>
#include "video_muxer.h"

using namespace std::chrono;

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

    void check_audio_properties(
        AVCodec* pCodec,
        AVCodecContext* pCodecCtx
    )
    {
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
        std::shared_ptr<std::ostream> out
    ) : _args(a),
        encoded(std::move(out))
    {
        connected = open();
    }

    encoder_ffmpeg::encoder_ffmpeg(
        const args &a,
        packet_callback clb
    ) : _args(a),
        packet_ready_callback(std::move(clb))
    {
        connected = open();
    }

    encoder_ffmpeg::~encoder_ffmpeg()
    {
        flush();
    }

    bool encoder_ffmpeg::open()
    {
        if (packet_ready_callback == nullptr)
        {
            DLIB_CASSERT(encoded != nullptr, "encoded must be set to a non-null pointer");
        }

        packet = make_avpacket();
        AVCodec* pCodec = nullptr;

        if (_args.args_common.codec != AV_CODEC_ID_NONE)
            pCodec = avcodec_find_encoder(_args.args_common.codec);
        else if (!_args.args_common.codec_name.empty())
            pCodec = avcodec_find_encoder_by_name(_args.args_common.codec_name.c_str());

        if (!pCodec)
        {
            printf("Codec %i : `%s` not found\n", _args.args_common.codec, _args.args_common.codec_name.c_str());
            return false;
        }

        pCodecCtx.reset(avcodec_alloc_context3(pCodec));
        if (!pCodecCtx)
        {
            printf("AV : failed to allocate codec context for `%s` : likely ran out of memory", pCodec->name);
            return false;
        }

        if (_args.args_common.nthreads > 0)
            pCodecCtx->thread_count = _args.args_common.nthreads;
        if (_args.args_common.bitrate > 0)
            pCodecCtx->bit_rate = _args.args_common.bitrate;
        if (_args.args_common.flags > 0)
            pCodecCtx->flags |= _args.args_common.flags;

        if (pCodec->type == AVMEDIA_TYPE_VIDEO)
        {
            DLIB_CASSERT(_args.args_image.h > 0, "height must be set");
            DLIB_CASSERT(_args.args_image.w > 0, "width must be set");
            DLIB_CASSERT(_args.args_image.fmt != AV_PIX_FMT_NONE, "pixel format must be set");
            DLIB_CASSERT(_args.args_image.fps.num > 0 && _args.args_image.fps.den > 0, "FPS must be set");

            pCodecCtx->height       = _args.args_image.h;
            pCodecCtx->width        = _args.args_image.w;
            pCodecCtx->pix_fmt      = _args.args_image.fmt;
            pCodecCtx->time_base    = (AVRational){_args.args_image.fps.den, _args.args_image.fps.num};
            pCodecCtx->framerate    = (AVRational){_args.args_image.fps.num, _args.args_image.fps.den};

            if (_args.args_image.gop_size > 0)
                pCodecCtx->gop_size = _args.args_image.gop_size;

            //don't know what src options are, but at least dst options are set
            resizer_image.reset(pCodecCtx->height, pCodecCtx->width, pCodecCtx->pix_fmt,
                                pCodecCtx->height, pCodecCtx->width, pCodecCtx->pix_fmt);
        }
        else if (pCodec->type == AVMEDIA_TYPE_AUDIO)
        {
            DLIB_CASSERT(_args.args_audio.sample_rate > 0, "sample rate not set");
            DLIB_CASSERT(_args.args_audio.channel_layout > 0, "channel layout not set");
            DLIB_CASSERT(_args.args_audio.fmt != AV_SAMPLE_FMT_NONE, "audio sample format not set");

            pCodecCtx->sample_rate      = _args.args_audio.sample_rate;
            pCodecCtx->sample_fmt       = _args.args_audio.fmt;
            pCodecCtx->channel_layout   = _args.args_audio.channel_layout;
            pCodecCtx->channels         = av_get_channel_layout_nb_channels(pCodecCtx->channel_layout);
            pCodecCtx->time_base        = (AVRational){ 1, pCodecCtx->sample_rate };
            check_audio_properties(pCodec, pCodecCtx.get());

            if (pCodecCtx->codec_id == AV_CODEC_ID_AAC)
                pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

            //don't know what src options are, but at least dst options are set
            resizer_audio.reset(
                    pCodecCtx->sample_rate, pCodecCtx->channel_layout, pCodecCtx->sample_fmt,
                    pCodecCtx->sample_rate, pCodecCtx->channel_layout, pCodecCtx->sample_fmt
            );
        }

        av_dict opt = _args.args_common.codec_options;
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
        return  connected &&
                pCodecCtx != nullptr &&
                (encoded != nullptr || packet_ready_callback != nullptr) &&
                FFMPEG_INITIALIZED;
    }

    bool encoder_ffmpeg::is_image_encoder() const
    {
        return is_open() && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
    }

    bool encoder_ffmpeg::is_audio_encoder() const
    {
        return is_open() && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO;
    }

    std::shared_ptr<std::ostream> encoder_ffmpeg::get_encoded_stream()
    {
        return encoded;
    }

    bool encoder_ffmpeg::push(Frame &&frame)
    {
        if (!is_open())
            return false;

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
                if (packet_ready_callback)
                {
                    /*invoke muxer callback*/
                    if (packet_ready_callback(packet.get(), pCodecCtx.get()))
                        suc = 1;
                }
                else
                {
                    encoded->write((char*)packet->data, packet->size);
                    suc = 1;
                }
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
            resizer_image.resize(frame, frame);
            frames.push_back(std::move(frame));
        }
        else if (frame.is_audio())
        {
            resizer_audio.resize(frame, frame);
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

    void encoder_ffmpeg::flush()
    {
        push(Frame{});
        pCodecCtx.reset(nullptr); //close encoder
        connected = false;
    }

    int encoder_ffmpeg::height() const
    {
        return is_image_encoder() ? pCodecCtx->height : 0;
    }

    int encoder_ffmpeg::width() const
    {
        return is_image_encoder() ? pCodecCtx->width : 0;
    }

    AVPixelFormat encoder_ffmpeg::pixel_fmt() const
    {
        return is_image_encoder() ? pCodecCtx->pix_fmt : AV_PIX_FMT_NONE;
    }

    int encoder_ffmpeg::sample_rate() const
    {
        return is_audio_encoder() ? pCodecCtx->sample_rate : 0;
    }

    uint64_t encoder_ffmpeg::channel_layout() const
    {
        return is_audio_encoder() ? pCodecCtx->channel_layout : 0;
    }

    int encoder_ffmpeg::nchannels() const
    {
        return is_audio_encoder() ? av_get_channel_layout_nb_channels(pCodecCtx->channel_layout) : 0;
    }

    AVSampleFormat encoder_ffmpeg::sample_fmt() const
    {
        return is_audio_encoder() ? pCodecCtx->sample_fmt : AV_SAMPLE_FMT_NONE;
    }

    muxer_ffmpeg::muxer_ffmpeg(const args &a)
    {
        st._args = a;
        st.connected = open();
    }

    muxer_ffmpeg::muxer_ffmpeg(muxer_ffmpeg &&other)
    {
        swap(other);
    }

    muxer_ffmpeg& muxer_ffmpeg::operator=(muxer_ffmpeg &&other)
    {
        swap(other);
        return *this;
    }

    muxer_ffmpeg::~muxer_ffmpeg()
    {
        flush();
    }

    void muxer_ffmpeg::swap(muxer_ffmpeg &b)
    {
        /*!
            The reason why we don't let the compiler synthesise swap() and move semantics is because we must
            manually reset the opaque pointers and callbacks.
        !*/
        using std::swap;
        swap(st, b.st);
        if (st.pFormatCtx)
            st.pFormatCtx->opaque = this;
        if (b.st.pFormatCtx)
            b.st.pFormatCtx->opaque = &b;
        st.encoder_image.packet_ready_callback = [this](AVPacket* pkt, AVCodecContext* ctx){return handle_packet(pkt, ctx);};
        st.encoder_image.packet_ready_callback = [this](AVPacket* pkt, AVCodecContext* ctx){return handle_packet(pkt, ctx);};
        b.st.encoder_image.packet_ready_callback = [&b](AVPacket* pkt, AVCodecContext* ctx){return b.handle_packet(pkt, ctx);};
        b.st.encoder_image.packet_ready_callback = [&b](AVPacket* pkt, AVCodecContext* ctx){return b.handle_packet(pkt, ctx);};
    }

    void swap(muxer_ffmpeg& a, muxer_ffmpeg& b)
    {
        a.swap(b);
    }

    bool muxer_ffmpeg::is_open() const
    {
        return st.connected && st.pFormatCtx != nullptr && (st.encoder_audio.is_open() || st.encoder_image.is_open());
    }

    bool muxer_ffmpeg::open()
    {
        if (!st._args.enable_audio && !st._args.enable_image)
        {
            printf("You need to set at least one of `enable_audio` or `enable_image`\n");
            return false;
        }

        {
            const char* const format_name   = st._args.output_format.empty() ? nullptr : st._args.output_format.c_str();
            const char* const filename      = st._args.filepath.empty()      ? nullptr : st._args.filepath.c_str();
            AVFormatContext* pFormatCtx = nullptr;
            int ret = avformat_alloc_output_context2(&pFormatCtx, nullptr, format_name, filename);
            if (ret < 0)
            {
                printf("avformat_alloc_output_context2() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }

            if (!pFormatCtx->oformat)
            {
                printf("Output format is null\n");
                return false;
            }

            st.pFormatCtx.reset(pFormatCtx);
        }

        auto setup_stream = [&](bool is_video, encoder_ffmpeg& enc) -> bool
        {
            encoder_ffmpeg::args args2;
            if (is_video)
            {
                args2.args_common = static_cast<encoder_ffmpeg::args::channel_args>(st._args.args_image);
                args2.args_image  = static_cast<encoder_ffmpeg::args::image_args>(st._args.args_image);
            }
            else
            {
                args2.args_common = static_cast<encoder_ffmpeg::args::channel_args>(st._args.args_audio);
                args2.args_audio  = static_cast<encoder_ffmpeg::args::audio_args>(st._args.args_audio);
            }

            if (st.pFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
                args2.args_common.flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

            enc = encoder_ffmpeg(args2, [this](AVPacket* pkt, AVCodecContext* ctx){return handle_packet(pkt, ctx);});

            if (!enc.is_open())
                return false;

            AVStream* stream = avformat_new_stream(st.pFormatCtx.get(), enc.pCodecCtx->codec);

            if (!stream)
            {
                printf("avformat_new_stream() failed\n");
                return false;
            }

            stream->id = st.pFormatCtx->nb_streams-1;

            int ret = avcodec_parameters_from_context(stream->codecpar, enc.pCodecCtx.get());
            if (ret < 0)
            {
                printf("avcodec_parameters_from_context() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }

            if (is_video)
                st.stream_image = stream;
            else
                st.stream_audio = stream;

            return true;
        };

        if (st._args.enable_image)
        {
            if (!setup_stream(true, st.encoder_image))
                return false;
        }

        if (st._args.enable_audio)
        {
            if (!setup_stream(false, st.encoder_audio))
                return false;
        }

        st.pFormatCtx->opaque = this; //be careful, this needs to be swapped when (*this) is swapped.
        st.pFormatCtx->interrupt_callback.opaque    = st.pFormatCtx.get();
        st.pFormatCtx->interrupt_callback.callback  = [](void* ctx) -> int {
            AVFormatContext* pFormatCtx = (AVFormatContext*)ctx;
            muxer_ffmpeg* me = (muxer_ffmpeg*)pFormatCtx->opaque;
            return me->interrupt_callback();
        };

        if (st._args.max_delay > 0)
            st.pFormatCtx->max_delay = st._args.max_delay;

        //TODO: is this necessary?
        //_pFormatCtx->flags = AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;

        if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
        {
            av_dict opt = st._args.protocol_options;

            int ret = avio_open2(&st.pFormatCtx->pb, st._args.filepath.c_str(), AVIO_FLAG_WRITE, &st.pFormatCtx->interrupt_callback, opt.avdic ? &opt.avdic : nullptr);

            if (ret < 0)
            {
                printf("avio_open2() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }
        }

        av_dict opt = st._args.format_options;

        int ret = avformat_write_header(st.pFormatCtx.get(), opt.avdic ? &opt.avdic : nullptr);
        if (ret < 0)
        {
            printf("avformat_write_header() failed : `%s`\n", get_av_error(ret).c_str());
            return false;
        }

        return true;
    }

    bool muxer_ffmpeg::interrupt_callback()
    {
        bool do_interrupt = false;
        const auto now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        if (st._args.connect_timeout_ms > 0 && st.connected_time_ms == 0)
        {
            if (st.connecting_time_ms == 0)
                st.connecting_time_ms = now;

            const auto diff_ms = now - st.connecting_time_ms;
            do_interrupt = do_interrupt || (diff_ms > st._args.connect_timeout_ms);
        }

        if (st._args.interrupter)
        {
            do_interrupt = do_interrupt || st._args.interrupter();
        }

        return do_interrupt;
    }

    bool muxer_ffmpeg::push(Frame &&frame)
    {
        if (!is_open())
            return false;

        if (frame.is_image())
        {
            if (!st.encoder_image.is_open())
            {
                printf("frame is an image type but image encoder is not initialized\n");
                return false;
            }

            if (!st.encoder_image.push(std::move(frame)))
                return false;
        }

        else if (frame.is_image())
        {
            if (!st.encoder_audio.is_open())
            {
                printf("frame is of audio type but audio encoder is not initialized\n");
                return false;
            }

            if (!st.encoder_audio.push(std::move(frame)))
                return false;
        }
        else
        {
            return false;
        }

        return true;
    }

    void muxer_ffmpeg::flush()
    {
        if (!is_open())
            return;

        if (st.encoder_image.is_open())
            st.encoder_image.flush();

        if (st.encoder_audio.is_open())
            st.encoder_audio.flush();

        int ret = av_write_trailer(st.pFormatCtx.get());
        if (ret < 0)
            printf("AV : failed to write trailer : `%s`\n", get_av_error(ret).c_str());

        if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
            avio_closep(&st.pFormatCtx->pb);

        st.pFormatCtx.reset(nullptr); //close
        st.connected = false;
    }

    bool muxer_ffmpeg::handle_packet(AVPacket* pkt, AVCodecContext* pCodecCtx)
    {
        AVStream* stream = pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO ? st.stream_image : st.stream_audio;
        av_packet_rescale_ts(pkt, pCodecCtx->time_base, stream->time_base);
        pkt->stream_index = stream->index;
        int ret = av_interleaved_write_frame(st.pFormatCtx.get(), pkt);
        if (ret < 0)
            printf("av_interleaved_write_frame() failed : `%s`\n", get_av_error(ret).c_str());
        return ret == 0;
    }

    int muxer_ffmpeg::height() const
    {
        return st.encoder_image.height();
    }

    int muxer_ffmpeg::width() const
    {
        return st.encoder_image.width();
    }

    AVPixelFormat muxer_ffmpeg::pixel_fmt() const
    {
        return st.encoder_image.pixel_fmt();
    }

    int muxer_ffmpeg::sample_rate() const
    {
        return st.encoder_audio.sample_rate();
    }

    uint64_t muxer_ffmpeg::channel_layout() const
    {
        return st.encoder_audio.channel_layout();
    }

    int muxer_ffmpeg::nchannels() const
    {
        return st.encoder_audio.nchannels();
    }

    AVSampleFormat muxer_ffmpeg::sample_fmt() const
    {
        return st.encoder_audio.sample_fmt();
    }
}