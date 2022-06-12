#include <thread>
#include <chrono>
#include "video_muxer.h"

using namespace std::chrono;

namespace dlib
{
    Frame dlib_image_to_frame(
        const array2d<rgb_pixel>& frame,
        std::chrono::system_clock::time_point timestamp
    )
    {
        Frame f = Frame::make_image(frame.nr(), frame.nc(), AV_PIX_FMT_RGB24, timestamp);

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
        std::chrono::system_clock::time_point timestamp
    )
    {
        Frame f = Frame::make_audio(frame.sample_rate, frame.samples.size(), AV_CH_LAYOUT_STEREO, AV_SAMPLE_FMT_S16, timestamp);
        memcpy(f.frame->data[0], frame.samples.data(), frame.samples.size()*sizeof(audio_frame::sample));
        return f;
    }

    encoder_ffmpeg::encoder_ffmpeg(
        const args &a,
        std::shared_ptr<std::ostream> out
    ) : _args(a),
        encoded(std::move(out))
    {
        if (!open())
            pCodecCtx = nullptr;
    }

    encoder_ffmpeg::encoder_ffmpeg(
        const args &a,
        packet_callback clb
    ) : _args(a),
        packet_ready_callback(std::move(clb))
    {
        if (!open())
            pCodecCtx = nullptr;
    }

    encoder_ffmpeg::~encoder_ffmpeg()
    {
        flush();
    }

    bool encoder_ffmpeg::open()
    {
        DLIB_CASSERT(packet_ready_callback != nullptr || encoded != nullptr, "Empty std::shared_ptr<std::ostream>");

        packet = make_avpacket();
        AVCodec* pCodec = nullptr;

        if (_args.args_common.codec != AV_CODEC_ID_NONE)
            pCodec = avcodec_find_encoder(_args.args_common.codec);
        else if (!_args.args_common.codec_name.empty())
            pCodec = avcodec_find_encoder_by_name(_args.args_common.codec_name.c_str());

        if (!pCodec)
        {
            printf("Codec `%s` or `%s` not found\n", avcodec_get_name(_args.args_common.codec), _args.args_common.codec_name.c_str());
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

            if (pCodecCtx->codec_id == AV_CODEC_ID_AAC) {
                pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
            }

            //don't know what src options are, but at least dst options are set
            resizer_audio.reset(
                    pCodecCtx->sample_rate, pCodecCtx->channel_layout, pCodecCtx->sample_fmt,
                    pCodecCtx->sample_rate, pCodecCtx->channel_layout, pCodecCtx->sample_fmt
            );
        }

        av_dict opt = _args.args_common.codec_options;
        const int ret = avcodec_open2(pCodecCtx.get(), pCodec, opt.get());
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

    bool encoder_ffmpeg::is_open() const noexcept
    {
        return  pCodecCtx != nullptr &&
                (encoded != nullptr || packet_ready_callback != nullptr) &&
                !flushed &&
                FFMPEG_INITIALIZED;
    }

    bool encoder_ffmpeg::is_image_encoder() const noexcept
    {
        return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
    }

    bool encoder_ffmpeg::is_audio_encoder() const noexcept
    {
        return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO;
    }

    AVCodecID encoder_ffmpeg::get_codec_id() const noexcept
    {
        return pCodecCtx ? pCodecCtx->codec_id : AV_CODEC_ID_NONE;
    }

    std::string encoder_ffmpeg::get_codec_name() const noexcept
    {
        return pCodecCtx ? avcodec_get_name(pCodecCtx->codec_id) : "NONE";
    }

    std::shared_ptr<std::ostream> encoder_ffmpeg::get_encoded_stream() const noexcept
    {
        return encoded;
    }

    typedef enum {
        ENCODING_SEND_FRAME = 0,
        ENCODING_RECV_PACKET,
        ENCODING_RECV_PACKET_SEND_FRAME,
        ENCODING_DONE,
        ENCODING_ERROR = -1
    } encoding_state;

    bool encoder_ffmpeg::push(Frame&& frame)
    {
        using namespace std::chrono;

        if (!is_open())
            return false;

        std::vector<Frame> frames;

        /*! Resize if image. Resample if audio. Push through audio fifo if necessary (audio codec requires fixed size blocks) !*/
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

        /*! Set pts based on timestamps or tracked state !*/
        for (auto& frame : frames)
        {
            if (frame.timestamp != std::chrono::system_clock::time_point{})
            {
                frame.frame->pts = av_rescale_q(
                        frame.timestamp.time_since_epoch().count(),
                        {nanoseconds::period::num,nanoseconds::period::den},
                        pCodecCtx->time_base);
            }
            else if (frame.frame)
            {
                frame.frame->pts = next_pts;
            }

            if (frame.frame)
                next_pts = frame.frame->pts + (frame.is_image() ? 1 : frame.frame->nb_samples);
        }

        encoding_state state = ENCODING_SEND_FRAME;

        auto send_frame = [&](Frame& frame)
        {
            const int ret = avcodec_send_frame(pCodecCtx.get(), frame.frame.get());

            if (ret == AVERROR_EOF)
                state = ENCODING_DONE;
            else if (ret == AVERROR(EAGAIN))
                state = ENCODING_RECV_PACKET_SEND_FRAME;
            else if (ret >= 0)
            {
                state = ENCODING_RECV_PACKET;
            }
            else
            {
                printf("avcodec_send_frame() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                state = ENCODING_ERROR;
            }
        };

        auto recv_packet = [&]
        {
            const int ret = avcodec_receive_packet(pCodecCtx.get(), packet.get());

            if (ret == AVERROR_EOF) {
                state   = ENCODING_DONE;
                flushed = true;
            }
            else if (ret == AVERROR(EAGAIN) && state == ENCODING_RECV_PACKET)
                state = ENCODING_DONE;
            else if (ret == AVERROR(EAGAIN) && state == ENCODING_RECV_PACKET_SEND_FRAME)
                state = ENCODING_SEND_FRAME;
            else if (ret < 0)
            {
                printf("avcodec_receive_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                state = ENCODING_ERROR;
            }
            else if (ret >= 0)
            {
                if (packet_ready_callback)
                {
                    /*! Invoke muxer callback !*/
                    if (!packet_ready_callback(packet.get(), pCodecCtx.get()))
                        state = ENCODING_ERROR;
                }
                else
                {
                    encoded->write((char*)packet->data, packet->size);
                }
            }
        };

        for (size_t i = 0 ; i < frames.size() && state != ENCODING_ERROR ; ++i)
        {
            state = ENCODING_SEND_FRAME;

            while (state != ENCODING_DONE && state != ENCODING_ERROR)
            {
                switch(state)
                {
                    case ENCODING_SEND_FRAME:               send_frame(frames[i]);   break;
                    case ENCODING_RECV_PACKET:              recv_packet();  break;
                    case ENCODING_RECV_PACKET_SEND_FRAME:   recv_packet();  break;
                    default: break;
                }
            }
        }

        return state != ENCODING_ERROR;
    }

    bool encoder_ffmpeg::push(
        const array2d<rgb_pixel> &frame,
        std::chrono::system_clock::time_point timestamp
    )
    {
        DLIB_CASSERT(is_image_encoder(), "This object is either empty or doesn't represent an image/video encoder");
        return push(dlib_image_to_frame(frame, timestamp));
    }

    bool encoder_ffmpeg::push(
        const audio_frame &frame,
        std::chrono::system_clock::time_point timestamp
    )
    {
        DLIB_CASSERT(is_audio_encoder(), "This object is either empty or doesn't represent an audio encoder");
        return push(dlib_audio_to_frame(frame, timestamp));
    }

    void encoder_ffmpeg::flush()
    {
        if (!flushed)
            push(Frame{});
    }

    int encoder_ffmpeg::height() const noexcept
    {
        return is_image_encoder() ? pCodecCtx->height : 0;
    }

    int encoder_ffmpeg::width() const noexcept
    {
        return is_image_encoder() ? pCodecCtx->width : 0;
    }

    AVPixelFormat encoder_ffmpeg::pixel_fmt() const noexcept
    {
        return is_image_encoder() ? pCodecCtx->pix_fmt : AV_PIX_FMT_NONE;
    }

    int encoder_ffmpeg::sample_rate() const noexcept
    {
        return is_audio_encoder() ? pCodecCtx->sample_rate : 0;
    }

    uint64_t encoder_ffmpeg::channel_layout() const noexcept
    {
        return is_audio_encoder() ? pCodecCtx->channel_layout : 0;
    }

    int encoder_ffmpeg::nchannels() const noexcept
    {
        return is_audio_encoder() ? av_get_channel_layout_nb_channels(pCodecCtx->channel_layout) : 0;
    }

    AVSampleFormat encoder_ffmpeg::sample_fmt() const noexcept
    {
        return is_audio_encoder() ? pCodecCtx->sample_fmt : AV_SAMPLE_FMT_NONE;
    }

    muxer_ffmpeg::muxer_ffmpeg(const args &a)
    {
        st._args = a;
        if (!open())
            st.pFormatCtx = nullptr;
    }

    void muxer_ffmpeg::set_av_opaque_pointers()
    {
        if (st.pFormatCtx)
            st.pFormatCtx->opaque = this;
        st.encoder_image.packet_ready_callback = [this](AVPacket* pkt, AVCodecContext* ctx){return handle_packet(pkt, ctx);};
        st.encoder_image.packet_ready_callback = [this](AVPacket* pkt, AVCodecContext* ctx){return handle_packet(pkt, ctx);};
    }

    muxer_ffmpeg::muxer_ffmpeg(muxer_ffmpeg &&other) noexcept
    : st{std::move(other.st)}
    {
        set_av_opaque_pointers();
    }

    muxer_ffmpeg& muxer_ffmpeg::operator=(muxer_ffmpeg &&other) noexcept
    {
        flush();
        st = std::move(other.st);
        set_av_opaque_pointers();
        return *this;
    }

    muxer_ffmpeg::~muxer_ffmpeg()
    {
        flush();
    }

    bool muxer_ffmpeg::is_open() const noexcept
    {
        return video_enabled() || audio_enabled();
    }

    bool muxer_ffmpeg::video_enabled() const noexcept
    {
        return st.pFormatCtx != nullptr && st.encoder_image.is_open() && FFMPEG_INITIALIZED;
    }

    bool muxer_ffmpeg::audio_enabled() const noexcept
    {
        return st.pFormatCtx != nullptr && st.encoder_audio.is_open() && FFMPEG_INITIALIZED;
    }

    bool muxer_ffmpeg::open()
    {
        if (!st._args.enable_audio && !st._args.enable_image)
        {
            printf("You need to set at least one of `enable_audio` or `enable_image`\n");
            return false;
        }

        {
            st.connecting_time = system_clock::now();
            st.connected_time  = system_clock::time_point::max();

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

        st.pFormatCtx->opaque = this;
        st.pFormatCtx->interrupt_callback.opaque    = st.pFormatCtx.get();
        st.pFormatCtx->interrupt_callback.callback  = [](void* ctx) -> int {
            AVFormatContext* pFormatCtx = (AVFormatContext*)ctx;
            muxer_ffmpeg* me = (muxer_ffmpeg*)pFormatCtx->opaque;
            return me->interrupt_callback();
        };

        if (st._args.max_delay > 0)
            st.pFormatCtx->max_delay = st._args.max_delay;

        //st.pFormatCtx->flags = AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;

        if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
        {
            av_dict opt = st._args.protocol_options;

            int ret = avio_open2(&st.pFormatCtx->pb, st._args.filepath.c_str(), AVIO_FLAG_WRITE, &st.pFormatCtx->interrupt_callback, opt.get());

            if (ret < 0)
            {
                printf("avio_open2() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }
        }

        av_dict opt = st._args.format_options;

        int ret = avformat_write_header(st.pFormatCtx.get(), opt.get());
        if (ret < 0)
        {
            printf("avformat_write_header() failed : `%s`\n", get_av_error(ret).c_str());
            return false;
        }

        st.connected_time = system_clock::now();

        return true;
    }

    bool muxer_ffmpeg::interrupt_callback()
    {
        const auto now = system_clock::now();

        if (st._args.connect_timeout < std::chrono::milliseconds::max())
            if (st.connected_time > now && now > (st.connecting_time + st._args.connect_timeout))
                return true;

        if (st._args.interrupter && st._args.interrupter())
            return true;

        return false;
    }

    bool muxer_ffmpeg::push(
        const array2d<rgb_pixel> &frame,
        std::chrono::system_clock::time_point timestamp
    )
    {
        if (!is_open())
            return false;

        if (!st.encoder_image.is_open())
        {
            printf("frame is an image type but image encoder is not initialized\n");
            return false;
        }

        return st.encoder_image.push(frame, timestamp);
    }

    bool muxer_ffmpeg::push(
        const audio_frame &frame,
        std::chrono::system_clock::time_point timestamp
    )
    {
        if (!is_open())
            return false;

        if (!st.encoder_audio.is_open())
        {
            printf("frame is of audio type but audio encoder is not initialized\n");
            return false;
        }

        return st.encoder_audio.push(frame, timestamp);
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

            return st.encoder_image.push(std::move(frame));
        }

        else if (frame.is_image())
        {
            if (!st.encoder_audio.is_open())
            {
                printf("frame is of audio type but audio encoder is not initialized\n");
                return false;
            }

            return st.encoder_audio.push(std::move(frame));
        }

        return false;
    }

    void muxer_ffmpeg::flush()
    {
        if (!is_open())
            return;

        st.encoder_image.flush();
        st.encoder_audio.flush();

        const int ret = av_write_trailer(st.pFormatCtx.get());
        if (ret < 0)
            printf("AV : failed to write trailer : `%s`\n", get_av_error(ret).c_str());

        if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
            avio_closep(&st.pFormatCtx->pb);

        st.pFormatCtx.reset(nullptr); //close
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