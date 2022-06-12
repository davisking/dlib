#include <chrono>
#include <thread>
#include "video_demuxer.h"
#include "../string.h"

using namespace std::chrono;

namespace dlib
{
    array2d<rgb_pixel> frame_to_dlib_image(
        Frame& f,
        sw_image_resizer& resizer
    )
    {
        if (f.pixfmt() != AV_PIX_FMT_RGB24)
        {
            resizer.resize(
                    f,
                    resizer.get_dst_h(),
                    resizer.get_dst_w(),
                    AV_PIX_FMT_RGB24,
                    f);
        }

        array2d<rgb_pixel> frame_image(f.frame->height, f.frame->width);

        for (int row = 0 ; row < f.frame->height ; row++)
        {
            memcpy(frame_image.begin() + row * f.frame->width,
                   f.frame->data[0] + row * f.frame->linesize[0],
                   f.frame->width*3);
        }

        return frame_image;
    }

    audio_frame frame_to_dlib_audio(
        Frame& f,
        sw_audio_resampler& resizer
    )
    {
        if (f.frame->channel_layout != AV_CH_LAYOUT_STEREO ||
            f.samplefmt() != AV_SAMPLE_FMT_S16)
        {
            resizer.resize(
                    f,
                    f.frame->sample_rate,
                    AV_CH_LAYOUT_STEREO,
                    AV_SAMPLE_FMT_S16,
                    f);
        }
        audio_frame frame_audio;
        frame_audio.sample_rate = f.frame->sample_rate;
        frame_audio.samples.resize(f.frame->nb_samples);
        memcpy(frame_audio.samples.data(), f.frame->data[0], frame_audio.samples.size()*sizeof(audio_frame::sample));

        return frame_audio;
    }

    decoder_ffmpeg::decoder_ffmpeg(const args& a)
    : _args(a)
    {
        if (!open())
            pCodecCtx = nullptr;
    }

    bool decoder_ffmpeg::open()
    {
        packet = make_avpacket();
        frame  = make_avframe();
        AVCodec* pCodec = nullptr;

        if (_args.args_common.codec != AV_CODEC_ID_NONE)
            pCodec = avcodec_find_decoder(_args.args_common.codec);
        else if (!_args.args_common.codec_name.empty())
            pCodec = avcodec_find_decoder_by_name(_args.args_common.codec_name.c_str());

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

        if (pCodecCtx->codec_id == AV_CODEC_ID_AAC)
            pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

        if (pCodecCtx->codec_id == AV_CODEC_ID_PCM_S16LE)
        {
            /*!
                Raw codecs require you to set parameters rather than reading it from the encoded stream.
                Indeed, there is no such metadata within the stream
            !*/
            DLIB_CASSERT(_args.args_audio.sample_rate > 0, "raw encoders require you to set sample rate manually");
            DLIB_CASSERT(_args.args_audio.channel_layout > 0, "raw encoder require you to set channel layout manually");
            pCodecCtx->sample_rate      = _args.args_audio.sample_rate;
            pCodecCtx->channel_layout   = _args.args_audio.channel_layout;
            pCodecCtx->channels         = av_get_channel_layout_nb_channels(_args.args_audio.channel_layout);
            pCodecCtx->sample_fmt       = AV_SAMPLE_FMT_S16;
        }

        av_dict opt = _args.args_common.codec_options;
        int ret = avcodec_open2(pCodecCtx.get(), pCodec, opt.get());
        if (ret < 0)
        {
            printf("avcodec_open2() failed : `%s`\n", get_av_error(ret).c_str());
            return false;
        }

        const bool no_parser_required = pCodecCtx->codec_id == AV_CODEC_ID_PCM_S16LE;

        if (!no_parser_required)
        {
            parser.reset(av_parser_init(pCodec->id));
            if (!parser)
            {
                printf("AV : parser for codec `%s` not found\n", pCodec->name);
                return false;
            }
        }

        return true;
    }

    bool decoder_ffmpeg::is_open() const noexcept
    {
        if (!src_frame_buffer.empty())
            return true;

        return FFMPEG_INITIALIZED && pCodecCtx != nullptr && !flushed;
    }

    bool decoder_ffmpeg::is_image_decoder() const noexcept
    {
        return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
    }

    bool decoder_ffmpeg::is_audio_decoder() const noexcept
    {
        return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO;
    }

    AVCodecID decoder_ffmpeg::get_codec_id() const noexcept
    {
        return pCodecCtx ? pCodecCtx->codec_id : AV_CODEC_ID_NONE;
    }

    std::string decoder_ffmpeg::get_codec_name() const noexcept
    {
        return pCodecCtx ? avcodec_get_name(pCodecCtx->codec_id) : "NONE";
    }

    int decoder_ffmpeg::height() const noexcept
    {
        return resizer_image.get_dst_h();
    }

    int decoder_ffmpeg::width() const noexcept
    {
        return resizer_image.get_dst_w();
    }

    AVPixelFormat decoder_ffmpeg::pixel_fmt() const noexcept
    {
        return resizer_image.get_dst_fmt();
    }

    int decoder_ffmpeg::sample_rate() const noexcept
    {
        return resizer_audio.get_dst_rate();
    }

    uint64_t decoder_ffmpeg::channel_layout() const noexcept
    {
        return resizer_audio.get_dst_layout();
    }

    AVSampleFormat decoder_ffmpeg::sample_fmt() const noexcept
    {
        return resizer_audio.get_dst_fmt();
    }

    int decoder_ffmpeg::nchannels() const noexcept
    {
        return av_get_channel_layout_nb_channels(channel_layout());
    }

    /*!
        We use a state machine to control decoding.
        State machines are awesome.
        Compile-time checked state machines are even better.
        Maybe at some point dlib could have one.
    !*/

    typedef enum {
        DECODING_RECV_PACKET = 0,
        DECODING_SEND_PACKET,
        DECODING_READ_FRAME_SEND_PACKET,
        DECODING_READ_FRAME_RECV_PACKET,
        DECODING_DONE,
        DECODING_ERROR = -1
    } decoder_state;

    bool decoder_ffmpeg::push_encoded(const uint8_t* encoded, int nencoded)
    {
        using namespace std::chrono;

        if (!pCodecCtx)
            return false;

        if (encoded && nencoded > 0)
        {
            /*! According to FFMPEG docs we need this padding because of SIMD. !*/
            encoded_buffer.resize(nencoded + AV_INPUT_BUFFER_PADDING_SIZE, 0);
            std::memcpy(&encoded_buffer[0], encoded, nencoded);
            encoded = &encoded_buffer[0];
        }

        decoder_state state     = DECODING_RECV_PACKET;
        const bool is_flushing  = encoded == nullptr && nencoded == 0;

        auto recv_packet = [&]
        {
            if (!is_flushing && nencoded == 0)
            {
                /*! Consumed all encoded data. Exit loop. !*/
                state = DECODING_DONE;
            }
            else if (parser)
            {
                const int ret = av_parser_parse2(
                        parser.get(),
                        pCodecCtx.get(),
                        &packet->data,
                        &packet->size,
                        encoded,
                        nencoded,
                        AV_NOPTS_VALUE,
                        AV_NOPTS_VALUE,
                        0
                );

                if (ret < 0)
                {
                    printf("AV : error while parsing encoded buffer\n");
                    state = DECODING_ERROR;
                }
                else
                {
                    encoded     += ret;
                    nencoded    -= ret;
                }
            } else
            {
                /*! Codec does not require parser !*/
                packet->data = const_cast<uint8_t *>(encoded);
                packet->size = nencoded;
                encoded      += nencoded;
                nencoded     = 0;
            }

            if (packet->size || is_flushing)
                state = DECODING_SEND_PACKET;
        };

        auto send_packet = [&]
        {
            const int ret = avcodec_send_packet(pCodecCtx.get(), packet.get());

            if (ret == AVERROR_EOF)
                state = DECODING_DONE;
            else if (ret == AVERROR(EAGAIN))
                state = DECODING_READ_FRAME_SEND_PACKET;
            else if (ret >= 0)
                state = DECODING_READ_FRAME_RECV_PACKET;
            else
            {
                printf("avcodec_send_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                state = DECODING_ERROR;
            }
        };

        auto recv_frame = [&]
        {
            const int ret = avcodec_receive_frame(pCodecCtx.get(), frame.get());

            if (ret == AVERROR_EOF) {
                state   = DECODING_DONE;
                flushed = true;
            } else if (ret == AVERROR(EAGAIN) && state == DECODING_READ_FRAME_SEND_PACKET)
                state = DECODING_SEND_PACKET;
            else if (ret == AVERROR(EAGAIN) && state == DECODING_READ_FRAME_RECV_PACKET)
                state = DECODING_RECV_PACKET;
            else if (ret < 0)
            {
                printf("avcodec_receive_frame() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                state = DECODING_ERROR;
            }
            else
            {
                const bool is_video         = pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
                const AVRational tb         = is_video ? pCodecCtx->time_base : AVRational{1, frame->sample_rate};
                const uint64_t pts          = is_video ? frame->pts : next_pts;
                const uint64_t timestamp_ns = av_rescale_q(pts, tb, {1,1000000000});
                next_pts                    += is_video ? 1 : frame->nb_samples;

                Frame decoded;
                Frame src;
                src.frame       = std::move(frame); //make sure you move it back when you're done
                src.timestamp   = system_clock::time_point{nanoseconds {timestamp_ns}};

                if (src.is_image())
                {
                    resizer_image.resize(
                            src,
                            _args.args_image.h > 0 ? _args.args_image.h : src.frame->height,
                            _args.args_image.w > 0 ? _args.args_image.w : src.frame->width,
                            _args.args_image.fmt != AV_PIX_FMT_NONE ? _args.args_image.fmt : src.pixfmt(),
                            decoded);
                }
                else
                {
                    resizer_audio.resize(
                            src,
                            _args.args_audio.sample_rate > 0            ? _args.args_audio.sample_rate      : src.frame->sample_rate,
                            _args.args_audio.channel_layout > 0         ? _args.args_audio.channel_layout   : src.frame->channel_layout,
                            _args.args_audio.fmt != AV_SAMPLE_FMT_NONE  ? _args.args_audio.fmt              : src.samplefmt(),
                            decoded);
                }

                frame = std::move(src.frame);
                src_frame_buffer.push(std::move(decoded));
            }
        };

        while (state != DECODING_ERROR && state != DECODING_DONE)
        {
            switch(state)
            {
                case DECODING_RECV_PACKET:              recv_packet(); break;
                case DECODING_SEND_PACKET:              send_packet(); break;
                case DECODING_READ_FRAME_SEND_PACKET:   recv_frame(); break;
                case DECODING_READ_FRAME_RECV_PACKET:   recv_frame(); break;
                default: break;
            }
        }

        return state != DECODING_ERROR;
    }

    void decoder_ffmpeg::flush()
    {
        push_encoded(nullptr, 0);
    }

    decoder_ffmpeg::suc_t decoder_ffmpeg::read(Frame& dst_frame)
    {
        if (!src_frame_buffer.empty())
        {
            dst_frame = std::move(src_frame_buffer.front());
            src_frame_buffer.pop();
            return FRAME_AVAILABLE;
        }

        if (!is_open())
            return CLOSED;

        return MORE_INPUT;
    }

    decoder_ffmpeg::suc_t decoder_ffmpeg::read(
        type_safe_union<array2d<rgb_pixel>, audio_frame> &frame,
        std::chrono::system_clock::time_point &timestamp
    )
    {
        Frame f;
        const auto suc = read(f);

        if (suc == FRAME_AVAILABLE)
        {
            if (f.is_image())
            {
                frame = frame_to_dlib_image(f, resizer_image);
            }
            else if (f.is_audio())
            {
                frame = frame_to_dlib_audio(f, resizer_audio);
            }

            timestamp = f.timestamp;
        }

        return suc;
    }

    demuxer_ffmpeg::args::args(
        std::string filepath_
    ) : filepath(std::move(filepath_))
    {
    }

    demuxer_ffmpeg::demuxer_ffmpeg(const args& a)
    {
        if (!open(a))
            st.pFormatCtx = nullptr;
    }

    demuxer_ffmpeg::demuxer_ffmpeg(demuxer_ffmpeg &&other) noexcept
    : st{std::move(other.st)}
    {
        if (st.pFormatCtx)
            st.pFormatCtx->opaque = this;
    }

    demuxer_ffmpeg& demuxer_ffmpeg::operator=(demuxer_ffmpeg &&other) noexcept
    {
        st = std::move(other.st);
        if (st.pFormatCtx)
            st.pFormatCtx->opaque = this;
        return *this;
    }

    void demuxer_ffmpeg::populate_metadata()
    {
        for (unsigned int i = 0; i < st.pFormatCtx->nb_streams; i++)
        {
            std::string metadata_str;
            {
                char* charbuf = 0;
                av_dict_get_string(st.pFormatCtx->streams[i]->metadata, &charbuf, ',', ';');
                metadata_str = std::string(charbuf);
                free(charbuf);
            }

            std::vector<std::string> keyvals = dlib::split(metadata_str, ";");
            for (size_t kv = 0; kv < keyvals.size(); kv++) {
                std::vector<std::string> kv_item = dlib::split(keyvals[kv], ",");
                assert(kv_item.size() == 2);
                st.metadata[i][kv_item[0]] = dlib::trim(kv_item[1]);
            }
        }
    }

    bool demuxer_ffmpeg::open(const args& a)
    {
        st = {};
        st._args = a;

        AVFormatContext* pFormatCtx = avformat_alloc_context();
        pFormatCtx->opaque = this;
        pFormatCtx->interrupt_callback.opaque   = pFormatCtx;
        pFormatCtx->interrupt_callback.callback = [](void* ctx) -> int {
            AVFormatContext* pFormatCtx = (AVFormatContext*)ctx;
            demuxer_ffmpeg* me = (demuxer_ffmpeg*)pFormatCtx->opaque;
            return me->interrupt_callback();
        };

        if (st._args.probesize > 0)
            pFormatCtx->probesize = st._args.probesize;

        av_dict opts = st._args.format_options;
        AVInputFormat* input_format = st._args.input_format.empty() ? nullptr : av_find_input_format(st._args.input_format.c_str());

        st.connecting_time = system_clock::now();
        st.connected_time  = system_clock::time_point::max();

        int ret = avformat_open_input(&pFormatCtx,
                                      st._args.filepath.c_str(),
                                      input_format,
                                      opts.get());

        if (ret != 0)
        {
            printf("avformat_open_input() failed with error `%s`\n", get_av_error(ret).c_str());
            return false;
        }

        st.connected_time = system_clock::now();
        st.pFormatCtx.reset(pFormatCtx);

        ret = avformat_find_stream_info(st.pFormatCtx.get(), NULL);

        if (ret < 0)
        {
            printf("avformat_find_stream_info() failed with error `%s`\n", get_av_error(ret).c_str());
            return false;
        }

        auto setup_stream = [&](bool is_video, channel& ch) -> bool
        {
            const args::channel_args& common    = is_video ? st._args.image_options.common : st._args.audio_options.common;
            const AVMediaType media_type        = is_video ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;

            AVCodec* pCodec = 0;
            ch.stream_id = av_find_best_stream(st.pFormatCtx.get(), media_type, -1, -1, &pCodec, 0);

            if (ch.stream_id == AVERROR_STREAM_NOT_FOUND)
            {
                ch = channel{}; //reset
                return true; //You might be asking for both video and audio but only video is available. That's OK. Just provide video.
            }

            else if (ch.stream_id == AVERROR_DECODER_NOT_FOUND)
            {
                printf("av_find_best_stream() : decoder not found for stream type `%s`\n", av_get_media_type_string(media_type));
                return false;
            }

            else if (ch.stream_id < 0)
            {
                printf("av_find_best_stream() failed : `%s`\n", get_av_error(ch.stream_id).c_str());
                return false;
            }

            /* create decoding context */
            ch.pCodecCtx.reset(avcodec_alloc_context3(pCodec));
            if (!ch.pCodecCtx)
            {
                printf("avcodec_alloc_context3() failed\n");
                return false;
            }

            int ret = avcodec_parameters_to_context(ch.pCodecCtx.get(), st.pFormatCtx->streams[ch.stream_id]->codecpar);
            if (ret < 0)
            {
                printf("avcodec_parameters_to_context() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }

            if (common.nthreads > 0)
                ch.pCodecCtx->thread_count = common.nthreads;

            av_dict opt = common.codec_options;
            ret = avcodec_open2(ch.pCodecCtx.get(), pCodec, opt.get());
            if (ret < 0)
            {
                printf("avcodec_open2() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }

            if (ch.pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                if (ch.pCodecCtx->height   == 0 ||
                    ch.pCodecCtx->width    == 0 ||
                    ch.pCodecCtx->pix_fmt  == AV_PIX_FMT_NONE)
                {
                    printf("Codec parameters look wrong : (h,w,pixel_fmt) : (%i,%i,%s)\n",
                             ch.pCodecCtx->height,
                             ch.pCodecCtx->width,
                             get_pixel_fmt_str(ch.pCodecCtx->pix_fmt).c_str());
                    return false;
                }

                ch.resizer_image.reset(
                        ch.pCodecCtx->height,
                        ch.pCodecCtx->width,
                        ch.pCodecCtx->pix_fmt,
                        st._args.image_options.h > 0 ? st._args.image_options.h : ch.pCodecCtx->height,
                        st._args.image_options.w > 0 ? st._args.image_options.w : ch.pCodecCtx->width,
                        st._args.image_options.fmt != AV_PIX_FMT_NONE ? st._args.image_options.fmt : ch.pCodecCtx->pix_fmt
                );
            }
            else if (ch.pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO)
            {
                if (ch.pCodecCtx->sample_rate == 0 ||
                    ch.pCodecCtx->sample_fmt  == AV_SAMPLE_FMT_NONE)
                {
                    printf("Codec parameters look wrong: sample_rate : %i sample_fmt : %i\n",
                             ch.pCodecCtx->sample_rate,
                             ch.pCodecCtx->sample_fmt);
                    return false;
                }

                if (ch.pCodecCtx->channel_layout == 0)
                    ch.pCodecCtx->channel_layout = av_get_default_channel_layout(ch.pCodecCtx->channels);

                ch.resizer_audio.reset(
                        ch.pCodecCtx->sample_rate,
                        ch.pCodecCtx->channel_layout,
                        ch.pCodecCtx->sample_fmt,
                        st._args.audio_options.sample_rate > 0            ? st._args.audio_options.sample_rate      : ch.pCodecCtx->sample_rate,
                        st._args.audio_options.channel_layout > 0         ? st._args.audio_options.channel_layout   : ch.pCodecCtx->channel_layout,
                        st._args.audio_options.fmt != AV_SAMPLE_FMT_NONE  ? st._args.audio_options.fmt              : ch.pCodecCtx->sample_fmt
                );
            }
            else
            {
                printf("Unrecognized media type %i\n", ch.pCodecCtx->codec_type);
                return false;
            }

            return true;
        };

        if (st._args.enable_image)
        {
            if (!setup_stream(true, st.channel_video))
                return false;
        }

        if (st._args.enable_audio)
        {
            if (!setup_stream(false, st.channel_audio))
                return false;
        }

        if (!st.channel_audio.is_enabled() && !st.channel_video.is_enabled())
        {
            printf("At least one of video and audio channels must be enabled\n");
            return false;
        }

        st.packet = make_avpacket();
        st.frame  = make_avframe();

        if (!st.packet || !st.frame)
            return false;

        populate_metadata();

        return true;
    }

    typedef enum {
        DEMUXER_RECV_PACKET = 0,
        DEMUXER_SEND_PACKET,
        DEMUXER_RECV_FRAME_SEND_PACKET,
        DEMUXER_RECV_FRAME_RECV_PACKET,
        DEMUXER_DONE,
        DEMUXER_FLUSH,
        DEMUXER_ERROR = -1
    } demuxer_state;

    bool demuxer_ffmpeg::fill_decoded_buffer()
    {
        using namespace std::chrono;

        if (!st.src_frame_buffer.empty())
            return true;

        if (!is_open())
            return false;

        demuxer_state state = DEMUXER_RECV_PACKET;
        channel* ch         = nullptr;

        auto recv_packet = [&]
        {
            if (!st.src_frame_buffer.empty())
            {
                state = DEMUXER_DONE;
            }
            else
            {
                av_packet_unref(st.packet.get());
                const int ret = av_read_frame(st.pFormatCtx.get(), st.packet.get());

                if (ret == AVERROR_EOF)
                {
                    state = DEMUXER_FLUSH;
                }
                else if (ret >= 0)
                {
                    if (st.packet->stream_index == st.channel_video.stream_id ||
                        st.packet->stream_index == st.channel_audio.stream_id)
                    {
                        ch      = st.packet->stream_index == st.channel_video.stream_id ? &st.channel_video : &st.channel_audio;
                        state   = DEMUXER_SEND_PACKET;
                    }
                }
                else
                {
                    printf("av_read_frame() failed : `%s`\n", get_av_error(ret).c_str());
                    state = DEMUXER_ERROR;
                }
            }
        };

        auto send_packet = [&]
        {
            const int ret = avcodec_send_packet(ch->pCodecCtx.get(), st.packet.get());

            if (ret == AVERROR_EOF)
                state = DEMUXER_DONE;
            else if (ret == AVERROR(EAGAIN))
                state = DEMUXER_RECV_FRAME_SEND_PACKET;
            else if (ret >= 0)
                state = DEMUXER_RECV_FRAME_RECV_PACKET;
            else
            {
                printf("avcodec_send_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                state = DEMUXER_ERROR;
            }
        };

        auto recv_frame = [&]
        {
            const int ret = avcodec_receive_frame(ch->pCodecCtx.get(), st.frame.get());

            if (ret == AVERROR_EOF)
                state = DEMUXER_DONE;
            else if (ret == AVERROR(EAGAIN) && state == DEMUXER_RECV_FRAME_SEND_PACKET)
                state = DEMUXER_SEND_PACKET;
            else if (ret == AVERROR(EAGAIN) && state == DEMUXER_RECV_FRAME_RECV_PACKET)
                state = DEMUXER_RECV_PACKET;
            else if (ret < 0)
            {
                printf("avcodec_receive_frame() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                state = DEMUXER_ERROR;
            }
            else
            {
                const bool is_video         = ch->pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
                const AVRational tb         = is_video ? st.pFormatCtx->streams[ch->stream_id]->time_base : AVRational{1, st.frame->sample_rate};
                const uint64_t pts          = is_video ? st.frame->pts : ch->next_pts;
                const uint64_t timestamp_ns = av_rescale_q(pts, tb, {1,1000000000});
                ch->next_pts                += is_video ? 1 : st.frame->nb_samples;

                Frame decoded;
                Frame src;
                src.timestamp   = system_clock::time_point{nanoseconds{timestamp_ns}};
                src.frame       = std::move(st.frame);

                if (src.is_image())
                    st.channel_video.resizer_image.resize(src, decoded);
                else
                    st.channel_audio.resizer_audio.resize(src, decoded);

                st.frame = std::move(src.frame);

                st.src_frame_buffer.push(std::move(decoded));
                st.last_read_time = system_clock::now();
            }
        };

        auto flushing_channel = [&]
        {
            state = DEMUXER_SEND_PACKET;

            while (state != DEMUXER_ERROR && state != DEMUXER_DONE)
            {
                switch(state)
                {
                    case DEMUXER_RECV_PACKET:              break;
                    case DEMUXER_SEND_PACKET:              send_packet(); break;
                    case DEMUXER_RECV_FRAME_SEND_PACKET:   recv_frame(); break;
                    case DEMUXER_RECV_FRAME_RECV_PACKET:   recv_frame(); break;
                    default: break;
                }
            }
        };

        auto flushing = [&]
        {
            st.packet = nullptr;
            if (st.channel_video.is_enabled())
            {
                ch = &st.channel_video;
                flushing_channel();
            }
            if (st.channel_audio.is_enabled())
            {
                ch = &st.channel_audio;
                flushing_channel();
            }
            st.flushed = true;
        };

        while (state != DEMUXER_ERROR && state != DEMUXER_DONE)
        {
            switch(state)
            {
                case DEMUXER_RECV_PACKET:              recv_packet(); break;
                case DEMUXER_SEND_PACKET:              send_packet(); break;
                case DEMUXER_RECV_FRAME_SEND_PACKET:   recv_frame(); break;
                case DEMUXER_RECV_FRAME_RECV_PACKET:   recv_frame(); break;
                case DEMUXER_FLUSH:                    flushing(); break;
                default: break;
            }
        }

        return state != DEMUXER_ERROR;
    }

    bool demuxer_ffmpeg::read(Frame& dst_frame)
    {
        if (!fill_decoded_buffer())
            return false;

        if (!st.src_frame_buffer.empty())
        {
            dst_frame = std::move(st.src_frame_buffer.front());
            st.src_frame_buffer.pop();
            return true;
        }

        return false;
    }

    bool demuxer_ffmpeg::read(
        type_safe_union<array2d<rgb_pixel>, audio_frame> &frame,
        std::chrono::system_clock::time_point &timestamp
    )
    {
        Frame f;

        if (!read(f))
            return false;

        if (f.is_image())
        {
            frame = frame_to_dlib_image(f, st.channel_video.resizer_image);
        }
        else if (f.is_audio())
        {
            frame = frame_to_dlib_audio(f, st.channel_audio.resizer_audio);
        }

        timestamp = f.timestamp;

        return true;
    }

    bool demuxer_ffmpeg::is_open() const noexcept
    {
        const bool frames_available = !st.src_frame_buffer.empty();
        const bool object_ok        = FFMPEG_INITIALIZED        &&
                                      st.pFormatCtx != nullptr  &&
                                      (st.channel_video.is_enabled() || st.channel_audio.is_enabled()) &&
                                      !st.flushed;
        return frames_available || object_ok;
    }

    bool demuxer_ffmpeg::video_enabled() const noexcept
    {
        return st.channel_video.is_enabled();
    }

    bool demuxer_ffmpeg::audio_enabled() const noexcept
    {
        return st.channel_audio.is_enabled();
    }

    int demuxer_ffmpeg::height() const noexcept
    {
        return st.channel_video.is_enabled() ? st.channel_video.resizer_image.get_dst_h() : 0;
    }

    int demuxer_ffmpeg::width() const noexcept
    {
        return st.channel_video.is_enabled() ? st.channel_video.resizer_image.get_dst_w() : 0;
    }

    AVPixelFormat demuxer_ffmpeg::pixel_fmt() const noexcept
    {
        return st.channel_video.is_enabled() ? st.channel_video.resizer_image.get_dst_fmt() : AV_PIX_FMT_NONE;
    }

    int demuxer_ffmpeg::estimated_nframes() const noexcept
    {
        return st.channel_video.is_enabled() ? st.pFormatCtx->streams[st.channel_video.stream_id]->nb_frames : 0;
    }

    AVCodecID demuxer_ffmpeg::get_video_codec_id() const noexcept
    {
        return st.channel_video.is_enabled() ? st.channel_video.pCodecCtx->codec_id : AV_CODEC_ID_NONE;
    }

    std::string demuxer_ffmpeg::get_video_codec_name() const noexcept
    {
        return st.channel_video.is_enabled() ? avcodec_get_name(st.channel_video.pCodecCtx->codec_id) : "NONE";
    }

    float demuxer_ffmpeg::fps() const noexcept
    {
        /*!
            Do we need to adjust _pFormatCtx->fps_probe_size ?
            Do we need to adjust _pFormatCtx->max_analyze_duration ?
        !*/
        if (st.channel_video.is_enabled() && st.pFormatCtx)
        {
            const float num = st.pFormatCtx->streams[st.channel_video.stream_id]->avg_frame_rate.num;
            const float den = st.pFormatCtx->streams[st.channel_video.stream_id]->avg_frame_rate.den;
            return num / den;
        }

        return 0.0f;
    }

    int demuxer_ffmpeg::sample_rate() const noexcept
    {
        return st.channel_audio.is_enabled() ? st.channel_audio.resizer_audio.get_dst_rate() : 0;
    }

    uint64_t demuxer_ffmpeg::channel_layout() const noexcept
    {
        return st.channel_audio.is_enabled() ? st.channel_audio.resizer_audio.get_dst_layout() : 0;
    }

    AVSampleFormat demuxer_ffmpeg::sample_fmt() const noexcept
    {
        return st.channel_audio.is_enabled() ? st.channel_audio.resizer_audio.get_dst_fmt() : AV_SAMPLE_FMT_NONE;
    }

    int demuxer_ffmpeg::nchannels() const noexcept
    {
        return st.channel_audio.is_enabled() ? av_get_channel_layout_nb_channels(channel_layout()) : 0;
    }

    AVCodecID demuxer_ffmpeg::get_audio_codec_id() const noexcept
    {
        return st.channel_audio.is_enabled() ? st.channel_audio.pCodecCtx->codec_id : AV_CODEC_ID_NONE;
    }

    std::string demuxer_ffmpeg::get_audio_codec_name() const noexcept
    {
        return st.channel_audio.is_enabled() ? avcodec_get_name(st.channel_audio.pCodecCtx->codec_id) : "NONE";
    }

    int demuxer_ffmpeg::estimated_total_samples() const noexcept
    {
        return st.channel_audio.is_enabled() ? st.pFormatCtx->streams[st.channel_audio.stream_id]->duration : 0;
    }

    float demuxer_ffmpeg::duration() const noexcept
    {
        return (float)av_rescale_q(st.pFormatCtx->duration, {1, AV_TIME_BASE}, {1, 1000000}) * 1e-6;
    }

    bool demuxer_ffmpeg::interrupt_callback()
    {
        const auto now = system_clock::now();

        if (st._args.connect_timeout < std::chrono::milliseconds::max())
            if (st.connected_time > now && now > (st.connecting_time + st._args.connect_timeout))
                return true;

        if (st._args.read_timeout < std::chrono::milliseconds::max())
            if (now > (st.last_read_time + st._args.read_timeout))
                return true;

        if (st._args.interrupter && st._args.interrupter())
            return true;

        return false;
    }

    std::map<int, std::map<std::string, std::string>> demuxer_ffmpeg::get_all_metadata() const noexcept
    {
        return st.metadata;
    }

    std::map<std::string, std::string> demuxer_ffmpeg::get_video_metadata() const noexcept
    {
        if (st.pFormatCtx &&
            st.channel_video.is_enabled() &&
            st.metadata.find(st.channel_video.stream_id) != st.metadata.end())
            return st.metadata.at(st.channel_video.stream_id);
        else
            return {};
    }

    float demuxer_ffmpeg::get_rotation_angle() const noexcept
    {
        const auto metadata = get_video_metadata();
        const auto it = metadata.find("rotate");
        return it != metadata.end() ? std::stof(it->second) : 0;
    }

    bool demuxer_ffmpeg::channel::is_enabled() const noexcept
    {
        return pCodecCtx != nullptr;
    }
}