#include <chrono>
#include "ffmpeg_demuxer.h"

using namespace std::chrono;

namespace dlib
{
    namespace details
    {
        decoder_extractor::decoder_extractor(
            const args& a,
            av_ptr<AVCodecContext> pCodecCtx_,
            const AVCodec* codec
        )
        {
            args_       = a;
            frame       = make_avframe();

            if (args_.args_codec.nthreads > 0)
                pCodecCtx_->thread_count = args_.args_codec.nthreads;
            if (args_.args_codec.bitrate > 0)
                pCodecCtx_->bit_rate = args_.args_codec.bitrate;
            if (args_.args_codec.flags > 0)
                pCodecCtx_->flags |= args_.args_codec.flags;

            av_dict opt = args_.args_codec.codec_options;
            int ret = avcodec_open2(pCodecCtx_.get(), codec, opt.get());

            if (ret >= 0)
            {
                pCodecCtx = std::move(pCodecCtx_);
            }
            else
                printf("avcodec_open2() failed : `%s`\n", get_av_error(ret).c_str());
        }

        bool decoder_extractor::is_open() const noexcept
        {
            const bool frames_cached = !frame_queue.empty();
            const bool object_alive  = pCodecCtx != nullptr;
            return FFMPEG_INITIALIZED && (frames_cached || object_alive);
        }

        bool decoder_extractor::is_image_decoder() const noexcept
        {
            return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
        }

        bool decoder_extractor::is_audio_decoder() const noexcept
        {
            return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO;
        }

        AVCodecID decoder_extractor::get_codec_id() const noexcept
        {
            return pCodecCtx ? pCodecCtx->codec_id : AV_CODEC_ID_NONE;
        }

        std::string decoder_extractor::get_codec_name() const noexcept
        {
            return pCodecCtx ? avcodec_get_name(pCodecCtx->codec_id) : "NONE";
        }

        int             decoder_extractor::height()         const noexcept { return resizer_image.get_dst_h(); }
        int             decoder_extractor::width()          const noexcept { return resizer_image.get_dst_w(); }
        AVPixelFormat   decoder_extractor::pixel_fmt()      const noexcept { return resizer_image.get_dst_fmt(); }
        int             decoder_extractor::sample_rate()    const noexcept { return resizer_audio.get_dst_rate(); }
        uint64_t        decoder_extractor::channel_layout() const noexcept { return resizer_audio.get_dst_layout(); }
        AVSampleFormat  decoder_extractor::sample_fmt()     const noexcept { return resizer_audio.get_dst_fmt(); }
        int             decoder_extractor::nchannels()      const noexcept { return av_get_channel_layout_nb_channels(channel_layout()); }

        /*!
            We use a basic state machine design.
            I would really like to use something like boost.MSM with a nice,
            compile-time checked transition table.
            Maybe dlib can have one at some point.
        !*/

        enum extract_state
        {
            EXTRACT_SEND_PACKET,
            EXTRACT_READ_FRAME_THEN_DONE,
            EXTRACT_READ_FRAME_THEN_SEND_PACKET,
            EXTRACT_DONE,
            EXTRACT_ERROR = -1
        };

        bool decoder_extractor::push(const av_ptr<AVPacket>& pkt)
        {
            const auto send_packet = [&](extract_state& state)
            {
                const int ret = avcodec_send_packet(pCodecCtx.get(), pkt.get());

                if (ret >= 0) {
                    state   = EXTRACT_READ_FRAME_THEN_DONE;
                } else if (ret == AVERROR(EAGAIN)) {
                    state   = EXTRACT_READ_FRAME_THEN_SEND_PACKET;
                } else if (ret == AVERROR_EOF) {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_DONE;
                } else {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_ERROR;
                    printf("avcodec_send_packet() failed : `%s`\n", get_av_error(ret).c_str());
                }
            };

            const auto recv_frame = [&](extract_state& state, bool resend)
            {
                const int ret = avcodec_receive_frame(pCodecCtx.get(), frame.get());

                if (ret == AVERROR(EAGAIN) && resend)
                    state   = EXTRACT_SEND_PACKET;
                else if (ret == AVERROR(EAGAIN))
                    state   = EXTRACT_DONE;
                else if (ret == AVERROR_EOF) {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_DONE;
                }
                else if (ret < 0)
                {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_ERROR;
                    printf("avcodec_receive_frame() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                }
                else
                {
                    const bool is_video         = pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
                    const AVRational tb         = is_video ? args_.time_base : AVRational{1, frame->sample_rate};
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
                            args_.args_image.h > 0 ? args_.args_image.h : src.frame->height,
                            args_.args_image.w > 0 ? args_.args_image.w : src.frame->width,
                            args_.args_image.fmt != AV_PIX_FMT_NONE ? args_.args_image.fmt : src.pixfmt(),
                            decoded);
                    }
                    else
                    {
                        resizer_audio.resize(
                            src,
                            args_.args_audio.sample_rate > 0            ? args_.args_audio.sample_rate      : src.frame->sample_rate,
                            args_.args_audio.channel_layout > 0         ? args_.args_audio.channel_layout   : src.frame->channel_layout,
                            args_.args_audio.fmt != AV_SAMPLE_FMT_NONE  ? args_.args_audio.fmt              : src.samplefmt(),
                            decoded);
                    }

                    frame = std::move(src.frame); // move back where it was
                    frame_queue.push(std::move(decoded));
                }
            };

            extract_state state = pCodecCtx ? EXTRACT_SEND_PACKET : EXTRACT_ERROR;

            while (state != EXTRACT_ERROR && state != EXTRACT_DONE)
            {
                switch(state)
                {
                    case EXTRACT_SEND_PACKET:                   send_packet(state);         break;
                    case EXTRACT_READ_FRAME_THEN_DONE:          recv_frame(state, false);   break;
                    case EXTRACT_READ_FRAME_THEN_SEND_PACKET:   recv_frame(state, true);    break;
                    default: break;
                }
            }

            return state != EXTRACT_ERROR;
        }

        decoder_status decoder_extractor::read(Frame &dst_frame)
        {
            if (!frame_queue.empty())
            {
                dst_frame = std::move(frame_queue.front());
                frame_queue.pop();
                return DECODER_FRAME_AVAILABLE;
            }

            if (!is_open())
                return DECODER_CLOSED;

            return DECODER_EAGAIN;
        }
    }

    ffmpeg_decoder::ffmpeg_decoder(const args &a)
    {
        DLIB_ASSERT(a.args_codec.codec != AV_CODEC_ID_NONE || a.args_codec.codec_name != "", "At least args_codec.codec or args_codec.codec_name must be set");
        AVCodec* pCodec = nullptr;

        if (a.args_codec.codec != AV_CODEC_ID_NONE)
            pCodec = avcodec_find_decoder(a.args_codec.codec);
        else if (!a.args_codec.codec_name.empty())
            pCodec = avcodec_find_decoder_by_name(a.args_codec.codec_name.c_str());

        if (!pCodec)
        {
            printf("Codec `%s` / `%s` not found\n", avcodec_get_name(a.args_codec.codec), a.args_codec.codec_name.c_str());
            return;
        }

        av_ptr<AVCodecContext> pCodecCtx{avcodec_alloc_context3(pCodec)};

        if (!pCodecCtx)
        {
            printf("avcodec_alloc_context3() failed to allocate codec context for `%s`\n", pCodec->name);
            return;
        }

        if (pCodecCtx->codec_id == AV_CODEC_ID_AAC)
            pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

        extractor = details::decoder_extractor{{a.args_codec, a.args_image, a.args_audio, pCodecCtx->time_base}, std::move(pCodecCtx), pCodec};
        if (!extractor.is_open())
            return;

        const bool no_parser_required = pCodec->id == AV_CODEC_ID_PCM_S16LE;

        if (!no_parser_required)
        {
            parser.reset(av_parser_init(pCodec->id));
            if (!parser)
            {
                printf("av_parser_init() failed codec `%s` not found\n", pCodec->name);
                return;
            }
        }

        packet = details::make_avpacket();
    }

    bool            ffmpeg_decoder::is_open()           const noexcept { return extractor.is_open(); }
    bool            ffmpeg_decoder::is_image_decoder()  const noexcept { return extractor.is_image_decoder(); }
    bool            ffmpeg_decoder::is_audio_decoder()  const noexcept { return extractor.is_audio_decoder(); }
    AVCodecID       ffmpeg_decoder::get_codec_id()      const noexcept { return extractor.get_codec_id(); }
    std::string     ffmpeg_decoder::get_codec_name()    const noexcept { return extractor.get_codec_name(); }
    int             ffmpeg_decoder::height()            const noexcept { return extractor.height(); }
    int             ffmpeg_decoder::width()             const noexcept { return extractor.width(); }
    AVPixelFormat   ffmpeg_decoder::pixel_fmt()         const noexcept { return extractor.pixel_fmt(); }
    int             ffmpeg_decoder::sample_rate()       const noexcept { return extractor.sample_rate(); }
    uint64_t        ffmpeg_decoder::channel_layout()    const noexcept { return extractor.channel_layout(); }
    AVSampleFormat  ffmpeg_decoder::sample_fmt()        const noexcept { return extractor.sample_fmt(); }
    int             ffmpeg_decoder::nchannels()         const noexcept { return extractor.nchannels(); }

    bool ffmpeg_decoder::push_encoded_padded(const uint8_t *encoded, int nencoded)
    {
        if (!is_open())
            return false;

        const auto parse = [&]
        {
            if (parser)
            {
                const int ret = av_parser_parse2(
                    parser.get(),
                    extractor.pCodecCtx.get(),
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
                    return false;
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

            return true;
        };

        const bool flushing = encoded == nullptr && nencoded == 0;
        bool ok = true;

        while (ok && (nencoded > 0 || flushing))
        {
            // Parse data OR flush parser
            ok = parse();

            // If data is available, decode
            if (ok && packet->size > 0)
                ok = extractor.push(packet);
            
            // If flushing, only flush parser once, so break
            if (flushing)
                break;
        }

        if (flushing)
        {
            // Flush codec
            ok = extractor.push(nullptr);
        }

        if (!ok || flushing)
        {
            extractor.pCodecCtx = nullptr; // close
        }
    
        return ok;
    }

    bool ffmpeg_decoder::push_encoded(const uint8_t *encoded, int nencoded)
    {
        bool ok = true;

        if (encoded == nullptr && nencoded == 0)
        {
            ok = push_encoded_padded(nullptr, 0);
        }
        else
        {
            if (nencoded > AV_INPUT_BUFFER_PADDING_SIZE)
            {
                const int blocksize = nencoded - AV_INPUT_BUFFER_PADDING_SIZE;

                ok = push_encoded_padded(encoded, blocksize);
                encoded  += blocksize;
                nencoded -= blocksize; // == AV_INPUT_BUFFER_PADDING_SIZE
            }

            if (ok)
            {
                encoded_buffer.resize(nencoded + AV_INPUT_BUFFER_PADDING_SIZE);
                std::memcpy(encoded_buffer.data(), encoded, nencoded);
                ok = push_encoded_padded(encoded_buffer.data(), nencoded);
            }
        }

        return ok;
    }

    void ffmpeg_decoder::flush()
    {
        push_encoded(nullptr, 0);
    }

    int ffmpeg_decoder::nframes_available() const
    {
        return extractor.frame_queue.size();
    }

    decoder_status ffmpeg_decoder::read(Frame& dst_frame)
    {
        return extractor.read(dst_frame);
    }

    // demuxer_ffmpeg::demuxer_ffmpeg(const args &a)
    // {
    //     if (!open(a))
    //         st.pFormatCtx = nullptr;
    // }

    // demuxer_ffmpeg::demuxer_ffmpeg(demuxer_ffmpeg &&other) noexcept
    // : st{std::move(other.st)}
    // {
    //     if (st.pFormatCtx)
    //         st.pFormatCtx->opaque = this;
    // }

    // demuxer_ffmpeg& demuxer_ffmpeg::operator=(demuxer_ffmpeg &&other) noexcept
    // {
    //     st = std::move(other.st);
    //     if (st.pFormatCtx)
    //         st.pFormatCtx->opaque = this;
    //     return *this;
    // }

    // bool demuxer_ffmpeg::open(const args& a)
    // {
    //     st = {};
    //     st.args_ = a;

    //     AVFormatContext* pFormatCtx = avformat_alloc_context();
    //     pFormatCtx->opaque = this;
    //     pFormatCtx->interrupt_callback.opaque   = pFormatCtx;
    //     pFormatCtx->interrupt_callback.callback = [](void* ctx) -> int {
    //         AVFormatContext* pFormatCtx = (AVFormatContext*)ctx;
    //         demuxer_ffmpeg* me = (demuxer_ffmpeg*)pFormatCtx->opaque;
    //         return me->interrupt_callback();
    //     };

    //     if (st.args_.probesize > 0)
    //         pFormatCtx->probesize = st.args_.probesize;

    //     av_dict opts = st.args_.format_options;
    //     AVInputFormat* input_format = st.args_.input_format.empty() ? nullptr : av_find_input_format(st.args_.input_format.c_str());

    //     st.connecting_time = system_clock::now();
    //     st.connected_time  = system_clock::time_point::max();

    //     int ret = avformat_open_input(&pFormatCtx,
    //                                   st.args_.filepath.c_str(),
    //                                   input_format,
    //                                   opts.get());

    //     if (ret != 0)
    //     {
    //         printf("avformat_open_input() failed with error `%s`\n", get_av_error(ret).c_str());
    //         return false;
    //     }

    //     st.connected_time = system_clock::now();
    //     st.pFormatCtx.reset(pFormatCtx);

    //     ret = avformat_find_stream_info(st.pFormatCtx.get(), NULL);

    //     if (ret < 0)
    //     {
    //         printf("avformat_find_stream_info() failed with error `%s`\n", get_av_error(ret).c_str());
    //         return false;
    //     }

    //     auto setup_stream = [&](bool is_video)
    //     {
    //         const AVMediaType media_type   = is_video ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;
    //         const auto&       channel_args = is_video ? static_cast<const dec_codec_args&>(st.args_.image_options) :
    //                                                     static_cast<const dec_codec_args&>(st.args_.audio_options);

    //         AVCodec* pCodec = nullptr;
    //         const int stream_id = av_find_best_stream(st.pFormatCtx.get(), media_type, -1, -1, &pCodec, 0);

    //         if (stream_id == AVERROR_STREAM_NOT_FOUND)
    //         {
    //             //You might be asking for both video and audio but only video is available. That's OK. Just provide video.
    //             return true;
    //         }
    //         else if (stream_id == AVERROR_DECODER_NOT_FOUND)
    //         {
    //             printf("av_find_best_stream() : decoder not found for stream type `%s`\n", av_get_media_type_string(media_type));
    //             return false;
    //         }
    //         else if (stream_id < 0)
    //         {
    //             printf("av_find_best_stream() failed : `%s`\n", get_av_error(stream_id).c_str());
    //             return false;
    //         }

    //         av_ptr<AVCodecContext> pCodecCtx{avcodec_alloc_context3(pCodec)};

    //         if (!pCodecCtx)
    //         {
    //             printf("avcodec_alloc_context3() failed to allocate codec context for `%s`\n", pCodec->name);
    //             return false;
    //         }

    //         const int ret = avcodec_parameters_to_context(pCodecCtx.get(), st.pFormatCtx->streams[stream_id]->codecpar);
    //         if (ret < 0)
    //         {
    //             printf("avcodec_parameters_to_context() failed : `%s`\n", get_av_error(ret).c_str());
    //             return false;
    //         }

    //         if (pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO)
    //         {
    //             if (pCodecCtx->height   == 0 ||
    //                 pCodecCtx->width    == 0 ||
    //                 pCodecCtx->pix_fmt  == AV_PIX_FMT_NONE)
    //             {
    //                 printf("Codec parameters look wrong : (h,w,pixel_fmt) : (%i,%i,%s)\n",
    //                        pCodecCtx->height,
    //                        pCodecCtx->width,
    //                        get_pixel_fmt_str(pCodecCtx->pix_fmt).c_str());
    //                 return false;
    //             }
    //         }
    //         else if (pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO)
    //         {
    //             if (pCodecCtx->sample_rate == 0 ||
    //                 pCodecCtx->sample_fmt  == AV_SAMPLE_FMT_NONE)
    //             {
    //                 printf("Codec parameters look wrong: sample_rate : %i sample_fmt : %i\n",
    //                        pCodecCtx->sample_rate,
    //                        pCodecCtx->sample_fmt);
    //                 return false;
    //             }

    //             if (pCodecCtx->channel_layout == 0)
    //                 pCodecCtx->channel_layout = av_get_default_channel_layout(pCodecCtx->channels);
    //         }
    //         else
    //         {
    //             printf("Unrecognized media type %i\n", pCodecCtx->codec_type);
    //             return false;
    //         }

    //         if (is_video)
    //         {
    //             decoder_extractor::args extractor_args;
    //             extractor_args.args_codec = st.args_.image_options;
    //             extractor_args.args_image = st.args_.image_options;
    //             extractor_args.time_base  = st.pFormatCtx->streams[stream_id]->time_base;
    //             st.channel_video = decoder_extractor{{extractor_args}, std::move(pCodecCtx), pCodec};
    //             st.channel_video.stream_id = stream_id;
    //             /*! Only really need this if you want to call properties without having read anything yet !*/
    //             st.channel_video.resizer_image.reset(
    //                 st.channel_video.pCodecCtx->height,
    //                 st.channel_video.pCodecCtx->width,
    //                 st.channel_video.pCodecCtx->pix_fmt,
    //                 st.args_.image_options.h > 0 ? st.args_.image_options.h : st.channel_video.pCodecCtx->height,
    //                 st.args_.image_options.w > 0 ? st.args_.image_options.w : st.channel_video.pCodecCtx->width,
    //                 st.args_.image_options.fmt != AV_PIX_FMT_NONE ? st.args_.image_options.fmt : st.channel_video.pCodecCtx->pix_fmt
    //             );
    //         }
    //         else
    //         {
    //             decoder_extractor::args extractor_args;
    //             extractor_args.args_codec = st.args_.audio_options;
    //             extractor_args.args_audio = st.args_.audio_options;
    //             extractor_args.time_base  = st.pFormatCtx->streams[stream_id]->time_base;
    //             st.channel_audio = decoder_extractor{{extractor_args}, std::move(pCodecCtx), pCodec};
    //             st.channel_audio.stream_id = stream_id;
    //             /*! Only really need this if you want to call properties without having read anything yet !*/
    //             st.channel_audio.resizer_audio.reset(
    //                 st.channel_audio.pCodecCtx->sample_rate,
    //                 st.channel_audio.pCodecCtx->channel_layout,
    //                 st.channel_audio.pCodecCtx->sample_fmt,
    //                 st.args_.audio_options.sample_rate > 0            ? st.args_.audio_options.sample_rate      : st.channel_audio.pCodecCtx->sample_rate,
    //                 st.args_.audio_options.channel_layout > 0         ? st.args_.audio_options.channel_layout   : st.channel_audio.pCodecCtx->channel_layout,
    //                 st.args_.audio_options.fmt != AV_SAMPLE_FMT_NONE  ? st.args_.audio_options.fmt              : st.channel_audio.pCodecCtx->sample_fmt
    //             );
    //         }

    //         return true;
    //     };

    //     if (st.args_.enable_image)
    //         setup_stream(true);

    //     if (st.args_.enable_audio)
    //         setup_stream(false);

    //     if (!st.channel_audio.is_open() && !st.channel_video.is_open())
    //     {
    //         printf("At least one of video and audio channels must be enabled\n");
    //         return false;
    //     }

    //     populate_metadata();

    //     st.packet = make_avpacket();
    //     st.opened = true;
    //     return true;
    // }

    // bool demuxer_ffmpeg::is_open() const noexcept
    // {
    //     const bool frames_cached = !st.frame_queue.empty();
    //     const bool object_alive  = st.pFormatCtx != nullptr &&
    //                                st.opened &&
    //                                (st.channel_video.is_open() || st.channel_audio.is_open());
    //     return FFMPEG_INITIALIZED && (frames_cached || object_alive);
    // }

    // bool demuxer_ffmpeg::interrupt_callback()
    // {
    //     const auto now = system_clock::now();

    //     if (st.args_.connect_timeout < std::chrono::milliseconds::max())
    //         if (st.connected_time > now && now > (st.connecting_time + st.args_.connect_timeout))
    //             return true;

    //     if (st.args_.read_timeout < std::chrono::milliseconds::max())
    //         if (now > (st.last_read_time + st.args_.read_timeout))
    //             return true;

    //     if (st.args_.interrupter && st.args_.interrupter())
    //         return true;

    //     return false;
    // }

    // void demuxer_ffmpeg::populate_metadata()
    // {
    //     for (unsigned int i = 0; i < st.pFormatCtx->nb_streams; i++)
    //     {
    //         std::string metadata_str;
    //         {
    //             char* charbuf = 0;
    //             av_dict_get_string(st.pFormatCtx->streams[i]->metadata, &charbuf, ',', ';');
    //             metadata_str = std::string(charbuf);
    //             free(charbuf);
    //         }

    //         std::vector<std::string> keyvals = dlib::split(metadata_str, ";");
    //         for (size_t kv = 0; kv < keyvals.size(); kv++) {
    //             std::vector<std::string> kv_item = dlib::split(keyvals[kv], ",");
    //             assert(kv_item.size() == 2);
    //             st.metadata[i][kv_item[0]] = dlib::trim(kv_item[1]);
    //         }
    //     }
    // }

    // typedef enum {
    //     DEMUXER_RECV_PACKET = 0,
    //     DEMUXER_SEND_PACKET,
    //     DEMUXER_FLUSH,
    //     DEMUXER_DONE,
    //     DEMUXER_ERROR = -1
    // } demuxer_state;

    // bool demuxer_ffmpeg::fill_queue()
    // {
    //     if (!is_open())
    //         return false;

    //     decoder_extractor *channel = nullptr;
    //     demuxer_state state = DEMUXER_RECV_PACKET;

    //     auto recv_packet = [&]
    //     {
    //         if (!st.frame_queue.empty())
    //         {
    //             state = DEMUXER_DONE;
    //         }
    //         else
    //         {
    //             av_packet_unref(st.packet.get());
    //             const int ret = av_read_frame(st.pFormatCtx.get(), st.packet.get());

    //             if (ret == AVERROR_EOF)
    //             {
    //                 state = DEMUXER_FLUSH;
    //             }
    //             else if (ret >= 0)
    //             {
    //                 if (st.packet->stream_index == st.channel_video.stream_id)
    //                 {
    //                     channel = &st.channel_video;
    //                     state   = DEMUXER_SEND_PACKET;
    //                 }
    //                 else if (st.packet->stream_index == st.channel_audio.stream_id)
    //                 {
    //                     channel = &st.channel_audio;
    //                     state   = DEMUXER_SEND_PACKET;
    //                 }
    //             }
    //             else
    //             {
    //                 printf("av_read_frame() failed : `%s`\n", get_av_error(ret).c_str());
    //                 state = DEMUXER_ERROR;
    //             }
    //         }
    //     };

    //     auto send_packet = [&]
    //     {
    //         if (!channel->push(st.packet))
    //             state = DEMUXER_ERROR;
    //         else
    //         {
    //             decoder_read_t suc;
    //             Frame frame;

    //             while ((suc = channel->read(frame)) == DECODER_FRAME_AVAILABLE)
    //                 st.frame_queue.push(std::move(frame));

    //             if (suc == DECODER_EAGAIN)
    //                 state = DEMUXER_RECV_PACKET;
    //             else
    //                 state = DEMUXER_DONE;
    //         }
    //     };

    //     auto flushing_channel = [&]
    //     {
    //         while (state != DEMUXER_ERROR && state != DEMUXER_DONE)
    //             send_packet();
    //     };

    //     auto flushing = [&]
    //     {
    //         st.packet = nullptr;
    //         if (st.channel_video.is_open())
    //         {
    //             channel = &st.channel_video;
    //             flushing_channel();
    //         }
    //         if (st.channel_audio.is_open())
    //         {
    //             channel = &st.channel_audio;
    //             flushing_channel();
    //         }
    //     };

    //     while (state != DEMUXER_DONE && state != DEMUXER_ERROR)
    //     {
    //         switch(state)
    //         {
    //             case DEMUXER_RECV_PACKET: recv_packet(); break;
    //             case DEMUXER_SEND_PACKET: send_packet(); break;
    //             case DEMUXER_FLUSH:       flushing(); break;
    //             default: break;
    //         }
    //     }

    //     return state != DEMUXER_ERROR;
    // }

    // bool demuxer_ffmpeg::read(Frame& dst_frame)
    // {
    //     if (!fill_queue())
    //         return false;

    //     if (!st.frame_queue.empty())
    //     {
    //         dst_frame = std::move(st.frame_queue.front());
    //         st.frame_queue.pop();
    //         return true;
    //     }

    //     return false;
    // }

    // bool demuxer_ffmpeg::read(
    //     type_safe_union<array2d<rgb_pixel>, audio_frame> &frame,
    //     std::chrono::system_clock::time_point &timestamp
    // )
    // {
    //     Frame f;

    //     if (!read(f))
    //         return false;

    //     if (f.is_image())
    //     {
    //         frame = frame_to_dlib_image(f, st.channel_video.resizer_image);
    //     }
    //     else if (f.is_audio())
    //     {
    //         frame = frame_to_dlib_audio(f, st.channel_audio.resizer_audio);
    //     }

    //     timestamp = f.timestamp;

    //     return true;
    // }

    // bool            demuxer_ffmpeg::video_enabled()     const noexcept { return st.channel_video.is_image_decoder(); }
    // bool            demuxer_ffmpeg::audio_enabled()     const noexcept { return st.channel_audio.is_audio_decoder(); }
    // int             demuxer_ffmpeg::height()            const noexcept { return st.channel_video.height(); }
    // int             demuxer_ffmpeg::width()             const noexcept { return st.channel_video.width(); }
    // AVPixelFormat   demuxer_ffmpeg::pixel_fmt()         const noexcept { return st.channel_video.pixel_fmt(); }
    // int             demuxer_ffmpeg::sample_rate()       const noexcept { return st.channel_audio.sample_rate(); }
    // uint64_t        demuxer_ffmpeg::channel_layout()    const noexcept { return st.channel_audio.channel_layout(); }
    // AVSampleFormat  demuxer_ffmpeg::sample_fmt()        const noexcept { return st.channel_audio.sample_fmt(); }
    // int             demuxer_ffmpeg::nchannels()         const noexcept { return st.channel_audio.nchannels(); }
    // AVCodecID   demuxer_ffmpeg::get_video_codec_id()    const noexcept { return st.channel_video.get_codec_id(); }
    // AVCodecID   demuxer_ffmpeg::get_audio_codec_id()    const noexcept { return st.channel_audio.get_codec_id(); }
    // std::string demuxer_ffmpeg::get_video_codec_name()  const noexcept { return st.channel_video.get_codec_name(); }
    // std::string demuxer_ffmpeg::get_audio_codec_name()  const noexcept { return st.channel_audio.get_codec_name(); }

    // float demuxer_ffmpeg::fps() const noexcept
    // {
    //     /*!
    //         Do we need to adjust _pFormatCtx->fps_probe_size ?
    //         Do we need to adjust _pFormatCtx->max_analyze_duration ?
    //     !*/
    //     if (st.channel_video.is_image_decoder() && st.pFormatCtx)
    //     {
    //         const float num = st.pFormatCtx->streams[st.channel_video.stream_id]->avg_frame_rate.num;
    //         const float den = st.pFormatCtx->streams[st.channel_video.stream_id]->avg_frame_rate.den;
    //         return num / den;
    //     }

    //     return 0.0f;
    // }

    // int demuxer_ffmpeg::estimated_nframes() const noexcept
    // {
    //     return st.channel_video.is_audio_decoder() ? st.pFormatCtx->streams[st.channel_video.stream_id]->nb_frames : 0;
    // }

    // int demuxer_ffmpeg::estimated_total_samples() const noexcept
    // {
    //     return st.channel_audio.is_audio_decoder() ? st.pFormatCtx->streams[st.channel_audio.stream_id]->duration : 0;
    // }

    // float demuxer_ffmpeg::duration() const noexcept
    // {
    //     return st.pFormatCtx ? (float)av_rescale_q(st.pFormatCtx->duration, {1, AV_TIME_BASE}, {1, 1000000}) * 1e-6 : 0.0f;
    // }

    // std::unordered_map<int, std::unordered_map<std::string, std::string>> demuxer_ffmpeg::get_all_metadata() const noexcept
    // {
    //     return st.metadata;
    // }

    // std::unordered_map<std::string, std::string> demuxer_ffmpeg::get_video_metadata() const noexcept
    // {
    //     auto it = st.metadata.find(st.channel_video.stream_id);
    //     return it != st.metadata.end() ? it->second : std::unordered_map<std::string, std::string>{};
    // }

    // std::unordered_map<std::string, std::string> demuxer_ffmpeg::get_audio_metadata() const noexcept
    // {
    //     auto it = st.metadata.find(st.channel_audio.stream_id);
    //     return it != st.metadata.end() ? it->second : std::unordered_map<std::string, std::string>{};
    // }

    // float demuxer_ffmpeg::get_rotation_angle() const noexcept
    // {
    //     const auto metadata = get_video_metadata();
    //     const auto it = metadata.find("rotate");
    //     return it != metadata.end() ? std::stof(it->second) : 0;
    // }
}