#include <chrono>
#include <thread>
#include "video_demuxer.h"
#include "../string.h"

using namespace std::chrono;

namespace dlib
{
    decoder_ffmpeg::decoder_ffmpeg(const args& a)
    : _args(a)
    {
        connected = open();
    }

    bool decoder_ffmpeg::open()
    {
        if (_args.options.is_empty())
        {
            DLIB_ASSERT("You must set `options` to either an instance of image_args or audio_args");
            return false;
        }

        packet = make_avpacket();
        frame  = make_avframe();
        AVCodec* pCodec = nullptr;

        if (_args.base.codec != AV_CODEC_ID_NONE)
            pCodec = avcodec_find_decoder(_args.base.codec);
        else if (!_args.base.codec_name.empty())
            pCodec = avcodec_find_decoder_by_name(_args.base.codec_name.c_str());

        if (!pCodec)
        {
            printf("Codec %i : `%s` not found\n", _args.base.codec, _args.base.codec_name.c_str());
            return false;
        }

        parser.reset(av_parser_init(pCodec->id));
        if (!parser)
        {
            printf("AV : parser for codec `%s` not found\n", pCodec->name);
            return false;
        }

        pCodecCtx.reset(avcodec_alloc_context3(pCodec));
        if (!pCodecCtx)
        {
            printf("AV : failed to allocate codec context for `%s` : likely ran out of memory", pCodec->name);
            return false;
        }

        pCodecCtx->thread_count = _args.base.nthreads > 0 ? _args.base.nthreads : std::thread::hardware_concurrency();

        av_dict opt = _args.base.codec_options;
        if (avcodec_open2(pCodecCtx.get(), pCodec, opt.avdic ? &opt.avdic : nullptr) < 0)
        {
            printf("AV : failed to open video encoder\n");
            return false;
        }

        return true;
    }

    bool decoder_ffmpeg::is_open() const
    {
        return connected && parser != nullptr && pCodecCtx != nullptr;
    }

    int decoder_ffmpeg::height() const
    {
        return resizer_image.get_dst_h();
    }

    int decoder_ffmpeg::width() const
    {
        return resizer_image.get_dst_w();
    }

    AVPixelFormat decoder_ffmpeg::fmt() const
    {
        return resizer_image.get_dst_fmt();
    }

    int decoder_ffmpeg::sample_rate() const
    {
        return resizer_audio.get_dst_rate();
    }

    uint64_t decoder_ffmpeg::channel_layout() const
    {
        return resizer_audio.get_dst_layout();
    }

    AVSampleFormat decoder_ffmpeg::sample_fmt() const
    {
        return resizer_audio.get_dst_fmt();
    }

    int decoder_ffmpeg::nchannels() const
    {
        return av_get_channel_layout_nb_channels(channel_layout());
    }

    bool decoder_ffmpeg::push_encoded(const uint8_t* encoded, int nencoded)
    {
        if (!is_open())
            return false;

        /*need this stupid padding*/
        encoded_buffer.resize(nencoded + AV_INPUT_BUFFER_PADDING_SIZE, 0);
        memcpy(&encoded_buffer[0], encoded, nencoded);
        encoded = &encoded_buffer[0];

        auto recv_packet = [&]
        {
            int ret = av_parser_parse2(
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
            }
            else
            {
                encoded     += ret;
                nencoded    -= ret;
            }
            return ret >= 0;
        };

        auto send_packet = [&](const AVPacket* pkt)
        {
            int ret = avcodec_send_packet(pCodecCtx.get(), pkt);
            int suc = -1; //default to not-ok

            if (ret == AVERROR(EAGAIN))
            {
                suc = 0;
            }
            else if (ret < 0 && ret != AVERROR_EOF)
            {
                printf("avcodec_send_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
            }
            else if (ret >= 0)
            {
                suc = 1;
            }

            return suc;
        };

        auto recv_frame = [&]
        {
            AVFrameRefLock lock(frame);
            int ret = avcodec_receive_frame(pCodecCtx.get(), frame.get());
            int suc = -1; //default to not-ok

            if (ret == AVERROR(EAGAIN))
            {
                suc = 0; //ok but need more input
            }
            else if (ret == AVERROR_EOF)
            {
                printf("AV : EOF\n");
            }
            else if (ret < 0)
            {
                printf("avcodec_receive_frame() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
            }
            else
            {
                const bool is_video         = pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
                const AVRational tb         = is_video ? pCodecCtx->time_base : AVRational{1, frame->sample_rate};
                const uint64_t pts          = is_video ? frame->pts : next_pts;
                const uint64_t timestamp_us = av_rescale_q(pts, tb, {1,1000000});
                next_pts                    += is_video ? 1 : frame->nb_samples;

                sw_frame decoded(frame, timestamp_us);

                if (is_video)
                {
                    auto& opts = _args.options.cast_to<args::image_args>();
                    resizer_image.resize_inplace(
                            opts.h > 0 ? opts.h : decoded.st.h,
                            opts.w > 0 ? opts.w : decoded.st.w,
                            opts.fmt != AV_PIX_FMT_NONE ? opts.fmt : decoded.st.pixfmt,
                            decoded);
                }
                else
                {
                    auto& opts = _args.options.cast_to<args::audio_args>();
                    resizer_audio.resize_inplace(
                            opts.sample_rate > 0            ? opts.sample_rate      : decoded.st.sample_rate,
                            opts.channel_layout > 0         ? opts.channel_layout   : decoded.st.channel_layout,
                            opts.fmt != AV_SAMPLE_FMT_NONE  ? opts.fmt              : decoded.st.samplefmt,
                            decoded);
                }

                src_frame_buffer.push(std::move(decoded));
                suc = 1; //ok, keep receiving
            }

            return suc;
        };

        auto decode = [&](const AVPacket* pkt)
        {
            int suc1 = 0; //-1 == error, 0 == ok but EAGAIN, 1 == ok
            int suc2 = 0; //-1 == error, 0 == ok but EAGAIN, 1 == ok
            bool ok = true;

            while (ok && suc1 == 0)
            {
                suc1 = send_packet(pkt);
                ok   = suc1 >= 0;

                if (suc1 >= 0)
                {
                    suc2 = 1;

                    while (suc2 > 0)
                        suc2 = recv_frame();

                    ok = suc2 >= 0;
                }
            }

            return ok;
        };

        bool ok = true;

        while (ok && nencoded > 0)
        {
            ok = recv_packet();

            if (ok && packet->size > 0)
            {
                ok = decode(packet.get());
            }
        }

        if (!ok)
        {
            /*! FLUSH !*/
            decode(nullptr);
        }

        connected = ok;
        return connected;
    }

    decoder_ffmpeg::suc_t decoder_ffmpeg::read(sw_frame& dst_frame)
    {
        if (!src_frame_buffer.empty())
        {
            dst_frame = std::move(src_frame_buffer.front());
            src_frame_buffer.pop();
            return decoder_ffmpeg::FRAME_AVAILABLE;
        }

        if (!is_open())
            return decoder_ffmpeg::CLOSED;

        return decoder_ffmpeg::MORE_INPUT;
    }

    demuxer_ffmpeg::demuxer_ffmpeg(const args& a)
    {
        st.connected = open(a);
    }

    demuxer_ffmpeg::demuxer_ffmpeg(demuxer_ffmpeg &&other)
    {
        swap(*this, other);
    }

    demuxer_ffmpeg& demuxer_ffmpeg::operator=(demuxer_ffmpeg &&other)
    {
        swap(*this, other);
        return *this;
    }

    void swap(demuxer_ffmpeg& a, demuxer_ffmpeg& b)
    {
        /*!
            The reason why we don't let the compiler synthesise swap() and move semantics is because we must
            manually reset the opaque pointers. If we don't do this, the interrupt callback breaks badly.
            If it wasn't for this, we could let compiler generate everything.
        !*/
        using std::swap;
        swap(a.st, b.st);
        if (a.st.pFormatCtx)
            a.st.pFormatCtx->opaque = &a;
        if (b.st.pFormatCtx)
            b.st.pFormatCtx->opaque = &b;
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

    void demuxer_ffmpeg::reset()
    {
        demuxer_ffmpeg empty;
        swap(empty, *this);
    }

    bool demuxer_ffmpeg::open(const args& a)
    {
        reset();
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

        int ret = avformat_open_input(&pFormatCtx,
                                      st._args.filepath.c_str(),
                                      input_format,
                                      opts.avdic ? &opts.avdic : NULL);

        if (ret != 0)
        {
            printf("avformat_open_input() failed with error `%s`\n", get_av_error(ret).c_str());
            return false;
        }

        st.connected_time_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
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
                printf("av_find_best_stream() : stream not found for stream type `%s`\n", av_get_media_type_string(media_type));
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

            ch.pCodecCtx->thread_count = common.nthreads > 0 ? common.nthreads : std::thread::hardware_concurrency() / 2;

            av_dict opt = common.codec_options;

            ret = avcodec_open2(ch.pCodecCtx.get(), pCodec, opt.avdic ? &opt.avdic : nullptr);
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
                    printf("Codec parameters look wrong : (h,w,fmt) : (%i,%i,%s)\n",
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
                    printf("Codec parameters look wrong: sample_rate : %i fmt : %i\n",
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

        st.connected = true;
        return st.connected;
    }

    void demuxer_ffmpeg::fill_decoded_buffer()
    {
        if (!st.src_frame_buffer.empty())
            return;

        if (!st.connected)
            return;

        auto recv_packet = [&]()
        {
            int ret = av_read_frame(st.pFormatCtx.get(), st.packet.get());
            if (ret < 0 && ret != AVERROR_EOF)
                printf("av_read_frame() failed : `%s`\n", get_av_error(ret).c_str());
            return ret >= 0;
        };

        auto send_packet = [&](channel &ch, const AVPacket* pkt)
        {
            int ret = avcodec_send_packet(ch.pCodecCtx.get(), pkt);
            int suc = -1; //default to not-ok

            if (ret == AVERROR(EAGAIN))
            {
                suc = 0;
            }
            else if (ret < 0 && ret != AVERROR_EOF)
            {
                printf("avcodec_send_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
            }
            else if (ret >= 0)
            {
                suc = 1;
            }

            return suc;
        };

        auto recv_frame = [&](channel &ch)
        {
            AVFrameRefLock lock(st.frame);
            int ret = avcodec_receive_frame(ch.pCodecCtx.get(), st.frame.get());
            int suc = -1; //default to not-ok

            if (ret == AVERROR(EAGAIN))
            {
                suc = 0; //ok but need more input
            }
            else if (ret == AVERROR_EOF)
            {
            }
            else if (ret < 0)
            {
                printf("avcodec_receive_frame() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
            }
            else
            {
                const bool is_video         = ch.pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
                const AVRational tb         = is_video ? st.pFormatCtx->streams[ch.stream_id]->time_base : AVRational{1, st.frame->sample_rate};
                const uint64_t pts          = is_video ? st.frame->pts : ch.next_pts;
                const uint64_t timestamp_us = av_rescale_q(pts, tb, {1,1000000});
                ch.next_pts                 += is_video ? 1 : st.frame->nb_samples;

                sw_frame decoded(st.frame, timestamp_us);

                if (is_video)
                    st.channel_video.resizer_image.resize_inplace(decoded);
                else
                    st.channel_audio.resizer_audio.resize_inplace(decoded);

                st.src_frame_buffer.push(std::move(decoded));
                st.last_read_time_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                suc = 1; //ok, keep receiving
            }

            return suc;
        };

        auto decode = [&](channel& ch, const AVPacket* pkt)
        {
            int suc1 = 0; //-1 == error, 0 == ok but EAGAIN, 1 == ok
            int suc2 = 0; //-1 == error, 0 == ok but EAGAIN, 1 == ok
            bool ok = true;

            while (ok && suc1 == 0)
            {
                suc1 = send_packet(ch, pkt);
                ok   = suc1 >= 0;

                if (suc1 >= 0)
                {
                    suc2 = 1;

                    while (suc2 > 0)
                        suc2 = recv_frame(ch);

                    ok = suc2 >= 0;
                }
            }

            return ok;
        };

        bool ok = true;
        size_t ndecoded_frames = st.src_frame_buffer.size();

        while (ok && (st.src_frame_buffer.size() - ndecoded_frames) == 0)
        {
            ok = recv_packet();

            if (ok)
            {
                AVPacketRefLock lock(st.packet);

                if (st.packet->stream_index == st.channel_video.stream_id ||
                    st.packet->stream_index == st.channel_audio.stream_id)
                {
                    channel& ch = st.packet->stream_index == st.channel_video.stream_id ? st.channel_video : st.channel_audio;

                    ok = decode(ch, st.packet.get());
                }
            }
        }

        if (!ok)
        {
            /*! FLUSH !*/
            if (st.channel_video.is_enabled())
                decode(st.channel_video, nullptr);
            if (st.channel_audio.is_enabled())
                decode(st.channel_audio, nullptr);
        }

        st.connected = ok;
    }

    bool demuxer_ffmpeg::read(sw_frame& dst_frame)
    {
        fill_decoded_buffer();

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
        uint64_t &timestamp_us
    )
    {
        sw_frame f;

        if (!read(f))
            return false;

        if (f.is_video())
        {
            if (f.st.pixfmt != AV_PIX_FMT_RGB24)
            {
                st.channel_video.resizer_image.resize_inplace(
                        st.channel_video.resizer_image.get_dst_h(),
                        st.channel_video.resizer_image.get_dst_w(),
                        AV_PIX_FMT_RGB24,
                        f);
            }

            array2d<rgb_pixel> frame_image(f.st.h, f.st.w);
            for (int row = 0 ; row < f.st.h ; row++)
            {
                memcpy(frame_image.begin() + row * f.st.w,
                       f.st.data[0] + row * f.st.linesize[0],
                       f.st.w*3);
            }

            frame = std::move(frame_image);
        }
        else if (f.is_audio())
        {
            audio_frame frame_audio;
            frame_audio.sample_rate = f.st.sample_rate;
            frame_audio.samples.resize(f.st.nb_samples);
            memcpy(frame_audio.samples.data(), f.st.data[0], frame_audio.samples.size()*sizeof(audio_frame::sample));

            frame = std::move(frame_audio);
        }

        timestamp_us = f.st.timestamp_us;

        return true;
    }

    bool demuxer_ffmpeg::is_open() const
    {
        const bool frames_available = !st.src_frame_buffer.empty();
        const bool object_ok        = FFMPEG_INITIALIZED        &&
                                      st.connected              &&
                                      st.pFormatCtx != nullptr  &&
                                      (st.channel_video.is_enabled() || st.channel_audio.is_enabled());
        return frames_available || object_ok;
    }

    bool demuxer_ffmpeg::video_enabled() const
    {
        return st.channel_video.is_enabled();
    }

    bool demuxer_ffmpeg::audio_enabled() const
    {
        return st.channel_audio.is_enabled();
    }

    int demuxer_ffmpeg::height() const
    {
        return st.channel_video.is_enabled() ? st.channel_video.resizer_image.get_dst_h() : -1;
    }

    int demuxer_ffmpeg::width() const
    {
        return st.channel_video.is_enabled() ? st.channel_video.resizer_image.get_dst_w() : -1;
    }

    AVPixelFormat demuxer_ffmpeg::fmt() const
    {
        return st.channel_video.is_enabled() ? st.channel_video.resizer_image.get_dst_fmt() : AV_PIX_FMT_NONE;
    }

    float demuxer_ffmpeg::fps() const
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

    int demuxer_ffmpeg::sample_rate() const
    {
        return st.channel_audio.is_enabled() ? st.channel_audio.resizer_audio.get_dst_rate() : -1;
    }

    uint64_t demuxer_ffmpeg::channel_layout() const
    {
        return st.channel_audio.is_enabled() ? st.channel_audio.resizer_audio.get_dst_layout() : 0;
    }

    AVSampleFormat demuxer_ffmpeg::sample_fmt() const
    {
        return st.channel_audio.is_enabled() ? st.channel_audio.resizer_audio.get_dst_fmt() : AV_SAMPLE_FMT_NONE;
    }

    int demuxer_ffmpeg::nchannels() const
    {
        return st.channel_audio.is_enabled() ? av_get_channel_layout_nb_channels(channel_layout()) : 0;
    }

    bool demuxer_ffmpeg::interrupt_callback()
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

        if (st._args.read_timeout_ms > 0 && st.last_read_time_ms > 0)
        {
            const auto diff_ms = now - st.last_read_time_ms;
            do_interrupt = do_interrupt || (diff_ms > st._args.read_timeout_ms);
        }

        if (st._args.interrupter)
        {
            do_interrupt = do_interrupt || st._args.interrupter();
        }

        return do_interrupt;
    }

    std::map<int, std::map<std::string, std::string>> demuxer_ffmpeg::get_all_metadata() const
    {
        return st.metadata;
    }

    std::map<std::string, std::string> demuxer_ffmpeg::get_video_metadata() const
    {
        const static std::map<std::string,std::string> empty;
        return st.pFormatCtx &&
               st.channel_video.is_enabled() &&
               st.metadata.find(st.channel_video.stream_id) != st.metadata.end() ?
               st.metadata.at(st.channel_video.stream_id) :
               empty;
    }

    float demuxer_ffmpeg::get_rotation_angle() const
    {
        const auto metadata = get_video_metadata();
        const auto it = metadata.find("rotate");
        return it != metadata.end() ? std::stof(it->second) : 0;
    }

    bool demuxer_ffmpeg::channel::is_enabled() const
    {
        return pCodecCtx != nullptr;
    }
}