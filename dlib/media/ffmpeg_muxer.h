// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_MUXER
#define DLIB_VIDEO_MUXER

#include <queue>
#include <functional>
#include <unordered_map>
#include "ffmpeg_utils.h"

namespace dlib
{
    namespace ffmpeg
    {
// ---------------------------------------------------------------------------------------------------

        struct encoder_image_args
        {
            int             h{0};
            int             w{0};
            AVPixelFormat   fmt{AV_PIX_FMT_YUV420P};
            int             framerate{0};
        };

// ---------------------------------------------------------------------------------------------------

        struct encoder_audio_args
        {
            int             sample_rate{0};
            uint64_t        channel_layout{AV_CH_LAYOUT_STEREO};
            AVSampleFormat  fmt{AV_SAMPLE_FMT_S16};
        };

// ---------------------------------------------------------------------------------------------------

        struct encoder_codec_args
        {
            AVCodecID                                    codec{AV_CODEC_ID_NONE};
            std::string                                  codec_name;
            std::unordered_map<std::string, std::string> codec_options;
            int64_t                                      bitrate{-1};
            int                                          gop_size{-1};
            int                                          flags{0};
        };

// ---------------------------------------------------------------------------------------------------

        class encoder
        {
        public:
            struct args
            {
                encoder_codec_args args_codec;
                encoder_image_args args_image;
                encoder_audio_args args_audio;
            };

            encoder()                             = default;
            encoder(encoder&& other)              = default;
            encoder& operator=(encoder&& other)   = default;

            encoder(
                const args& a,
                std::function<bool(std::size_t, const char*)> sink
            );

            ~encoder();

            bool            is_open()           const noexcept;
            bool            is_image_encoder()  const noexcept;
            bool            is_audio_encoder()  const noexcept;
            AVCodecID       get_codec_id()      const noexcept;
            std::string     get_codec_name()    const noexcept;
            /*! video properties !*/
            int             height()            const noexcept;
            int             width()             const noexcept;
            AVPixelFormat   pixel_fmt()         const noexcept;
            int             fps()               const noexcept;
            /*! audio properties !*/
            int             sample_rate()       const noexcept;
            uint64_t        channel_layout()    const noexcept;
            AVSampleFormat  sample_fmt()        const noexcept;
            int             nchannels()         const noexcept;

            bool push(frame frame);
            void flush();

        private:
            friend class muxer;

            encoder(
                const args& a,
                std::function<bool(AVCodecContext*,AVPacket*)> sink,
                std::shared_ptr<logger> log_
            );

            bool open();

            args                            args_;
            bool                            open_{false};
            details::av_ptr<AVCodecContext> pCodecCtx;
            details::av_ptr<AVPacket>       packet;
            int                             next_pts{0};
            details::resizer                resizer_image;
            details::resampler              resizer_audio;
            details::audio_fifo             fifo;
            std::function<bool(AVCodecContext*,AVPacket*)> sink;
            std::shared_ptr<logger>         log;
        };

// ---------------------------------------------------------------------------------------------------

        class muxer
        {
        public:
            struct args
            {
                args() = default;
                args(const std::string& filepath);
 
                std::string filepath;
                std::string output_format;
                std::unordered_map<std::string, std::string>  format_options;     //An AVDictionary filled with AVFormatContext and muxer-private options. Used by avformat_write_header()
                std::unordered_map<std::string, std::string>  protocol_options;   //An AVDictionary filled with protocol-private options. Used by avio_open2()

                int max_delay{-1};  //See documentation for AVFormatContext::max_delay
                std::chrono::milliseconds   connect_timeout{std::chrono::milliseconds::max()};
                std::chrono::milliseconds   read_timeout{std::chrono::milliseconds::max()};
                std::function<bool()>       interrupter;
                
                struct : encoder_codec_args, encoder_image_args{} args_image;
                struct : encoder_codec_args, encoder_audio_args{} args_audio;
                bool enable_image{true};
                bool enable_audio{true};
            };

            muxer() = default;
            muxer(const args& a);
            muxer(muxer&& other) noexcept;
            muxer& operator=(muxer&& other) noexcept;
            ~muxer();

            bool is_open() const noexcept;
            bool audio_enabled() const noexcept;
            bool video_enabled() const noexcept;

            int             height()                    const noexcept;
            int             width()                     const noexcept;
            AVPixelFormat   pixel_fmt()                 const noexcept;
            float           fps()                       const noexcept;
            int             estimated_nframes()         const noexcept;
            AVCodecID       get_video_codec_id()        const noexcept;
            std::string     get_video_codec_name()      const noexcept;

            int             sample_rate()               const noexcept;
            uint64_t        channel_layout()            const noexcept;
            AVSampleFormat  sample_fmt()                const noexcept;
            int             nchannels()                 const noexcept;
            int             estimated_total_samples()   const noexcept;
            AVCodecID       get_audio_codec_id()        const noexcept;
            std::string     get_audio_codec_name()      const noexcept;

            bool push(frame f);
            void flush();

        private:

            bool open(const args& a);
            bool interrupt_callback();

            struct {
                args                                    args_;
                details::av_ptr<AVFormatContext>        pFormatCtx;
                encoder                                 encoder_image;
                encoder                                 encoder_audio;
                std::chrono::system_clock::time_point   connecting_time{};
                std::chrono::system_clock::time_point   connected_time{};
                std::chrono::system_clock::time_point   last_read_time{};
                std::shared_ptr<logger>                 log;
            } st;
        };

// ---------------------------------------------------------------------------------------------------

    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// DEFINITIONS  ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

namespace dlib
{
    namespace ffmpeg
    {
        namespace details
        {
            inline bool operator==(const AVRational& a, const AVRational& b) {return a.num == b.num && a.den == b.den;}
            inline bool operator!=(const AVRational& a, const AVRational& b) {return !(a == b);}
            inline bool operator==(const AVRational& a, int framerate)       {return a.den > 0 && (a.num / a.den) == framerate;}
            inline bool operator!=(const AVRational& a, int framerate)       {return !(a == framerate);}
            inline int  to_int(const AVRational& a)                          {return a.num / a.den;}
            inline AVRational inv(const AVRational& a)                       {return {a.den, a.num};}

            inline void check_properties(
                const AVCodec*  pCodec,
                AVCodecContext* pCodecCtx,
                logger&         log
            )
            {
                // Video properties
                if (pCodec->supported_framerates && pCodecCtx->framerate != 0)
                {
                    bool framerate_supported = false;

                    for (int i = 0 ; pCodec->supported_framerates[i] != AVRational{0,0} ; i++)
                    {
                        if (pCodecCtx->framerate == pCodec->supported_framerates[i])
                        {
                            framerate_supported = true;
                            break;
                        }
                    }

                    if (!framerate_supported)
                    {
                        log << LINFO 
                            << "Requested framerate "
                            << pCodecCtx->framerate.num / pCodecCtx->framerate.den
                            << " not supported. Changing to default "
                            << pCodec->supported_framerates[0].num / pCodec->supported_framerates[0].den;

                        pCodecCtx->framerate = pCodec->supported_framerates[0];
                    }
                }

                if (pCodec->pix_fmts)
                {
                    bool pix_fmt_supported = false;

                    for (int i = 0 ; pCodec->pix_fmts[i] != AV_PIX_FMT_NONE ; i++)
                    {
                        if (pCodecCtx->pix_fmt == pCodec->pix_fmts[i])
                        {
                            pix_fmt_supported = true;
                            break;
                        }
                    }

                    if (!pix_fmt_supported)
                    {
                        log << LINFO
                            << "Requested pixel format "
                            << av_get_pix_fmt_name(pCodecCtx->pix_fmt)
                            << " not supported. Changing to default "
                            << av_get_pix_fmt_name(pCodec->pix_fmts[0]);

                        pCodecCtx->pix_fmt = pCodec->pix_fmts[0];
                    }
                }

                // Audio properties
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
                        log << LINFO
                            << "Requested sample rate "
                            << pCodecCtx->sample_rate
                            << " not supported. Changing to default "
                            << pCodec->supported_samplerates[0];

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
                        log << LINFO
                            << "Requested sample format "
                            << av_get_sample_fmt_name(pCodecCtx->sample_fmt)
                            << " not supported. Changing to default "
                            << av_get_sample_fmt_name(pCodec->sample_fmts[0]);

                        pCodecCtx->sample_fmt = pCodec->sample_fmts[0];
                    }
                }

#if FF_API_OLD_CHANNEL_LAYOUT
                if (pCodec->ch_layouts)
                {
                    bool channel_layout_supported = false;

                    for (int i = 0 ; av_channel_layout_check(&pCodec->ch_layouts[i]) ; ++i)
                    {
                        if (av_channel_layout_compare(&pCodecCtx->ch_layout, &pCodec->ch_layouts[i]) == 0)
                        {
                            channel_layout_supported = true;
                            break;
                        }
                    }

                    if (!channel_layout_supported)
                    {
                        log << LINFO
                            << "Channel layout "
                            << details::get_channel_layout_str(pCodecCtx)
                            << " not supported. Changing to default "
                            << details::get_channel_layout_str(pCodec->ch_layouts[0]);

                        av_channel_layout_copy(&pCodecCtx->ch_layout, &pCodec->ch_layouts[0]);
                    }
                }
#else
                if (pCodec->channel_layouts)
                {
                    bool channel_layout_supported = false;

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
                        log << LINFO 
                            << "Channel layout "
                            << details::get_channel_layout_str(pCodecCtx)
                            << " not supported. Changing to default "
                            << dlib::ffmpeg::get_channel_layout_str(pCodec->channel_layouts[0]);

                        pCodecCtx->channel_layout = pCodec->channel_layouts[0];
                    }
                }
#endif
            }
        }

        inline encoder::encoder(
            const args &a,
            std::function<bool(std::size_t, const char*)> sink
        ) : encoder(a, [sink](AVCodecContext*, AVPacket* pkt) {
                return sink(pkt->size, (const char*)pkt->data);
            }, std::make_shared<logger>("ffmpeg::encoder"))
        {
        }

        inline encoder::encoder(
            const args& a,
            std::function<bool(AVCodecContext*,AVPacket*)> sink_,
            std::shared_ptr<logger> log_
        ) : args_(a),
            sink(std::move(sink_)),
            log(log_)
        {
            if (!open())
                pCodecCtx = nullptr;
        }

        inline encoder::~encoder()
        {
            flush();
        }

        inline bool encoder::open()
        {
            using namespace std;
            using namespace details;

            DLIB_CASSERT(sink != nullptr, "must provide an appriate sink callback");

            const bool init = details::register_ffmpeg::get(); // This must be used somewhere otherwise compiler might optimize it away.

            packet = make_avpacket();
            const AVCodec* pCodec = nullptr;

            if (args_.args_codec.codec != AV_CODEC_ID_NONE)
                pCodec = init ? avcodec_find_encoder(args_.args_codec.codec) : nullptr;
            else if (!args_.args_codec.codec_name.empty())
                pCodec = init ? avcodec_find_encoder_by_name(args_.args_codec.codec_name.c_str()) : nullptr;

            if (!pCodec)
                return fail(*log, "Codec ",  avcodec_get_name(args_.args_codec.codec), " or ", args_.args_codec.codec_name, " not found");

            pCodecCtx.reset(avcodec_alloc_context3(pCodec));
            if (!pCodecCtx)
                return fail(*log, "AV : failed to allocate codec context for ", pCodec->name, " : likely ran out of memory");

            if (args_.args_codec.bitrate > 0)
                pCodecCtx->bit_rate = args_.args_codec.bitrate;
            if (args_.args_codec.gop_size > 0)
                pCodecCtx->gop_size = args_.args_codec.gop_size;
            if (args_.args_codec.flags > 0)
                pCodecCtx->flags |= args_.args_codec.flags;

            if (pCodec->type == AVMEDIA_TYPE_VIDEO)
            {
                if (args_.args_image.h          <= 0               ||
                    args_.args_image.w          <= 0               ||
                    args_.args_image.fmt        == AV_PIX_FMT_NONE ||
                    args_.args_image.framerate  <= 0)
                {
                    return fail(*log, pCodec->name, " is an image codec. height, width, fmt (pixel format) and framerate must be set");
                }

                pCodecCtx->height       = args_.args_image.h;
                pCodecCtx->width        = args_.args_image.w;
                pCodecCtx->pix_fmt      = args_.args_image.fmt;
                pCodecCtx->framerate    = AVRational{args_.args_image.framerate, 1};
                check_properties(pCodec, pCodecCtx.get(), *log);
                pCodecCtx->time_base    = inv(pCodecCtx->framerate);

                //don't know what src options are, but at least dst options are set
                resizer_image.reset(pCodecCtx->height, pCodecCtx->width, pCodecCtx->pix_fmt,
                                    pCodecCtx->height, pCodecCtx->width, pCodecCtx->pix_fmt);
            }
            else if (pCodec->type == AVMEDIA_TYPE_AUDIO)
            {
                if (args_.args_audio.sample_rate <= 0 ||
                    args_.args_audio.channel_layout <= 0 ||
                    args_.args_audio.fmt == AV_SAMPLE_FMT_NONE) 
                {
                    return fail(*log, pCodec->name, " is an audio codec. sample_rate, channel_layout and fmt (sample format) must be set");
                }

                pCodecCtx->sample_rate      = args_.args_audio.sample_rate;
                pCodecCtx->sample_fmt       = args_.args_audio.fmt;
                set_layout(pCodecCtx.get(), args_.args_audio.channel_layout);
                check_properties(pCodec, pCodecCtx.get(), *log);
                pCodecCtx->time_base        = AVRational{ 1, pCodecCtx->sample_rate };

                if (pCodecCtx->codec_id == AV_CODEC_ID_AAC) {
                    pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
                }

                //don't know what src options are, but at least dst options are set
                resizer_audio.reset(
                        pCodecCtx->sample_rate, get_layout(pCodecCtx.get()), pCodecCtx->sample_fmt,
                        pCodecCtx->sample_rate, get_layout(pCodecCtx.get()), pCodecCtx->sample_fmt
                );
            }

            av_dict opt = args_.args_codec.codec_options;
            const int ret = avcodec_open2(pCodecCtx.get(), pCodec, opt.get());
            if (ret < 0)
                return fail(*log, "avcodec_open2() failed : ", get_av_error(ret));

            if (pCodec->type == AVMEDIA_TYPE_AUDIO)
            {
                fifo = audio_fifo(pCodecCtx->frame_size,
                                  pCodecCtx->sample_fmt,
                                  get_nchannels(pCodecCtx.get()));
            }

            open_ = true;
            return open_;
        }

        inline bool            encoder::is_open()          const noexcept { return pCodecCtx != nullptr && sink != nullptr && open_; }
        inline bool            encoder::is_image_encoder() const noexcept { return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO; }
        inline bool            encoder::is_audio_encoder() const noexcept { return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO; }
        inline AVCodecID       encoder::get_codec_id()     const noexcept { return pCodecCtx ? pCodecCtx->codec_id : AV_CODEC_ID_NONE; }
        inline std::string     encoder::get_codec_name()   const noexcept { return pCodecCtx ? avcodec_get_name(pCodecCtx->codec_id) : "NONE"; }
        inline int             encoder::fps()              const noexcept { return pCodecCtx ? details::to_int(pCodecCtx->framerate) : 0; }
        inline int             encoder::height()           const noexcept { return resizer_image.get_dst_h(); }
        inline int             encoder::width()            const noexcept { return resizer_image.get_dst_w(); }
        inline AVPixelFormat   encoder::pixel_fmt()        const noexcept { return resizer_image.get_dst_fmt(); }
        inline int             encoder::sample_rate()      const noexcept { return resizer_audio.get_dst_rate(); }
        inline uint64_t        encoder::channel_layout()   const noexcept { return resizer_audio.get_dst_layout(); }
        inline AVSampleFormat  encoder::sample_fmt()       const noexcept { return resizer_audio.get_dst_fmt(); }
        inline int             encoder::nchannels()        const noexcept { return details::get_nchannels(channel_layout()); }

        enum encoding_state
        {
            ENCODE_SEND_FRAME,
            ENCODE_READ_PACKET_THEN_DONE,
            ENCODE_READ_PACKET_THEN_SEND_FRAME,
            ENCODE_DONE,
            ENCODE_ERROR = -1
        };
        
        inline bool encoder::push(frame f_)
        {
            using namespace std::chrono;
            using namespace details;

            if (!is_open())
                return false;

            std::vector<frame> frames;

            // Resize if image. Resample if audio. Push through audio fifo if necessary (some audio codecs requires fixed size frames)
            if (f_.is_image())
            {
                resizer_image.resize(f_, f_);
                frames.push_back(std::move(f_));
            }
            else if (f_.is_audio())
            {
                resizer_audio.resize(f_, f_);
                frames = fifo.push_pull(std::move(f_));
            }
            else
            {
                // FLUSH
                frames.push_back(std::move(f_));
            }

            // Set pts based on tracked state. Ignore timestamps for now
            for (auto& f : frames)
            {
                if (f.f)
                {
                    f.f->pts = next_pts;
                    next_pts += (f.is_image() ? 1 : f.nsamples());
                }
            }

            const auto send_frame = [&](encoding_state& state, frame& f)
            {
                const int ret = avcodec_send_frame(pCodecCtx.get(), f.f.get());

                if (ret >= 0) {
                    state   = ENCODE_READ_PACKET_THEN_DONE;
                } else if (ret == AVERROR(EAGAIN)) {
                    state   = ENCODE_READ_PACKET_THEN_SEND_FRAME;
                } else if (ret == AVERROR_EOF) {
                    open_   = false;
                    state   = ENCODE_DONE;
                } else {
                    open_   = false;
                    state   = ENCODE_ERROR;
                    (*log) << LERROR << "avcodec_send_frame() failed : " << get_av_error(ret);
                }
            };

            const auto recv_packet = [&](encoding_state& state, bool resend)
            {
                const int ret = avcodec_receive_packet(pCodecCtx.get(), packet.get());

                if (ret == AVERROR(EAGAIN) && resend)
                    state   = ENCODE_SEND_FRAME;
                else if (ret == AVERROR(EAGAIN))
                    state   = ENCODE_DONE;
                else if (ret == AVERROR_EOF) {
                    open_   = false;
                    state   = ENCODE_DONE;
                }
                else if (ret < 0)
                {
                    open_   = false;
                    state   = ENCODE_ERROR;
                    (*log) << LERROR << "avcodec_receive_packet() failed : " << get_av_error(ret);
                }
                else
                {
                    if (!sink(pCodecCtx.get(), packet.get()))
                    {
                        open_   = false;
                        state   = ENCODE_ERROR;
                    }
                }
            };

            encoding_state state = ENCODE_SEND_FRAME;

            for (size_t i = 0 ; i < frames.size() && is_open() ; ++i)
            {
                state = ENCODE_SEND_FRAME;

                while (state != ENCODE_DONE && state != ENCODE_ERROR)
                {
                    switch(state)
                    {
                        case ENCODE_SEND_FRAME:                     send_frame(state, frames[i]);   break;
                        case ENCODE_READ_PACKET_THEN_DONE:          recv_packet(state, false);      break;
                        case ENCODE_READ_PACKET_THEN_SEND_FRAME:    recv_packet(state, true);       break;
                        default: break;
                    }
                }
            }

            return state != ENCODE_ERROR;
        }

        inline void encoder::flush()
        {
            push(frame{});
        }

        inline muxer::muxer(const args &a)
        {
            if (!open(a))
                st.pFormatCtx = nullptr;
        }

        inline muxer::muxer(muxer &&other) noexcept
        : st{std::move(other.st)}
        {
            if (st.pFormatCtx)
                st.pFormatCtx->opaque = this;
        }

        inline muxer& muxer::operator=(muxer &&other) noexcept
        {
            if (this != &other)
            {
                flush();
                st = std::move(other.st);
                if (st.pFormatCtx)
                    st.pFormatCtx->opaque = this;
            }
            return *this;
        }

        inline muxer::~muxer()
        {
            flush();
        }

        inline bool muxer::open(const args& a)
        {
            using namespace std;
            using namespace std::chrono;
            using namespace details;

            st = {};
            st.log   = std::make_shared<logger>("ffmpeg::muxer");
            st.args_ = a;

            if (!st.args_.enable_audio && !st.args_.enable_image)
                return fail(*st.log, "You need to set at least one of `enable_audio` or `enable_image`");

            static const auto all_codecs = list_codecs();

            {
                st.connecting_time = system_clock::now();
                st.connected_time  = system_clock::time_point::max();

                const char* const format_name   = st.args_.output_format.empty() ? nullptr : st.args_.output_format.c_str();
                const char* const filename      = st.args_.filepath.empty()      ? nullptr : st.args_.filepath.c_str();
                AVFormatContext* pFormatCtx = nullptr;
                int ret = avformat_alloc_output_context2(&pFormatCtx, nullptr, format_name, filename);

                if (ret < 0)
                    return fail(*st.log, "avformat_alloc_output_context2() failed : ", get_av_error(ret));

                st.pFormatCtx.reset(pFormatCtx);
            }

            int stream_counter{0};

            const auto setup_stream = [&](bool is_video)
            {
                // Setup encoder for this stream
                auto& enc = is_video ? st.encoder_image : st.encoder_audio;

                encoder::args args;

                if (is_video)
                {
                    args.args_codec = st.args_.args_image;
                    args.args_image = st.args_.args_image;
                }
                else
                {
                    args.args_codec = st.args_.args_audio;
                    args.args_audio = st.args_.args_audio;
                }

                if (st.pFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
                    args.args_codec.flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                const auto handle_packet = 
                [
                    pFormatCtx = st.pFormatCtx.get(),
                    stream_id = stream_counter,
                    log = st.log
                ]
                (
                    AVCodecContext* pCodecCtx,
                    AVPacket*       pkt
                )
                {
                    AVStream* stream = pFormatCtx->streams[stream_id];
                    av_packet_rescale_ts(pkt, pCodecCtx->time_base, stream->time_base);
                    pkt->stream_index = stream_id;
                    int ret = av_interleaved_write_frame(pFormatCtx, pkt);
                    if (ret < 0)
                        (*log) << LERROR << "av_interleaved_write_frame() failed : " << get_av_error(ret);
                    return ret == 0;
                };

                // Before we create the encoder, check the codec is supported by this muxer
                const auto supported_codecs = list_codecs_for_muxer(st.pFormatCtx->oformat, all_codecs);

                if (std::find_if(begin(supported_codecs), end(supported_codecs), [&](const auto& supported) {
                    return args.args_codec.codec != AV_CODEC_ID_NONE ? 
                                supported.codec_id   == args.args_codec.codec :
                                supported.codec_name == args.args_codec.codec_name;
                }) == end(supported_codecs))
                {
                    (*st.log) << LERROR
                              << "Codec " << avcodec_get_name(args.args_codec.codec) << " or " << args.args_codec.codec_name
                              << " cannot be stored in this file";
                    (*st.log) << LINFO 
                              << "List of supported codecs for muxer " << st.pFormatCtx->oformat->name << " in this installation of ffmpeg:";
                    for (const auto& supported : supported_codecs)
                        (*st.log) << LINFO << "    " << supported.codec_name;
                    return false;
                }

                // Codec is supported by muxer, so create encoder
                enc = encoder(args, handle_packet, st.log);

                if (!enc.is_open())
                    return false;

                AVStream* stream = avformat_new_stream(st.pFormatCtx.get(), enc.pCodecCtx->codec);

                if (!stream)
                    return fail(*st.log, "avformat_new_stream() failed");

                stream->id          = stream_counter;
                stream->time_base   = enc.pCodecCtx->time_base;
                ++stream_counter;

                int ret = avcodec_parameters_from_context(stream->codecpar, enc.pCodecCtx.get());

                if (ret < 0)
                    return fail(*st.log, "avcodec_parameters_from_context() failed : ", get_av_error(ret));

                return true;
            };

            if (st.args_.enable_image && !setup_stream(true))
                return false;

            if (st.args_.enable_audio && !setup_stream(false))
                return false;

            st.pFormatCtx->opaque = this;
            st.pFormatCtx->interrupt_callback.opaque    = st.pFormatCtx.get();
            st.pFormatCtx->interrupt_callback.callback  = [](void* ctx) -> int {
                AVFormatContext* pFormatCtx = (AVFormatContext*)ctx;
                muxer* me = (muxer*)pFormatCtx->opaque;
                return me->interrupt_callback();
            };

            if (st.args_.max_delay > 0)
                st.pFormatCtx->max_delay = st.args_.max_delay;

            if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
            {
                av_dict opt = st.args_.protocol_options;

                int ret = avio_open2(&st.pFormatCtx->pb, st.args_.filepath.c_str(), AVIO_FLAG_WRITE, &st.pFormatCtx->interrupt_callback, opt.get());

                if (ret < 0)
                    return fail(*st.log, "avio_open2() failed : ", get_av_error(ret));
            }

            av_dict opt = st.args_.format_options;

            int ret = avformat_write_header(st.pFormatCtx.get(), opt.get());

            if (ret < 0)
                return fail(*st.log, "avformat_write_header() failed : ", get_av_error(ret));

            st.connected_time = system_clock::now();

            return true;
        }

        inline bool muxer::interrupt_callback()
        {
            const auto now = std::chrono::system_clock::now();

            if (st.args_.connect_timeout < std::chrono::milliseconds::max() && // check there is a timeout
                now < st.connected_time &&                                     // we haven't already connected
                now > (st.connecting_time + st.args_.connect_timeout)          // we've timed-out
            )
                return true;

            if (st.args_.read_timeout < std::chrono::milliseconds::max() &&   // check there is a timeout
                now > (st.last_read_time + st.args_.read_timeout)             // we've timed-out
            )
                return true;

            if (st.args_.interrupter && st.args_.interrupter())               // check user-specified callback
                return true;

            return false;
        }

        inline bool muxer::push(frame f)
        {
            using namespace std;
            using namespace details;

            if (!is_open())
                return false;

            if (f.is_image())
            {
                if (!st.encoder_image.is_open())
                    return fail(*st.log, "frame is an image type but image encoder is not initialized");

                return st.encoder_image.push(std::move(f));
            }

            else if (f.is_audio())
            {
                if (!st.encoder_audio.is_open())
                    return fail(*st.log, "frame is of audio type but audio encoder is not initialized");

                return st.encoder_audio.push(std::move(f));
            }

            return false;
        }

        inline void muxer::flush()
        {
            if (!is_open())
                return;

            // Flush the encoder but don't actually close the underlying AVCodecContext
            st.encoder_image.flush();
            st.encoder_audio.flush();

            const int ret = av_write_trailer(st.pFormatCtx.get());
            if (ret < 0)
                (*st.log) << LERROR << "av_write_trailer() failed : " << details::get_av_error(ret);

            if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
                avio_closep(&st.pFormatCtx->pb);

            st.pFormatCtx = nullptr;
            st.encoder_image = {};
            st.encoder_audio = {};
        }

        inline bool             muxer::is_open()                const noexcept { return video_enabled() || audio_enabled(); }
        inline bool             muxer::video_enabled()          const noexcept { return st.pFormatCtx != nullptr && st.encoder_image.is_image_encoder(); }
        inline bool             muxer::audio_enabled()          const noexcept { return st.pFormatCtx != nullptr && st.encoder_audio.is_audio_encoder(); }
        inline int              muxer::height()                 const noexcept { return st.encoder_image.height(); }
        inline int              muxer::width()                  const noexcept { return st.encoder_image.width(); }
        inline AVPixelFormat    muxer::pixel_fmt()              const noexcept { return st.encoder_image.pixel_fmt(); }
        inline AVCodecID        muxer::get_video_codec_id()     const noexcept { return st.encoder_image.get_codec_id(); }
        inline std::string      muxer::get_video_codec_name()   const noexcept { return st.encoder_image.get_codec_name(); }
        inline int              muxer::sample_rate()            const noexcept { return st.encoder_audio.sample_rate(); }
        inline uint64_t         muxer::channel_layout()         const noexcept { return st.encoder_audio.channel_layout(); }
        inline int              muxer::nchannels()              const noexcept { return st.encoder_audio.nchannels(); }
        inline AVSampleFormat   muxer::sample_fmt()             const noexcept { return st.encoder_audio.sample_fmt(); }
        inline AVCodecID        muxer::get_audio_codec_id()     const noexcept { return st.encoder_audio.get_codec_id(); }
        inline std::string      muxer::get_audio_codec_name()   const noexcept { return st.encoder_audio.get_codec_name(); }

    }

}

#endif //DLIB_VIDEO_MUXER
