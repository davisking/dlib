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
            AVPixelFormat   fmt{AV_PIX_FMT_RGB24};
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
            int                                          flags{0};
        };

// ---------------------------------------------------------------------------------------------------

        class encoder
        {
        public:
            struct args
            {
                encoder_codec_args  args_codec;
                encoder_image_args  args_image;
                encoder_audio_args  args_audio;
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
            /*! audio properties !*/
            int             sample_rate()       const noexcept;
            uint64_t        channel_layout()    const noexcept;
            AVSampleFormat  sample_fmt()        const noexcept;
            int             nchannels()         const noexcept;

            bool push(frame&& frame);
            void flush();

        private:
            friend class muxer;

            encoder(
                const args& a,
                std::function<bool(AVCodecContext*,AVPacket*)> sink
            );

            bool open();

            args                            args_;
            details::av_ptr<AVCodecContext> pCodecCtx;
            details::av_ptr<AVPacket>       packet;
            int                             next_pts{0};
            details::resizer                resizer_image;
            details::resampler              resizer_audio;
            details::audio_fifo             fifo;
            std::function<bool(AVCodecContext*,AVPacket*)> sink;
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
                
                struct : encoder_codec_args, encoder_image_args{} image_options;
                struct : encoder_codec_args, encoder_audio_args{} audio_options;
                bool enable_image{true};
                bool enable_audio{true};
            };

            muxer() = default;
            muxer(const args& a);
            muxer(muxer&& other) noexcept;
            muxer& operator=(muxer&& other) noexcept;
            ~muxer();

            bool            is_open()           const noexcept;
            bool            audio_enabled()     const noexcept;
            bool            video_enabled()     const noexcept;
            /*video dims*/
            int             height()            const noexcept;
            int             width()             const noexcept;
            AVPixelFormat   pixel_fmt()         const noexcept;
            /*audio dims*/
            int             sample_rate()       const noexcept;
            uint64_t        channel_layout()    const noexcept;
            AVSampleFormat  sample_fmt()        const noexcept;
            int             nchannels()         const noexcept;

            bool push(frame&& f);
            void flush();

        private:

            bool open();
            bool interrupt_callback();

            struct {
                args                                    args_;
                details::av_ptr<AVFormatContext>        pFormatCtx;
                encoder                                 encoder_image;
                encoder                                 encoder_audio;
                std::chrono::system_clock::time_point   connecting_time{};
                std::chrono::system_clock::time_point   connected_time{};
                std::chrono::system_clock::time_point   last_read_time{};
            } st;
        };

// ---------------------------------------------------------------------------------------------------

    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// DECLARATIONS ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

namespace dlib
{
    namespace ffmpeg
    {
        inline bool operator==(const AVRational& a, const AVRational& b) {return a.num == b.num && a.den == b.den;}
        inline bool operator!=(const AVRational& a, const AVRational& b) {return !(a == b);}
        inline bool operator==(const AVRational& a, int framerate)       {return a.den > 0 && (a.num / a.den) == framerate;}
        inline bool operator!=(const AVRational& a, int framerate)       {return !(a == framerate);}

        inline void check_properties(
            const AVCodec* pCodec,
            AVCodecContext* pCodecCtx
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
                    printf("Requested framerate %i not supported. Changing to default %i\n",
                        pCodecCtx->framerate.num / pCodecCtx->framerate.den,
                        pCodec->supported_framerates[0].num / pCodec->supported_framerates[0].den);
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
                    printf("Requested pixel format %s not supported. Changing to default %s\n",
                        av_get_pix_fmt_name(pCodecCtx->pix_fmt),
                        av_get_pix_fmt_name(pCodec->pix_fmts[0]));
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

        inline encoder::encoder(
            const args &a,
            std::function<bool(std::size_t, const char*)> sink
        ) : encoder(a, [sink](AVCodecContext*, AVPacket* pkt) {
                return sink(pkt->size, (const char*)pkt->data);
            })
        {
        }

        inline encoder::encoder(
            const args& a,
            std::function<bool(AVCodecContext*,AVPacket*)> sink_
        ) : args_(a),
            sink(std::move(sink_))
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
            {
                printf("Codec `%s` or `%s` not found\n", avcodec_get_name(args_.args_codec.codec), args_.args_codec.codec_name.c_str());
                return false;
            }

            pCodecCtx.reset(avcodec_alloc_context3(pCodec));
            if (!pCodecCtx)
            {
                printf("AV : failed to allocate codec context for `%s` : likely ran out of memory", pCodec->name);
                return false;
            }

            if (args_.args_codec.bitrate > 0)
                pCodecCtx->bit_rate = args_.args_codec.bitrate;
            if (args_.args_codec.flags > 0)
                pCodecCtx->flags |= args_.args_codec.flags;

            if (pCodec->type == AVMEDIA_TYPE_VIDEO)
            {
                DLIB_CASSERT(args_.args_image.h > 0, "height must be set");
                DLIB_CASSERT(args_.args_image.w > 0, "width must be set");
                DLIB_CASSERT(args_.args_image.fmt != AV_PIX_FMT_NONE, "pixel format must be set");
                DLIB_CASSERT(args_.args_image.framerate > 0, "framerate must be set");

                pCodecCtx->height       = args_.args_image.h;
                pCodecCtx->width        = args_.args_image.w;
                pCodecCtx->pix_fmt      = args_.args_image.fmt;
                pCodecCtx->time_base    = AVRational{1, args_.args_image.framerate};
                pCodecCtx->framerate    = AVRational{args_.args_image.framerate, 1};

                //don't know what src options are, but at least dst options are set
                resizer_image.reset(pCodecCtx->height, pCodecCtx->width, pCodecCtx->pix_fmt,
                                    pCodecCtx->height, pCodecCtx->width, pCodecCtx->pix_fmt);
            }
            else if (pCodec->type == AVMEDIA_TYPE_AUDIO)
            {
                DLIB_CASSERT(args_.args_audio.sample_rate > 0, "sample rate not set");
                DLIB_CASSERT(args_.args_audio.channel_layout > 0, "channel layout not set");
                DLIB_CASSERT(args_.args_audio.fmt != AV_SAMPLE_FMT_NONE, "audio sample format not set");

                pCodecCtx->sample_rate      = args_.args_audio.sample_rate;
                pCodecCtx->sample_fmt       = args_.args_audio.fmt;
                pCodecCtx->channel_layout   = args_.args_audio.channel_layout;
                pCodecCtx->channels         = av_get_channel_layout_nb_channels(pCodecCtx->channel_layout);
                pCodecCtx->time_base        = AVRational{ 1, pCodecCtx->sample_rate };

                if (pCodecCtx->codec_id == AV_CODEC_ID_AAC) {
                    pCodecCtx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
                }

                //don't know what src options are, but at least dst options are set
                resizer_audio.reset(
                        pCodecCtx->sample_rate, pCodecCtx->channel_layout, pCodecCtx->sample_fmt,
                        pCodecCtx->sample_rate, pCodecCtx->channel_layout, pCodecCtx->sample_fmt
                );
            }

            check_properties(pCodec, pCodecCtx.get());
            av_dict opt = args_.args_codec.codec_options;
            const int ret = avcodec_open2(pCodecCtx.get(), pCodec, opt.get());
            if (ret < 0)
            {
                printf("avcodec_open2() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }

            if (pCodec->type == AVMEDIA_TYPE_AUDIO)
            {
                fifo = audio_fifo(pCodecCtx->frame_size,
                                  pCodecCtx->sample_fmt,
                                  pCodecCtx->channels);
            }

            return true;
        }

        inline bool            encoder::is_open()          const noexcept { return pCodecCtx != nullptr && sink != nullptr; }
        inline bool            encoder::is_image_encoder() const noexcept { return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO; }
        inline bool            encoder::is_audio_encoder() const noexcept { return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO; }
        inline AVCodecID       encoder::get_codec_id()     const noexcept { return pCodecCtx ? pCodecCtx->codec_id : AV_CODEC_ID_NONE; }
        inline std::string     encoder::get_codec_name()   const noexcept { return pCodecCtx ? avcodec_get_name(pCodecCtx->codec_id) : "NONE"; }
        inline int             encoder::height()           const noexcept { return resizer_image.get_dst_h(); }
        inline int             encoder::width()            const noexcept { return resizer_image.get_dst_w(); }
        inline AVPixelFormat   encoder::pixel_fmt()        const noexcept { return resizer_image.get_dst_fmt(); }
        inline int             encoder::sample_rate()      const noexcept { return resizer_audio.get_dst_rate(); }
        inline uint64_t        encoder::channel_layout()   const noexcept { return resizer_audio.get_dst_layout(); }
        inline AVSampleFormat  encoder::sample_fmt()       const noexcept { return resizer_audio.get_dst_fmt(); }
        inline int             encoder::nchannels()        const noexcept { return av_get_channel_layout_nb_channels(resizer_audio.get_dst_layout()); }

        enum encoding_state
        {
            ENCODE_SEND_FRAME,
            ENCODE_READ_PACKET_THEN_DONE,
            ENCODE_READ_PACKET_THEN_SEND_FRAME,
            ENCODE_DONE,
            ENCODE_ERROR = -1
        };
        
        inline bool encoder::push(frame&& f)
        {
            using namespace std::chrono;
            using namespace details;

            if (!is_open())
                return false;

            std::vector<frame> frames;

            // Resize if image. Resample if audio. Push through audio fifo if necessary (some audio codecs requires fixed size frames)
            if (f.is_image())
            {
                resizer_image.resize(f, f);
                frames.push_back(std::move(f));
            }
            else if (f.is_audio())
            {
                resizer_audio.resize(f, f);
                frames = fifo.push_pull(std::move(f));
            }
            else
            {
                // FLUSH
                frames.push_back(std::move(f));
            }

            // Set pts based on timestamps or tracked state
            for (auto& f : frames)
            {
                if (f.get_timestamp() != std::chrono::system_clock::time_point{})
                {
                    f.f->pts = av_rescale_q(
                            f.timestamp.time_since_epoch().count(),
                            {decltype(f.timestamp)::period::num,decltype(f.timestamp)::period::den},
                            pCodecCtx->time_base);
                }
                else if (f.f)
                {
                    f.f->pts = next_pts;
                }

                if (f.f)
                    next_pts = f.f->pts + (f.is_image() ? 1 : f.nsamples());
            }

            const auto send_frame = [&](encoding_state& state, frame& f)
            {
                const int ret = avcodec_send_frame(pCodecCtx.get(), f.f.get());

                if (ret >= 0) {
                    state   = ENCODE_READ_PACKET_THEN_DONE;
                } else if (ret == AVERROR(EAGAIN)) {
                    state   = ENCODE_READ_PACKET_THEN_SEND_FRAME;
                } else if (ret == AVERROR_EOF) {
                    pCodecCtx = nullptr;
                    state   = ENCODE_DONE;
                } else {
                    pCodecCtx = nullptr;
                    state   = ENCODE_ERROR;
                    printf("avcodec_send_frame() failed : `%s`\n", get_av_error(ret).c_str());
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
                    pCodecCtx = nullptr;
                    state   = ENCODE_DONE;
                }
                else if (ret < 0)
                {
                    pCodecCtx = nullptr;
                    state   = ENCODE_ERROR;
                    printf("avcodec_receive_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
                }
                else
                {
                    if (!sink(pCodecCtx.get(), packet.get()))
                    {
                        pCodecCtx = nullptr;
                        state     = ENCODE_ERROR;
                    }
                }
            };

            encoding_state state = ENCODE_SEND_FRAME;

            for (size_t i = 0 ; i < frames.size() && pCodecCtx != nullptr ; ++i)
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
            st.args_ = a;
            if (!open())
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

        bool muxer::open()
        {
            using namespace std::chrono;
            using namespace details;

            if (!st.args_.enable_audio && !st.args_.enable_image)
            {
                printf("You need to set at least one of `enable_audio` or `enable_image`\n");
                return false;
            }

            {
                st.connecting_time = system_clock::now();
                st.connected_time  = system_clock::time_point::max();

                const char* const format_name   = st.args_.output_format.empty() ? nullptr : st.args_.output_format.c_str();
                const char* const filename      = st.args_.filepath.empty()      ? nullptr : st.args_.filepath.c_str();
                AVFormatContext* pFormatCtx = nullptr;
                int ret = avformat_alloc_output_context2(&pFormatCtx, nullptr, format_name, filename);
                if (ret < 0)
                {
                    printf("avformat_alloc_output_context2() failed : `%s`\n", details::get_av_error(ret).c_str());
                    return false;
                }

                if (!pFormatCtx->oformat)
                {
                    printf("Output format is null\n");
                    return false;
                }

                st.pFormatCtx.reset(pFormatCtx);
            }

            int stream_counter{0};

            const auto setup_stream = [&](bool is_video)
            {
                auto& enc = is_video ? st.encoder_image : st.encoder_audio;

                encoder::args args;

                if (is_video)
                {
                    args.args_codec = st.args_.image_options;
                    args.args_image = st.args_.image_options;
                }
                else
                {
                    args.args_codec = st.args_.audio_options;
                    args.args_audio = st.args_.audio_options;
                }

                if (st.pFormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
                    args.args_codec.flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                const auto handle_packet = 
                [
                    pFormatCtx = st.pFormatCtx.get(),
                    stream_id = stream_counter
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
                        printf("av_interleaved_write_frame() failed : `%s`\n", details::get_av_error(ret).c_str());
                    return ret == 0;
                };

                enc = encoder(args, handle_packet);

                if (!enc.is_open())
                    return false;

                AVStream* stream = avformat_new_stream(st.pFormatCtx.get(), enc.pCodecCtx->codec);

                if (!stream)
                {
                    printf("avformat_new_stream() failed\n");
                    return false;
                }

                stream->id = stream_counter;
                ++stream_counter;

                int ret = avcodec_parameters_from_context(stream->codecpar, enc.pCodecCtx.get());
                if (ret < 0)
                {
                    printf("avcodec_parameters_from_context() failed : `%s`\n", details::get_av_error(ret).c_str());
                    return false;
                }

                return true;
            };

            if (st.args_.enable_image)
            {
                if (!setup_stream(true))
                    return false;
            }

            if (st.args_.enable_audio)
            {
                if (!setup_stream(false))
                    return false;
            }

            st.pFormatCtx->opaque = this;
            st.pFormatCtx->interrupt_callback.opaque    = st.pFormatCtx.get();
            st.pFormatCtx->interrupt_callback.callback  = [](void* ctx) -> int {
                AVFormatContext* pFormatCtx = (AVFormatContext*)ctx;
                muxer* me = (muxer*)pFormatCtx->opaque;
                return me->interrupt_callback();
            };

            if (st.args_.max_delay > 0)
                st.pFormatCtx->max_delay = st.args_.max_delay;

            //st.pFormatCtx->flags = AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;

            if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
            {
                av_dict opt = st.args_.protocol_options;

                int ret = avio_open2(&st.pFormatCtx->pb, st.args_.filepath.c_str(), AVIO_FLAG_WRITE, &st.pFormatCtx->interrupt_callback, opt.get());

                if (ret < 0)
                {
                    printf("avio_open2() failed : `%s`\n", get_av_error(ret).c_str());
                    return false;
                }
            }

            av_dict opt = st.args_.format_options;

            int ret = avformat_write_header(st.pFormatCtx.get(), opt.get());
            if (ret < 0)
            {
                printf("avformat_write_header() failed : `%s`\n", get_av_error(ret).c_str());
                return false;
            }

            st.connected_time = system_clock::now();

            return true;
        }

        bool muxer::interrupt_callback()
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

        bool muxer::push(frame &&frame)
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

        void muxer::flush()
        {
            if (!is_open())
                return;

            st.encoder_image.flush();
            st.encoder_audio.flush();

            const int ret = av_write_trailer(st.pFormatCtx.get());
            if (ret < 0)
                printf("AV : failed to write trailer : `%s`\n", details::get_av_error(ret).c_str());

            if ((st.pFormatCtx->oformat->flags & AVFMT_NOFILE) == 0)
                avio_closep(&st.pFormatCtx->pb);

            st.pFormatCtx.reset(nullptr); //close
        }

        inline bool             muxer::is_open()        const noexcept { return video_enabled() || audio_enabled(); }
        inline bool             muxer::video_enabled()  const noexcept { return st.pFormatCtx != nullptr && st.encoder_image.is_open(); }
        inline bool             muxer::audio_enabled()  const noexcept { return st.pFormatCtx != nullptr && st.encoder_audio.is_open(); }
        inline int              muxer::height()         const noexcept { return st.encoder_image.height(); }
        inline int              muxer::width()          const noexcept { return st.encoder_image.width(); }
        inline AVPixelFormat    muxer::pixel_fmt()      const noexcept { return st.encoder_image.pixel_fmt(); }
        inline int              muxer::sample_rate()    const noexcept { return st.encoder_audio.sample_rate(); }
        uint64_t                muxer::channel_layout() const noexcept { return st.encoder_audio.channel_layout(); }
        int                     muxer::nchannels()      const noexcept { return st.encoder_audio.nchannels(); }
        AVSampleFormat          muxer::sample_fmt()     const noexcept { return st.encoder_audio.sample_fmt(); }
    }
}


#endif //DLIB_VIDEO_MUXER
