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
                std::function<bool(AVCodecContext*,AVPacket*)> sink
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
        inline int  to_int(const AVRational& a)                          {return a.num / a.den;}
        inline AVRational inv(const AVRational& a)                       {return {a.den, a.num};}

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
                    printf("Channel layout `%s` not supported. Changing to default `%s`\n",
                        details::get_channel_layout_str(pCodecCtx).c_str(),
                        details::get_channel_layout_str(pCodec->ch_layouts[0]).c_str());
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
                    printf("Channel layout `%s` not supported. Changing to default `%s`\n",
                        get_channel_layout_str(pCodecCtx->channel_layout).c_str(),
                        get_channel_layout_str(pCodec->channel_layouts[0]).c_str());
                    pCodecCtx->channel_layout = pCodec->channel_layouts[0];
                }
            }
#endif
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
                return fail(cerr, "Codec ",  avcodec_get_name(args_.args_codec.codec), " or ", args_.args_codec.codec_name, " not found");

            pCodecCtx.reset(avcodec_alloc_context3(pCodec));
            if (!pCodecCtx)
                return fail(cerr, "AV : failed to allocate codec context for ", pCodec->name, " : likely ran out of memory");

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
                    return fail(cerr, pCodec->name, " is an image codec. height, width, fmt (pixel format) and framerate must be set");
                }

                pCodecCtx->height       = args_.args_image.h;
                pCodecCtx->width        = args_.args_image.w;
                pCodecCtx->pix_fmt      = args_.args_image.fmt;
                pCodecCtx->framerate    = AVRational{args_.args_image.framerate, 1};
                check_properties(pCodec, pCodecCtx.get());
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
                    return fail(cerr, pCodec->name, " is an audio codec. sample_rate, channel_layout and fmt (sample format) must be set");
                }

                pCodecCtx->sample_rate      = args_.args_audio.sample_rate;
                pCodecCtx->sample_fmt       = args_.args_audio.fmt;
                set_layout(pCodecCtx.get(), args_.args_audio.channel_layout);
                check_properties(pCodec, pCodecCtx.get());
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
                return fail(cerr, "avcodec_open2() failed : ", get_av_error(ret));

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
        inline int             encoder::fps()              const noexcept { return pCodecCtx ? to_int(pCodecCtx->framerate) : 0; }
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
                // if (f.get_timestamp() != std::chrono::system_clock::time_point{})
                // {
                //     f.f->pts = av_rescale_q(
                //             f.timestamp.time_since_epoch().count(),
                //             {decltype(f.timestamp)::period::num,decltype(f.timestamp)::period::den},
                //             pCodecCtx->time_base);
                // }
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
                    open_   = false;
                    state   = ENCODE_DONE;
                }
                else if (ret < 0)
                {
                    open_   = false;
                    state   = ENCODE_ERROR;
                    printf("avcodec_receive_packet() failed : %i - `%s`\n", ret, get_av_error(ret).c_str());
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

    }
}

#endif //DLIB_VIDEO_MUXER
