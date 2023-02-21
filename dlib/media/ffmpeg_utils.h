// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_UTILS
#define DLIB_FFMPEG_UTILS

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

#include <cstdint>
#include <stdexcept>
#include <cassert>
#include <memory>
#include <algorithm>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <unordered_map>
#include "../image_processing/generic_image.h"
#include "../pixel.h"
#include "../assert.h"
#include "ffmpeg_abstract.h"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/frame.h>
#include <libavutil/channel_layout.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavcodec/avcodec.h>
}

namespace dlib
{
    namespace ffmpeg
    {
        class frame;
        
        namespace details
        {

// ---------------------------------------------------------------------------------------------------

            class register_ffmpeg
            {
                /*!
                    WHAT THIS OBJECT REPRESENTS
                        The point of this class is to statically register ffmpeg globally
                        and ensure its done only ONCE.
                        In C++17 you can have inline variables and this can be done much easier.
                        In C++14 and below you need something like this class.
                !*/
            
            public:

                static bool get()
                {
                    static const bool v = register_library();
                    return v;
                }
            
            private:

                static bool register_library();
            };

// ---------------------------------------------------------------------------------------------------

            struct av_deleter
            {
                void operator()(AVFrame* ptr)               const;
                void operator()(AVPacket* ptr)              const;
                void operator()(AVAudioFifo* ptr)           const;
                void operator()(SwsContext* ptr)            const;
                void operator()(SwrContext* ptr)            const;
                void operator()(AVCodecContext* ptr)        const;
                void operator()(AVCodecParserContext* ptr)  const;
                void operator()(AVFormatContext* ptr)       const;
                void operator()(AVDeviceInfoList* ptr)      const;
            };

            template<class AVObject>
            using av_ptr = std::unique_ptr<AVObject, details::av_deleter>;

            av_ptr<AVFrame>  make_avframe();
            av_ptr<AVPacket> make_avpacket();

// ---------------------------------------------------------------------------------------------------

            struct av_dict
            {
                av_dict() = default;
                av_dict(const std::unordered_map<std::string, std::string> &options);
                av_dict(const av_dict &ori);
                av_dict &operator=(const av_dict &ori);
                av_dict(av_dict &&ori) noexcept;
                av_dict &operator=(av_dict &&ori) noexcept;
                ~av_dict();
                size_t size() const;
                void print() const;
                AVDictionary** get();

                AVDictionary *avdic = nullptr;
            };

// ---------------------------------------------------------------------------------------------------

            class resizer
            {
            public:
                void reset(
                    const int src_h, const int src_w, const AVPixelFormat src_fmt,
                    const int dst_h, const int dst_w, const AVPixelFormat dst_fmt
                );

                void resize(
                    const frame &src,
                    const int dst_h, const int dst_w, const AVPixelFormat dst_fmt,
                    frame &dst
                );

                void resize(
                    const frame &src,
                    frame &dst
                );

                int             get_src_h()    const noexcept { return src_h; }
                int             get_src_w()    const noexcept { return src_w; }
                AVPixelFormat   get_src_fmt()  const noexcept { return src_fmt; }
                int             get_dst_h()    const noexcept { return dst_h; }
                int             get_dst_w()    const noexcept { return dst_w; }
                AVPixelFormat   get_dst_fmt()  const noexcept { return dst_fmt; }

            private:

                int                 src_h{0};
                int                 src_w{0};
                AVPixelFormat       src_fmt{AV_PIX_FMT_NONE};
                int                 dst_h{0};
                int                 dst_w{0};
                AVPixelFormat       dst_fmt{AV_PIX_FMT_NONE};
                av_ptr<SwsContext>  imgConvertCtx;
            };

// ---------------------------------------------------------------------------------------------------

            class resampler
            {
            public:
                void reset(
                    const int src_sample_rate, const uint64_t src_channel_layout, const AVSampleFormat src_fmt,
                    const int dst_sample_rate, const uint64_t dst_channel_layout, const AVSampleFormat dst_fmt
                );

                void resize(
                    const frame &src,
                    const int dst_sample_rate, const uint64_t dst_channel_layout, const AVSampleFormat dst_fmt,
                    frame &dst
                );

                void resize(
                    const frame &src,
                    frame &dst
                );

                int             get_src_rate()   const noexcept { return src_sample_rate; }
                uint64_t        get_src_layout() const noexcept { return src_channel_layout; }
                AVSampleFormat  get_src_fmt()    const noexcept { return src_fmt; }
                int             get_dst_rate()   const noexcept { return dst_sample_rate; }
                uint64_t        get_dst_layout() const noexcept { return dst_channel_layout; }
                AVSampleFormat  get_dst_fmt()    const noexcept { return dst_fmt; }

            private:

                int             src_sample_rate{0};
                uint64_t        src_channel_layout{AV_CH_LAYOUT_STEREO};
                AVSampleFormat  src_fmt{AV_SAMPLE_FMT_NONE};

                int             dst_sample_rate{0};
                uint64_t        dst_channel_layout{AV_CH_LAYOUT_STEREO};
                AVSampleFormat  dst_fmt{AV_SAMPLE_FMT_NONE};

                av_ptr<SwrContext>  audioResamplerCtx;
                uint64_t            tracked_samples{0};
            };

// ---------------------------------------------------------------------------------------------------

            class audio_fifo
            {
            public:
                audio_fifo() = default;

                audio_fifo(
                    const int            codec_frame_size,
                    const AVSampleFormat sample_format,
                    const int            nchannels
                );

                std::vector<frame> push_pull(
                    frame &&in
                );

            private:

                int                 frame_size{0};
                AVSampleFormat      fmt{AV_SAMPLE_FMT_NONE};
                int                 nchannels{0};
                uint64_t            sample_count{0};
                av_ptr<AVAudioFifo> fifo;
            };

// ---------------------------------------------------------------------------------------------------

            std::string get_av_error(int ret);

// ---------------------------------------------------------------------------------------------------

        }

// ---------------------------------------------------------------------------------------------------

        std::string get_pixel_fmt_str(AVPixelFormat fmt);
        std::string get_audio_fmt_str(AVSampleFormat fmt);
        std::string get_channel_layout_str(uint64_t layout);
        uint64_t    get_layout_from_channels(std::size_t nchannels);

// ---------------------------------------------------------------------------------------------------

        namespace details
        {
            class decoder_extractor;
        }

        class frame
        {
        public:
            frame()                         = default;
            frame(frame&& ori)              = default;
            frame& operator=(frame&& ori)   = default;
            frame(const frame& ori);
            frame& operator=(const frame& ori);

            frame(
                int                                     h,
                int                                     w,
                AVPixelFormat                           pixfmt,
                std::chrono::system_clock::time_point   timestamp_us
            );

            frame(
                int                                     sample_rate,
                int                                     nb_samples,
                uint64_t                                channel_layout,
                AVSampleFormat                          samplefmt,
                std::chrono::system_clock::time_point   timestamp
            );

            bool            is_empty()      const noexcept;
            bool            is_image()      const noexcept;
            bool            is_audio()      const noexcept;
            /*image*/
            AVPixelFormat   pixfmt()        const noexcept;
            int             height()        const noexcept;
            int             width()         const noexcept;
            /*audio*/
            int             nsamples()      const noexcept;
            int             nchannels()     const noexcept;
            uint64_t        layout()        const noexcept;
            AVSampleFormat  samplefmt()     const noexcept;
            int             sample_rate()   const noexcept;

            std::chrono::system_clock::time_point get_timestamp() const noexcept;
            const AVFrame&  get_frame() const;
            AVFrame&        get_frame();

        private:

            friend class details::resampler;
            friend class details::decoder_extractor;

            frame(
                int                                     h,
                int                                     w,
                AVPixelFormat                           pixfmt,
                int                                     sample_rate,
                int                                     nb_samples,
                uint64_t                                channel_layout,
                AVSampleFormat                          samplefmt,
                std::chrono::system_clock::time_point   timestamp
            );

            void copy_from(const frame& other);

            details::av_ptr<AVFrame> f;
            std::chrono::system_clock::time_point timestamp;
        };

// ---------------------------------------------------------------------------------------------------

        template<class PixelType>
        struct pix_traits {};

        template<> struct pix_traits<uint8_t>           {constexpr static AVPixelFormat fmt = AV_PIX_FMT_GRAY8; };
        template<> struct pix_traits<rgb_pixel>         {constexpr static AVPixelFormat fmt = AV_PIX_FMT_RGB24; };
        template<> struct pix_traits<bgr_pixel>         {constexpr static AVPixelFormat fmt = AV_PIX_FMT_BGR24; };
        template<> struct pix_traits<rgb_alpha_pixel>   {constexpr static AVPixelFormat fmt = AV_PIX_FMT_RGBA;  };
        template<> struct pix_traits<bgr_alpha_pixel>   {constexpr static AVPixelFormat fmt = AV_PIX_FMT_BGRA;  };

// ---------------------------------------------------------------------------------------------------

        template<class SampleType>
        struct sample_traits {};

        template<> struct sample_traits<uint8_t> {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_U8; };
        template<> struct sample_traits<int16_t> {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_S16; };
        template<> struct sample_traits<int32_t> {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_S32; };
        template<> struct sample_traits<float>   {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_FLT; };
        template<> struct sample_traits<double>  {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_DBL; };

// ---------------------------------------------------------------------------------------------------

        template<class SampleType, std::size_t Channels>
        struct audio
        {
            using sample = std::array<SampleType, Channels>;

            std::vector<sample>                     samples;
            float                                   sample_rate{0};
            std::chrono::system_clock::time_point   timestamp{};
        };

// ---------------------------------------------------------------------------------------------------

        struct codec_details
        {
            std::string codec_name;
            bool supports_encoding{false};
            bool supports_decoding{false};
        };

        struct device_details
        {
            struct instance
            {
                std::string name;
                std::string description;
            };

            std::string device_type;
            std::vector<instance> devices;
        };

        std::vector<std::string>    list_protocols();
        std::vector<std::string>    list_demuxers();
        std::vector<std::string>    list_muxers();
        std::vector<codec_details>  list_codecs();
        std::vector<device_details> list_input_devices();
        std::vector<device_details> list_output_devices();

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

// ---------------------------------------------------------------------------------------------------

        namespace details
        {
            inline std::string get_av_error(int ret)
            {
                char buf[128] = {0};
                int suc = av_strerror(ret, buf, sizeof(buf));
                return suc == 0 ? buf : "couldn't set error";
            }
        }

        inline std::string get_pixel_fmt_str(AVPixelFormat fmt)
        {
            const char* name = av_get_pix_fmt_name(fmt);
            return name ? std::string(name) : std::string("unknown");
        }

        inline std::string get_audio_fmt_str(AVSampleFormat fmt)
        {
            const char* name = av_get_sample_fmt_name(fmt);
            return name ? std::string(name) : std::string("unknown");
        }

        inline std::string get_channel_layout_str(uint64_t layout)
        {
            std::string buf(32, '\0');
            av_get_channel_layout_string(&buf[0], buf.size(), 0, layout);
            return buf;
        }

        inline uint64_t get_layout_from_channels(std::size_t nchannels)
        {
            switch(nchannels)
            {
                case 1: return AV_CH_LAYOUT_MONO;
                case 2: return AV_CH_LAYOUT_STEREO;
                default: return 0;
            }
        }

// ---------------------------------------------------------------------------------------------------

        namespace details
        {

// ---------------------------------------------------------------------------------------------------

            inline bool register_ffmpeg::register_library()
            {
                avdevice_register_all();
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 10, 100)
                // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L91
                avcodec_register_all();
#endif
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100) 
                // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L86
                av_register_all();
#endif
                return true;
            }

// ---------------------------------------------------------------------------------------------------

            inline void av_deleter::operator()(AVFrame *ptr)               const { if (ptr) av_frame_free(&ptr); }
            inline void av_deleter::operator()(AVPacket *ptr)              const { if (ptr) av_packet_free(&ptr); }
            inline void av_deleter::operator()(AVAudioFifo *ptr)           const { if (ptr) av_audio_fifo_free(ptr); }
            inline void av_deleter::operator()(SwsContext *ptr)            const { if (ptr) sws_freeContext(ptr); }
            inline void av_deleter::operator()(SwrContext *ptr)            const { if (ptr) swr_free(&ptr); }
            inline void av_deleter::operator()(AVCodecContext *ptr)        const { if (ptr) avcodec_free_context(&ptr); }
            inline void av_deleter::operator()(AVCodecParserContext *ptr)  const { if (ptr) av_parser_close(ptr); }
            inline void av_deleter::operator()(AVDeviceInfoList* ptr)      const { if (ptr) avdevice_free_list_devices(&ptr); }
            inline void av_deleter::operator()(AVFormatContext *ptr)       const 
            { 
                if (ptr) 
                {
                    if (ptr->iformat)
                        avformat_close_input(&ptr); 
                    else if (ptr->oformat)
                        avformat_free_context(ptr);
                }
            }

            inline av_ptr<AVFrame> make_avframe()
            {
                av_ptr<AVFrame> obj(av_frame_alloc());
                if (!obj)
                    throw std::runtime_error("Failed to allocate AVframe");
                return obj;
            }

            inline av_ptr<AVPacket> make_avpacket()
            {
                av_ptr<AVPacket> obj(av_packet_alloc());
                if (!obj)
                    throw std::runtime_error("Failed to allocate AVPacket");
                return obj;
            }

// ---------------------------------------------------------------------------------------------------

            inline av_dict::av_dict(const std::unordered_map<std::string, std::string>& options)
            {
                int ret = 0;

                for (const auto& opt : options) {
                    if ((ret = av_dict_set(&avdic, opt.first.c_str(), opt.second.c_str(), 0)) < 0) {
                        printf("av_dict_set() failed : %s\n", get_av_error(ret).c_str());
                        break;
                    }
                }
            }

            inline av_dict::av_dict(const av_dict& ori)
            {
                av_dict_copy(&avdic, ori.avdic, 0);
            }

            inline av_dict& av_dict::operator=(const av_dict& ori)
            {
                *this = std::move(av_dict{ori});
                return *this;
            }

            inline av_dict::av_dict(av_dict &&ori) noexcept
            : avdic{std::exchange(ori.avdic, nullptr)}
            {
            }

            inline av_dict &av_dict::operator=(av_dict &&ori) noexcept
            {
                if (this != &ori)
                    avdic = std::exchange(ori.avdic, nullptr);
                return *this;
            }

            inline av_dict::~av_dict()
            {
                if (avdic)
                    av_dict_free(&avdic);
            }

            inline AVDictionary** av_dict::get()
            {
                return avdic ? &avdic: nullptr;
            }

            inline std::size_t av_dict::size() const
            {
                return avdic ? av_dict_count(avdic) : 0;
            }

            inline void av_dict::print() const
            {
                if (avdic)
                {
                    AVDictionaryEntry *tag = nullptr;
                    while ((tag = av_dict_get(avdic, "", tag, AV_DICT_IGNORE_SUFFIX)))
                        printf("%s : %s\n", tag->key, tag->value);
                }
            }
        
// ---------------------------------------------------------------------------------------------------

            inline void resizer::reset (
                const int src_h_, const int src_w_, const AVPixelFormat src_fmt_,
                const int dst_h_, const int dst_w_, const AVPixelFormat dst_fmt_
            )
            {
                auto this_params = std::tie(src_h,  src_w,  src_fmt,  dst_h,  dst_w,  dst_fmt);
                auto new_params  = std::tie(src_h_, src_w_, src_fmt_, dst_h_, dst_w_, dst_fmt_);

                if (this_params != new_params)
                {
                    this_params = new_params;

                    imgConvertCtx = nullptr;

                    if (std::tie(src_h, src_w, src_fmt) != 
                        std::tie(dst_h, dst_w, dst_fmt))
                    {
                        imgConvertCtx.reset(sws_getContext(src_w, src_h, src_fmt,
                                                           dst_w, dst_h, dst_fmt,
                                                           SWS_FAST_BILINEAR, NULL, NULL, NULL));
                    }
                }
            }

            inline void resizer::resize (
                const frame& src,
                const int dst_h_, const int dst_w_, const AVPixelFormat dst_pixfmt_,
                frame& dst
            )
            {
                DLIB_CASSERT(src.is_image(), "src.is_image() == false");

                const bool is_same_object = std::addressof(src) == std::addressof(dst);

                reset(src.height(), src.width(), src.pixfmt(),
                      dst_h_, dst_w_, dst_pixfmt_);

                if (imgConvertCtx)
                {
                    frame tmp;
                    frame* ptr = std::addressof(dst);

                    if (is_same_object ||
                        !dst.is_image() ||
                        std::make_tuple(dst.height(), dst.width(), dst.pixfmt()) !=
                        std::make_tuple(dst_h,        dst_w,       dst_fmt))
                    {
                        tmp = frame(dst_h, dst_w, dst_fmt, src.get_timestamp());
                        ptr = std::addressof(tmp);
                    }

                    sws_scale(imgConvertCtx.get(),
                               src.get_frame().data,  src.get_frame().linesize, 0, src.height(),
                              ptr->get_frame().data, ptr->get_frame().linesize);

                    if (ptr != std::addressof(dst))
                        dst = std::move(tmp);
                }
                else if (!is_same_object)
                {
                    dst = src;
                }
            }

            inline void resizer::resize(
                const frame& src,
                frame& dst
            )
            {
                resize(src, dst_h, dst_w, dst_fmt, dst);
            }

// ---------------------------------------------------------------------------------------------------

            inline void resampler::reset(
                const int src_sample_rate_, const uint64_t src_channel_layout_, const AVSampleFormat src_fmt_,
                const int dst_sample_rate_, const uint64_t dst_channel_layout_, const AVSampleFormat dst_fmt_
            )
            {
                auto this_params = std::tie(src_sample_rate,
                                            src_channel_layout,
                                            src_fmt,
                                            dst_sample_rate,
                                            dst_channel_layout,
                                            dst_fmt);
                auto new_params  = std::tie(src_sample_rate_,
                                            src_channel_layout_,
                                            src_fmt_,
                                            dst_sample_rate_,
                                            dst_channel_layout_,
                                            dst_fmt_);

                if (this_params != new_params)
                {
                    this_params = new_params;

                    audioResamplerCtx = nullptr;

                    if (std::tie(src_sample_rate, src_channel_layout, src_fmt) !=
                        std::tie(dst_sample_rate, dst_channel_layout, dst_fmt))
                    {
                        audioResamplerCtx.reset(swr_alloc_set_opts(NULL,
                                                                dst_channel_layout_, dst_fmt_, dst_sample_rate_,
                                                                src_channel_layout_, src_fmt_, src_sample_rate_,
                                                                0, NULL));
                        int ret = 0;
                        if ((ret = swr_init(audioResamplerCtx.get())) < 0)
                        {
                            audioResamplerCtx = nullptr;
                            throw std::runtime_error("swr_init() failed : " + get_av_error(ret));
                        }
                    }
                }
            }

            inline void resampler::resize(
                const frame&            src,
                const int               dst_sample_rate_,
                const uint64_t          dst_channel_layout_,
                const AVSampleFormat    dst_samplefmt_,
                frame&                  dst
            )
            {
                using namespace std::chrono;

                DLIB_CASSERT(src.is_audio(), "src.is_audio() == false");

                const bool is_same_object = std::addressof(src) == std::addressof(dst);

                reset(src.sample_rate(),         src.layout(), src.samplefmt(),
                      dst_sample_rate_,  dst_channel_layout_,  dst_samplefmt_);

                if (audioResamplerCtx)
                {
                    av_ptr<AVFrame> tmp = make_avframe();
                    tmp->sample_rate    = dst_sample_rate;
                    tmp->channel_layout = dst_channel_layout;
                    tmp->format         = (int)dst_fmt;

                    const int ret = swr_convert_frame(audioResamplerCtx.get(), tmp.get(), &src.get_frame());
                    if (ret < 0)
                        throw std::runtime_error("swr_convert_frame() failed : " + get_av_error(ret));

                    dst.f           = std::move(tmp);
                    dst.f->pts      = tracked_samples;
                    dst.timestamp   = system_clock::time_point{nanoseconds{av_rescale_q(tracked_samples,
                                                                                        {1, dst_sample_rate},
                                                                                        {nanoseconds::period::num, nanoseconds::period::den})}};
                    tracked_samples += dst.nsamples();

                }
                else if (!is_same_object)
                {
                    dst = src;
                }
            }

            inline void resampler::resize(
                const frame& src,
                frame& dst
            )
            {
                resize(src, dst_sample_rate, dst_channel_layout, dst_fmt, dst);
            }

// ---------------------------------------------------------------------------------------------------

            inline audio_fifo::audio_fifo(
                const int            codec_frame_size_,
                const AVSampleFormat sample_format_,
                const int            nchannels_
            ) : frame_size(codec_frame_size_),
                fmt(sample_format_),
                nchannels(nchannels_)
            {
                if (frame_size > 0)
                {
                    fifo.reset(av_audio_fifo_alloc(fmt, nchannels, frame_size));
                    if (!fifo)
                        throw std::runtime_error("av_audio_fifo_alloc() failed");
                }
            }

            inline std::vector<frame> audio_fifo::push_pull(
                frame&& in
            )
            {
                using namespace std::chrono;
                assert(in.is_audio());

                std::vector<frame> outs;

                //check that the configuration hasn't suddenly changed this would be exceptional
                auto current_params = std::tie(fmt, nchannels);
                auto new_params     = std::make_tuple(in.samplefmt(), in.nchannels());

                if (current_params != new_params)
                    throw std::runtime_error("new audio frame params differ from first ");

                if (frame_size == 0)
                {
                    outs.push_back(std::move(in));
                }
                else
                {
                    if (av_audio_fifo_write(fifo.get(), (void**)in.get_frame().data, in.nsamples()) != in.nsamples())
                        throw std::runtime_error("av_audio_fifo_write() failed to write all samples");

                    while (av_audio_fifo_size(fifo.get()) >= frame_size)
                    {
                        const system_clock::time_point timestamp{nanoseconds{av_rescale_q(
                                sample_count,
                                {1, in.sample_rate()},
                                {nanoseconds::period::num, nanoseconds::period::den})}};

                        frame out(in.sample_rate(), frame_size, in.layout(), in.samplefmt(), timestamp);

                        if (av_audio_fifo_read(fifo.get(), (void**)out.get_frame().data, out.nsamples()) != out.nsamples())
                            throw std::runtime_error("av_audio_fifo_read() failed to read all requested samples");

                        sample_count += out.nsamples();
                        outs.push_back(std::move(out));
                    }
                }

                return outs;
            }

// ---------------------------------------------------------------------------------------------------

        }

// ---------------------------------------------------------------------------------------------------

        inline frame::frame(
            int             h,
            int             w,
            AVPixelFormat   pixfmt,
            int             sample_rate,
            int             nb_samples,
            uint64_t        channel_layout,
            AVSampleFormat  samplefmt,
            std::chrono::system_clock::time_point timestamp_
        )
        {
            using namespace details;

            f = make_avframe();
            f->height           = h;
            f->width            = w;
            f->sample_rate      = sample_rate;
            f->channel_layout   = channel_layout;
            f->nb_samples       = nb_samples;
            f->format           = h > 0 && w > 0 ? (int)pixfmt : (int)samplefmt;
            timestamp           = timestamp_;

            // The ffmpeg documentation recommends you always use align==0.
            // However, in ffmpeg 3.2, there is a bug where if you do that, data buffers don't get allocated.
            // So workaround is to manually set align==32
            // Not ideal, but i've checked the source code in ffmpeg 3.3, 4.4 and 5.0 libavutil/frame.c
            // and in every case, if align==0, then align=32
            const int align = 32;

            const int ret = av_frame_get_buffer(f.get(), align);
            if (ret < 0)
            {
                f = nullptr;
                throw std::runtime_error("av_frame_get_buffer() failed : " + get_av_error(ret));
            }

    //        ret = av_frame_make_writable(obj.frame.get());
    //        if (ret < 0)
    //        {
    //            obj.frame.reset(nullptr);
    //            throw std::runtime_error("av_frame_make_writable() failed : " + get_av_error(ret));
    //        }

            if (is_audio())
                f->pts = av_rescale_q(timestamp.time_since_epoch().count(),
                                      {decltype(timestamp)::period::num, (decltype(timestamp)::period::den)},
                                      {1, f->sample_rate});
        }

        inline frame::frame(
            int h,
            int w,
            AVPixelFormat fmt,
            std::chrono::system_clock::time_point timestamp
        ) : frame(h, w, fmt, 0, 0, 0, AV_SAMPLE_FMT_NONE, timestamp)
        {
        }

        inline frame::frame(
            int             sample_rate,
            int             nb_samples,
            uint64_t        channel_layout,
            AVSampleFormat  fmt,
            std::chrono::system_clock::time_point timestamp
        ) : frame(0,0,AV_PIX_FMT_NONE, sample_rate, nb_samples, channel_layout, fmt, timestamp)
        {
        }

        inline frame::frame(const frame &ori)
        {
            copy_from(ori);
        }

        inline frame& frame::operator=(const frame& ori)
        {
            if (this != &ori)
                copy_from(ori);
            return *this;
        }

        inline void frame::copy_from(const frame& ori)
        {
            if (ori.is_empty())
            {
                frame empty{std::move(*this)};
            }
            else
            {
                const bool same_image_dims =
                        is_image() &&
                        std::tie(    f->height,     f->width,     f->format) ==
                        std::tie(ori.f->height, ori.f->width, ori.f->format);

                const bool same_audio_dims =
                        is_audio() &&
                        std::tie(    f->sample_rate,     f->nb_samples,     f->channel_layout,     f->format) ==
                        std::tie(ori.f->sample_rate, ori.f->nb_samples, ori.f->channel_layout, ori.f->format);

                if (!same_image_dims && !same_audio_dims)
                {
                    *this = std::move(frame(ori.f->height,
                                            ori.f->width,
                                            (AVPixelFormat)ori.f->format,
                                            ori.f->sample_rate,
                                            ori.f->nb_samples,
                                            ori.f->channel_layout,
                                            (AVSampleFormat)ori.f->format,
                                            ori.timestamp));
                }

                av_frame_copy(f.get(), ori.f.get());
                av_frame_copy_props(f.get(), ori.f.get());
            }
        }

        inline bool frame::is_image() const noexcept
        {
            return f && f->width > 0 && f->height > 0 && f->format != AV_PIX_FMT_NONE;
        }

        inline bool frame::is_audio() const noexcept
        {
            return f && f->nb_samples > 0 && f->channel_layout > 0 && f->sample_rate > 0 && f->format != AV_SAMPLE_FMT_NONE;
        }

        inline bool            frame::is_empty()   const noexcept { return !is_image() && !is_audio(); }
        inline AVPixelFormat   frame::pixfmt()     const noexcept { return is_image() ? (AVPixelFormat)f->format : AV_PIX_FMT_NONE; }
        inline int             frame::height()     const noexcept { return is_image() ? f->height : 0; }
        inline int             frame::width()      const noexcept { return is_image() ? f->width : 0; }
        inline AVSampleFormat  frame::samplefmt()  const noexcept { return is_audio() ? (AVSampleFormat)f->format : AV_SAMPLE_FMT_NONE; }
        inline int             frame::nsamples()   const noexcept { return is_audio() ? f->nb_samples : 0; }
        inline uint64_t        frame::layout()     const noexcept { return is_audio() ? f->channel_layout : 0; }
        inline int             frame::nchannels()  const noexcept { return is_audio() ? av_get_channel_layout_nb_channels(f->channel_layout) : 0; }
        inline int             frame::sample_rate() const noexcept{ return is_audio() ? f->sample_rate : 0; }

        inline std::chrono::system_clock::time_point frame::get_timestamp() const noexcept { return timestamp; }

        inline const AVFrame&  frame::get_frame() const { DLIB_CASSERT(f, "is_empty() == true"); return *f; }
        inline AVFrame&        frame::get_frame()       { DLIB_CASSERT(f, "is_empty() == true"); return *f; }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<std::string> list_protocols()
        {
            const bool init = details::register_ffmpeg::get(); // Don't let this get optimized away
            std::vector<std::string> protocols;
            void* opaque = nullptr;
            const char* name = 0;
            while (init && (name = avio_enum_protocols(&opaque, 0)))
                protocols.emplace_back(name);

            opaque  = nullptr;
            name    = 0;

            while (init && (name = avio_enum_protocols(&opaque, 1)))
                protocols.emplace_back(name);

            return protocols;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<std::string> list_demuxers()
        {
            const bool init = details::register_ffmpeg::get(); // Don't let this get optimized away
            std::vector<std::string> demuxers;
            const AVInputFormat* demuxer = nullptr;

    #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
            // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L86
            while (init && (demuxer = av_iformat_next(demuxer)))
    #else
            void* opaque = nullptr;
            while (init && (demuxer = av_demuxer_iterate(&opaque)))
    #endif
                demuxers.push_back(demuxer->name);

            return demuxers;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<std::string> list_muxers()
        {
            const bool init = details::register_ffmpeg::get(); // Don't let this get optimized away
            std::vector<std::string> muxers;
            const AVOutputFormat* muxer = nullptr;

    #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
            // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L86
            while (init && (muxer = av_oformat_next(muxer)))
    #else
            void* opaque = nullptr;
            while (init && (muxer = av_muxer_iterate(&opaque)))
    #endif
                muxers.push_back(muxer->name);
        
            return muxers;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<codec_details> list_codecs()
        {
            const bool init = details::register_ffmpeg::get(); // Don't let this get optimized away
            std::vector<codec_details> details;

    #if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 10, 100)
            // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L91
            AVCodec* codec = nullptr;
            while (init && (codec = av_codec_next(codec)))
    #else
            const AVCodec* codec = nullptr;
            void* opaque = nullptr;
            while (init && (codec = av_codec_iterate(&opaque)))
    #endif
            {
                codec_details detail;
                detail.codec_name = codec->name;
                detail.supports_encoding = av_codec_is_encoder(codec);
                detail.supports_decoding = av_codec_is_decoder(codec);
                details.push_back(std::move(detail));
            }

            //sort
            std::sort(details.begin(), details.end(), [](const codec_details& a, const codec_details& b) {return a.codec_name < b.codec_name;});
            //merge
            for (size_t i = 0 ; i < details.size() ; ++i)
            {
                for (size_t j = i + 1 ; j < details.size() ; ++j)
                {
                    if (details[i].codec_name == details[j].codec_name)
                    {
                        details[i].supports_encoding |= details[j].supports_encoding;
                        details[i].supports_decoding |= details[j].supports_decoding;
                        details[j] = {};
                    }
                }
            }
            
            details.erase(std::remove_if(details.begin(), details.end(), [](const auto& d) {return d.codec_name.empty();}), details.end());

            return details;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<device_details> list_input_devices()
        {
            const bool init = details::register_ffmpeg::get(); // Don't let this get optimized away
            std::vector<device_details> devices;
            AVInputFormat* device = nullptr;

            details::av_ptr<AVDeviceInfoList> managed;

            const auto iter = [&](AVInputFormat* device)
            {
                device_details details;
                details.device_type = std::string(device->name);

                AVDeviceInfoList* device_list = nullptr;
                avdevice_list_input_sources(device, nullptr, nullptr, &device_list);
                managed.reset(device_list);

                if (device_list)
                {
                    for (int i = 0 ; i < device_list->nb_devices ; ++i)
                    {
                        device_details::instance instance;
                        instance.name        = std::string(device_list->devices[i]->device_name);
                        instance.description = std::string(device_list->devices[i]->device_description);
                        details.devices.push_back(std::move(instance));
                    }
                }

                devices.push_back(std::move(details));
            };

            while (init && (device = av_input_audio_device_next(device)))
                iter(device);

            device = nullptr;

            while (init && (device = av_input_video_device_next(device)))
                iter(device);

            return devices;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<device_details> list_output_devices()
        {
            const bool init = details::register_ffmpeg::get(); // Don't let this get optimized away
            std::vector<device_details> devices;
            AVOutputFormat* device = nullptr;

            details::av_ptr<AVDeviceInfoList> managed;

            const auto iter = [&](AVOutputFormat* device)
            {
                device_details details;
                details.device_type = std::string(device->name);

                AVDeviceInfoList* device_list = nullptr;
                avdevice_list_output_sinks(device, nullptr, nullptr, &device_list);
                managed.reset(device_list);

                if (device_list)
                {
                    for (int i = 0 ; i < device_list->nb_devices ; ++i)
                    {
                        device_details::instance instance;
                        instance.name        = std::string(device_list->devices[i]->device_name);
                        instance.description = std::string(device_list->devices[i]->device_description);
                        details.devices.push_back(std::move(instance));
                    }
                }

                devices.push_back(std::move(details));
            };

            while (init && (device = av_output_audio_device_next(device)))
                iter(device);

            device = nullptr;

            while (init && (device = av_output_video_device_next(device)))
                iter(device);

            return devices;
        }

// ---------------------------------------------------------------------------------------------------

        template <
          class ImageContainer, 
          is_image_check<ImageContainer> = true
        >
        void convert(const frame& f, ImageContainer& image)
        {
            using pixel = typename image_traits<ImageContainer>::pixel_type;
            
            DLIB_ASSERT(f.is_image(), "frame isn't an image type");
            DLIB_ASSERT(f.pixfmt() == pix_traits<pixel>::fmt, "frame doesn't have correct format");
        
            image.set_size(f.height(), f.width());
            const size_t imgsize = image.nr()*image.nc()*sizeof(pixel);
            const size_t expsize = av_image_get_buffer_size(f.pixfmt(), f.width(), f.height(), 1);
            DLIB_ASSERT(imgsize == expsize, "image size in bytes != expected buffer size required by ffmpeg to do a copy");
            (void)imgsize;
            (void)expsize;

            const int ret = av_image_copy_to_buffer((uint8_t*)image.begin(), 
                                                    imgsize, 
                                                    f.get_frame().data, 
                                                    f.get_frame().linesize, 
                                                    f.pixfmt(), 
                                                    f.width(), 
                                                    f.height(), 
                                                    1);    
            
            DLIB_ASSERT(ret == (int)expsize, "av_image_copy_to_buffer()  error : " << details::get_av_error(ret));
            (void)ret;
        }

        template<
          class ImageContainer, 
          is_image_check<ImageContainer> = true
        >
        void convert(const ImageContainer& img, frame& f)
        {
            using pixel = typename image_traits<ImageContainer>::pixel_type;

            if (f.height() != img.nr() ||
                f.width()  != img.nc() ||
                f.pixfmt() != pix_traits<pixel>::fmt)
            {
                f = frame(img.nr(), img.nc(), pix_traits<pixel>::fmt, {});
            }

            const size_t imgsize            = img.nr()*img.nc()*sizeof(pixel);
            int         src_linesizes[4]    = {0};
            uint8_t*    src_pointers[4]     = {nullptr};

            const int ret = av_image_fill_arrays(src_pointers, src_linesizes, (uint8_t*)img.begin(), f.pixfmt(), f.width(), f.height(), 1);
            DLIB_ASSERT(ret == imgsize, "av_image_fill_arrays()  error : " << details::get_av_error(ret));
            (void)ret;
            
            av_image_copy(f.get_frame().data,
                          f.get_frame().linesize,
                          (const uint8_t**)src_pointers,
                          src_linesizes,
                          f.pixfmt(),
                          f.width(),
                          f.height());
        }

        template<class SampleFmt, std::size_t Channels>
        inline void convert(const frame& f, audio<SampleFmt, Channels>& obj)
        {
            using sample = typename audio<SampleFmt, Channels>::sample;

            DLIB_ASSERT(f.is_audio(), "frame must be of audio type");
            DLIB_ASSERT(f.samplefmt() == sample_traits<SampleFmt>::fmt, "audio buffer has wrong format for this type. Make sure correct args are passed to constructor of decoder/demuxer/encoder/muxer");
            DLIB_ASSERT(f.nchannels() == Channels, "wrong number of channels");

            obj.timestamp   = f.get_timestamp();
            obj.sample_rate = f.sample_rate();
            obj.samples.resize(f.nsamples());

            uint8_t* dst_pointers[8]    = {nullptr}; 
            int      dst_linesize       = 0;

            const int bufsize  = obj.samples.size()*sizeof(sample);
            const int expsize1 = av_samples_get_buffer_size(&dst_linesize, f.nchannels(), f.nsamples(), f.samplefmt(), 1);
            const int expsize2 = av_samples_fill_arrays(dst_pointers, &dst_linesize, (uint8_t*)obj.samples.data(), f.nchannels(), f.nsamples(), f.samplefmt(), 1);
            DLIB_ASSERT(bufsize == expsize1, "audio size in bytes != expected buffer size required by ffmpeg to do a copy");
            DLIB_ASSERT(expsize1 == expsize2, "inconsistent audio buffer sizes returned by ffmpeg");
            (void)bufsize;
            (void)expsize1;
            (void)expsize2;

            av_samples_copy(dst_pointers, f.get_frame().data, 0, 0, f.nsamples(), f.nchannels(), f.samplefmt());
        }

        template<class SampleFmt, std::size_t Channels>
        inline void convert(const audio<SampleFmt, Channels>& obj, frame& f)
        {
            using sample = typename audio<SampleFmt, Channels>::sample;

            if (f.samplefmt()   != sample_traits<SampleFmt>::fmt ||
                f.layout()      != get_layout_from_channels(Channels) ||
                f.sample_rate() != obj.sample_rate ||
                f.nsamples()    != obj.samples.size())
            {
                f = frame(obj.sample_rate, 
                          obj.samples.size(), 
                          get_layout_from_channels(Channels), 
                          sample_traits<SampleFmt>::fmt, 
                          obj.timestamp);
            }

            uint8_t* src_pointers[8]    = {nullptr}; 
            int      src_linesize       = 0;

            const int bufsize  = obj.samples.size()*sizeof(sample);
            const int expsize1 = av_samples_get_buffer_size(&src_linesize, f.nchannels(), f.nsamples(), f.samplefmt(), 1);
            const int expsize2 = av_samples_fill_arrays(src_pointers, &src_linesize, (const uint8_t*)obj.samples.data(), f.nchannels(), f.nsamples(), f.samplefmt(), 1);
            DLIB_ASSERT(bufsize == expsize1, "audio size in bytes != expected buffer size required by ffmpeg to do a copy");
            DLIB_ASSERT(expsize1 == expsize2, "inconsistent audio buffer sizes returned by ffmpeg");
            (void)bufsize;
            (void)expsize1;
            (void)expsize2;

            av_samples_copy(f.get_frame().data, src_pointers, 0, 0, f.nsamples(), f.nchannels(), f.samplefmt());
        }

// ---------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_FFMPEG_UTILS
