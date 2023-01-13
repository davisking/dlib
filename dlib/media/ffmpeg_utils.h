// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_UTILS
#define DLIB_FFMPEG_UTILS

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

#include <cstdint>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <memory>
#include "../array2d.h"
#include "../pixel.h"
#include "../type_safe_union.h"
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

            class decoder_extractor;
        
        // ---------------------------------------------------------------------------------------------------

            extern const bool FFMPEG_INITIALIZED;

        // ---------------------------------------------------------------------------------------------------

            std::string get_av_error(int ret);
            std::string get_pixel_fmt_str(AVPixelFormat fmt);
            std::string get_audio_fmt_str(AVSampleFormat fmt);
            std::string get_channel_layout_str(uint64_t layout);

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
            };

            template<class AVObject>
            using av_ptr = std::unique_ptr<AVObject, details::av_deleter>;
        
        // ---------------------------------------------------------------------------------------------------

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

                int             get_src_h()     const noexcept;
                int             get_src_w()     const noexcept;
                AVPixelFormat   get_src_fmt()   const noexcept;
                int             get_dst_h()     const noexcept;
                int             get_dst_w()     const noexcept;
                AVPixelFormat   get_dst_fmt()   const noexcept;

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

                int             get_src_rate()      const noexcept;
                uint64_t        get_src_layout()    const noexcept;
                AVSampleFormat  get_src_fmt()       const noexcept;
                int             get_dst_rate()      const noexcept;
                uint64_t        get_dst_layout()    const noexcept;
                AVSampleFormat  get_dst_fmt()       const noexcept;

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

            class sw_audio_fifo
            {
            public:
                sw_audio_fifo() = default;

                sw_audio_fifo(
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

        }

        // ---------------------------------------------------------------------------------------------------

        extern const int DLIB_LIBAVFORMAT_MAJOR_VERSION;
        extern const int DLIB_LIBAVFORMAT_MINOR_VERSION;
        extern const int DLIB_LIBAVFORMAT_MICRO_VERSION;

        extern const int DLIB_LIBAVCODEC_MAJOR_VERSION;
        extern const int DLIB_LIBAVCODEC_MINOR_VERSION;
        extern const int DLIB_LIBAVCODEC_MICRO_VERSION;

        extern const int DLIB_LIBAVUTIL_MAJOR_VERSION;
        extern const int DLIB_LIBAVUTIL_MINOR_VERSION;
        extern const int DLIB_LIBAVUTIL_MICRO_VERSION;

        extern const int DLIB_LIBAVDEVICE_MAJOR_VERSION;
        extern const int DLIB_LIBAVDEVICE_MINOR_VERSION;
        extern const int DLIB_LIBAVDEVICE_MICRO_VERSION;

        void check_ffmpeg_versions(
            int avformat_major = LIBAVFORMAT_VERSION_MAJOR,
            int avformat_minor = LIBAVFORMAT_VERSION_MINOR,
            int avformat_micro = LIBAVFORMAT_VERSION_MICRO,
            int avcodec_major  = LIBAVCODEC_VERSION_MAJOR,
            int avcodec_minor  = LIBAVCODEC_VERSION_MINOR,
            int avcodec_micro  = LIBAVCODEC_VERSION_MICRO,
            int avutil_major   = LIBAVUTIL_VERSION_MAJOR,
            int avutil_minor   = LIBAVUTIL_VERSION_MINOR,
            int avutil_micro   = LIBAVUTIL_VERSION_MICRO,
            int avdevice_major = LIBAVDEVICE_VERSION_MAJOR,
            int avdevice_minor = LIBAVDEVICE_VERSION_MINOR,
            int avdevice_micro = LIBAVDEVICE_VERSION_MICRO
        );

        // ---------------------------------------------------------------------------------------------------

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

        struct audio_frame
        {
            struct sample
            {
                int16_t ch1 = 0;
                int16_t ch2 = 0;
            };

            std::vector<sample>                     samples;
            float                                   sample_rate{0};
            std::chrono::system_clock::time_point   timestamp{};
        };

        // ---------------------------------------------------------------------------------------------------

        struct codec_details
        {
            std::string codec_name;
            bool supports_encoding;
            bool supports_decoding;
        };

        std::vector<std::string>   ffmpeg_list_protocols();
        std::vector<std::string>   ffmpeg_list_demuxers();
        std::vector<std::string>   ffmpeg_list_muxers();
        std::vector<codec_details> ffmpeg_list_codecs();

        // ---------------------------------------------------------------------------------------------------

        void convert(const frame& f, type_safe_union<array2d<rgb_pixel>, audio_frame>& obj);
        void convert(const frame& f, array2d<rgb_pixel>& obj);
        void convert(const frame& f, audio_frame& obj);
        type_safe_union<array2d<rgb_pixel>, audio_frame> convert(const frame& f);

        frame convert(const array2d<rgb_pixel>& img);
        frame convert(const audio_frame& audio);

        // ---------------------------------------------------------------------------------------------------
    }
}

#endif //DLIB_FFMPEG_UTILS