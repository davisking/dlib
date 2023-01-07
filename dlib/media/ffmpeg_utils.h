// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_UTILS
#define DLIB_FFMPEG_UTILS

#include <cstdint>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <memory>
#include "../smart_pointers/observer_ptr.h"
#include "../array2d.h"
#include "../pixel.h"
#include "../type_safe_union.h"

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
    namespace details
    {
    
    // ---------------------------------------------------------------------------------------------------

        extern const bool FFMPEG_INITIALIZED;

    // ---------------------------------------------------------------------------------------------------

        std::string get_av_error(int ret);
        std::string get_pixel_fmt_str(AVPixelFormat fmt);
        std::string get_audio_fmt_str(AVSampleFormat fmt);
        std::string get_channel_layout_str(uint64_t layout);

    // ---------------------------------------------------------------------------------------------------

        struct rational
        {
            int num;
            int denom;
            float get();
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
        };
    
    // ---------------------------------------------------------------------------------------------------

    }

    // ---------------------------------------------------------------------------------------------------

    template<class AVObject>
    using av_ptr = std::unique_ptr<AVObject, details::av_deleter>;

    template<class AVObject>
    using av_raw = dlib::observer_ptr<AVObject>;

    // ---------------------------------------------------------------------------------------------------

    namespace details
    {

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

        av_ptr<AVFrame>  make_avframe();
        av_ptr<AVPacket> make_avpacket();
    
    // ---------------------------------------------------------------------------------------------------

        class sw_image_resizer;
        class sw_audio_resampler;
        class sw_audio_fifo;
        class decoder_extractor;
    
    // ---------------------------------------------------------------------------------------------------
    }

    // ---------------------------------------------------------------------------------------------------

    class Frame;
    class audio_frame;
    Frame convert(const audio_frame& frame);
    Frame convert(const array2d<rgb_pixel>& frame);

    class Frame
    {
    public:
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class wraps AVFrame* into a std::unique_ptr with an appropriate deleter.
                It also has a std::chrono timestamp which closely matches the AVFrame's internal pts.
                It has a bunch of helper functions for retrieving the frame's properties.
                See ffmeg documentation for AVFrame.
                It contains frame data, which may be image, audio or other.
                The format and layout of the frame is not necessarily RGB (for image) or stereo s16 (for audio).
                It depends on codec configuration.
                It is up to the user to appropriately read the data using get_frame().data, get_frame().linesize.
                There are however helper functions convert() with overloads to convert to/from dlib objects.
        !*/

        Frame()                         = default;
        Frame(Frame&& ori)              = default;
        Frame& operator=(Frame&& ori)   = default;
        Frame(const Frame& ori);
        Frame& operator=(const Frame& ori);

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
        const AVFrame& get_frame() const noexcept;

    private:
        // Users should only be able to access const functions
        // However, the implementation details should be able to access non-const functions and data.
        friend class details::sw_image_resizer;
        friend class details::sw_audio_resampler;
        friend class details::sw_audio_fifo;
        friend class details::decoder_extractor;
        friend Frame convert(const audio_frame& frame);
        friend Frame convert(const array2d<rgb_pixel>& frame);

        static Frame make(
            int h,
            int w,
            AVPixelFormat pixfmt,
            int sample_rate,
            int nb_samples,
            uint64_t channel_layout,
            AVSampleFormat samplefmt,
            std::chrono::system_clock::time_point timestamp
        );

        static Frame make_image(
            int h,
            int w,
            AVPixelFormat pixfmt,
            std::chrono::system_clock::time_point timestamp_us
        );

        static Frame make_audio(
            int sample_rate,
            int nb_samples,
            uint64_t channel_layout,
            AVSampleFormat samplefmt,
            std::chrono::system_clock::time_point timestamp
        );

        void copy_from(const Frame& other);

        av_ptr<AVFrame> frame;
        std::chrono::system_clock::time_point timestamp;
    };

    // ---------------------------------------------------------------------------------------------------

    namespace details
    {
    
    // ---------------------------------------------------------------------------------------------------

        class sw_image_resizer
        {
        public:
            void reset(
                int src_h, int src_w, AVPixelFormat src_fmt,
                int dst_h, int dst_w, AVPixelFormat dst_fmt
            );

            void resize(
                const Frame &src,
                int dst_h, int dst_w, AVPixelFormat dst_fmt,
                Frame &dst
            );

            void resize(
                const Frame &src,
                Frame &dst
            );

            int get_src_h() const;
            int get_src_w() const;
            AVPixelFormat get_src_fmt() const;
            int get_dst_h() const;
            int get_dst_w() const;
            AVPixelFormat get_dst_fmt() const;

        private:

            int             _src_h      = 0;
            int             _src_w      = 0;
            AVPixelFormat   _src_fmt    = AV_PIX_FMT_NONE;
            int             _dst_h      = 0;
            int             _dst_w      = 0;
            AVPixelFormat   _dst_fmt    = AV_PIX_FMT_NONE;
            av_ptr<SwsContext> _imgConvertCtx;
        };

    // ---------------------------------------------------------------------------------------------------

        class sw_audio_resampler
        {
        public:
            void reset(
                int src_sample_rate, uint64_t src_channel_layout, AVSampleFormat src_fmt,
                int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt
            );

            void resize(
                const Frame &src,
                int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt,
                Frame &dst
            );

            void resize(
                const Frame &src,
                Frame &dst
            );

            int             get_src_rate()      const;
            uint64_t        get_src_layout()    const;
            AVSampleFormat  get_src_fmt()       const;
            int             get_dst_rate()      const;
            uint64_t        get_dst_layout()    const;
            AVSampleFormat  get_dst_fmt()       const;

        private:

            int             src_sample_rate_    = 0;
            uint64_t        src_channel_layout_ = AV_CH_LAYOUT_STEREO;
            AVSampleFormat  src_fmt_            = AV_SAMPLE_FMT_NONE;

            int             dst_sample_rate_    = 0;
            uint64_t        dst_channel_layout_ = AV_CH_LAYOUT_STEREO;
            AVSampleFormat  dst_fmt_            = AV_SAMPLE_FMT_NONE;

            av_ptr<SwrContext>  audioResamplerCtx_;
            uint64_t            tracked_samples_ = 0;
        };

    // ---------------------------------------------------------------------------------------------------

        class sw_audio_fifo
        {
        public:
            sw_audio_fifo() = default;
            sw_audio_fifo(
                int codec_frame_size,
                int sample_format,
                int nchannels
            );

            std::vector<Frame> push_pull(
                Frame &&in
            );

        private:

            int frame_size          = 0;
            int fmt                 = 0;
            int channels            = 0;
            uint64_t sample_count   = 0;
            av_ptr<AVAudioFifo> fifo;
        };

    // ---------------------------------------------------------------------------------------------------

    }

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

    std::vector<std::string> ffmpeg_list_protocols();
    /*!
        ensures
            - returns a list of all registered ffmpeg protocols
    !*/
    std::vector<std::string> ffmpeg_list_demuxers();
    /*!
        ensures
            - returns a list of all registered ffmpeg demuxers
    !*/

    std::vector<std::string> ffmpeg_list_muxers();
    /*!
        ensures
            - returns a list of all registered ffmpeg muxers
    !*/

    struct codec_details
    {
        std::string codec_name;
        bool supports_encoding;
        bool supports_decoding;
    };
    std::vector<codec_details> ffmpeg_list_codecs();
    /*!
        ensures
            - returns a list of all registered ffmpeg codecs with information on whether decoding and/or encoding is supported.
              Note that not all codecs support encoding, unless your installation of ffmpeg is built with third party library
              dependencies like libx264, libx265, etc.
    !*/

    // ---------------------------------------------------------------------------------------------------

    type_safe_union<array2d<rgb_pixel>, audio_frame> convert(const Frame& frame);
    /*!
        ensures
            - converts a Frame object into dlib objects if possible, i.e. if the format and layout is already correct
    !*/

    Frame convert(const array2d<rgb_pixel>& frame);
    /*!
        ensures
            - converts a dlib image into a frame object
    !*/

    Frame convert(const audio_frame& frame);
    /*!
        ensures
            - converts a dlib audio frame into a frame object
    !*/

    // ---------------------------------------------------------------------------------------------------
}

#endif //DLIB_FFMPEG_UTILS