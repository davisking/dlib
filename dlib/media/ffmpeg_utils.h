// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_UTILS
#define DLIB_FFMPEG_UTILS

#include "../test_for_odr_violations.h"

#ifndef DLIB_USE_FFMPEG
static_assert(false, "This version of dlib isn't built with the FFMPEG wrappers");
#endif

#include <cstdint>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <string>
#include <chrono>
#include <vector>
#include <array>
#include <unordered_map>
#include <iostream>
#include "../image_processing/generic_image.h"
#include "../pixel.h"
#include "../assert.h"
#include "ffmpeg_details.h"

namespace dlib
{
    namespace ffmpeg
    {

// ---------------------------------------------------------------------------------------------------

        std::string get_pixel_fmt_str(AVPixelFormat fmt);
        /*!
            ensures
                - Returns a string description of AVPixelFormat
        !*/

        std::string get_audio_fmt_str(AVSampleFormat fmt);
        /*!
            ensures
                - Returns a string description of AVSampleFormat
        !*/

        std::string get_channel_layout_str(uint64_t layout);
        /*!
            ensures
                - Returns a string description of a channel layout, where layout is e.g. AV_CH_LAYOUT_STEREO
        !*/

// ---------------------------------------------------------------------------------------------------

        dlib::logger& logger_ffmpeg();
        /*!
            ensures
                - Returns a global logger used by the internal ffmpeg libraries. 
                - You may set the logging level using .set_level() to supress or enable certain logs.
        !*/

        dlib::logger& logger_dlib_wrapper();
        /*!
            ensures
                - Returns a global logger used by dlib's ffmpeg wrappers.
                - You may set the logging level using .set_level() to supress or enable certain logs.
        !*/

// ---------------------------------------------------------------------------------------------------

        namespace details { class resampler; }

        class frame
        {
        public:
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class wraps AVFrame* into a std::unique_ptr with an appropriate deleter.
                    It also has a std::chrono timestamp which closely matches the AVFrame's internal pts.
                    It has a bunch of helper functions for retrieving the frame's properties.
                    We strongly recommend you read ffmegs documentation on AVFrame.

                    FFmpeg's AVFrame object is basically a type-erased frame object, which can contain, 
                    image, audio or other types of streamable data.
                    The pixel format (image), sample format (audio), number of channels (audio), 
                    pixel/sample type (u8, s16, f32, etc) are also erased and defined as runtime parameters.

                    Users should avoid using this object directly if they can, instead use the conversion functions
                    dlib::ffmpeg::convert() which will convert to and back appropriate dlib objects.
                    For example, when using dlib::ffmpeg::decoder or dlib::ffmpeg::demuxer, directly after calling
                    .read(), use convert() to get a dlib object which you can then use for your computer vision,
                    or DNN application.

                    If users need to use frame objects directly, maybe because RGB or BGR aren't appropriate, 
                    and they would rather use the default format returned by their codec, then use
                    frame::get_frame().data and frame::get_frame().linesize to iterate or copy the data.
                    Please carefully read FFMpeg's documentation on how to interpret those fields.
                    Also, users must not copy AVFrame directly. It is a C object, and therefore does not
                    support RAII. If you need to make copies, use the frame object (which wraps AVFrame)
                    which has well defined copy (and move) semantics.
            !*/

            frame() = default;
            /*!
                ensures
                    - is_empty() == true
            !*/

            frame(frame&& ori) = default;
            /*!
                ensures
                    - Move constructor
                    - After move, ori.is_empty() == true
            !*/

            frame& operator=(frame&& ori) = default;
            /*!
                ensures
                    - Move assign operator
                    - After move, ori.is_empty() == true
            !*/

            frame(const frame& ori);
            /*!
                ensures
                    - Copy constructor
            !*/

            frame& operator=(const frame& ori);
            /*!
                ensures
                    - Copy assign operator
            !*/

            frame(
                int                                     h,
                int                                     w,
                AVPixelFormat                           pixfmt,
                std::chrono::system_clock::time_point   timestamp_us
            );
            /*!
                ensures
                    - Create a an image frame object with these parameters.
                    - is_image() == true
                    - is_audio() == false
                    - is_empty() == false
            !*/

            frame(
                int                                     sample_rate,
                int                                     nb_samples,
                uint64_t                                channel_layout,
                AVSampleFormat                          samplefmt,
                std::chrono::system_clock::time_point   timestamp
            );
            /*!
                ensures
                    - Create a an audio frame object with these parameters.
                    - is_image() == false
                    - is_audio() == true
                    - is_empty() == false
            !*/

            bool is_empty() const noexcept;
            /*!
                ensures
                    - Returns true if is_image() == false and is_audio() == false
            !*/

            bool is_image() const noexcept;
            /*!
                ensures
                    - Returns true if underlying AVFrame* != nullptr, height() > 0, width() > 0 and pixfmt() != AV_PIX_FMT_NONE
            !*/

            bool is_audio() const noexcept;
            /*!
                ensures
                    - Returns true if underlying AVFrame* != nullptr, height() > 0, width() > 0 and pixfmt() != AV_PIX_FMT_NONE
            !*/

            AVPixelFormat pixfmt() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is image type, returns pixel format, otherwise, returns AV_PIX_FMT_NONE
            !*/

            int height() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is image type, returns height, otherwise 0
            !*/

            int width() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is image type, returns width, otherwise 0
            !*/

            int nsamples() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is audio type, returns number of samples, otherwise 0
            !*/

            int  nchannels() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is audio type, returns number of channels, e.g. 1 for mono, 2 for stereo, otherwise 0
            !*/

            uint64_t layout() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is audio type, returns channel layout, e.g. AV_CH_LAYOUT_MONO or AV_CH_LAYOUT_STEREO, otherwise 0
            !*/

            AVSampleFormat samplefmt() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is audio type, returns sample format, otherwise, returns AV_SAMPLE_FMT_NONE
            !*/

            int sample_rate() const noexcept;
            /*!
                ensures
                    - If underlying AVFrame* is audio type, returns sample rate, otherwise, returns 0
            !*/

            std::chrono::system_clock::time_point get_timestamp() const noexcept;
            /*!
                ensures
                    - If possible, returns a timestamp associtated with this frame. This is not always possible, it depends on whether the information
                      is provided by the codec and/or the muxer. dlib will do it's best to get a timestamp for you.
            !*/

            const AVFrame& get_frame() const;
            /*!
                requires
                    - is_empty() == false

                ensures
                    - Returns a const reference to the underyling AVFrame object. DO NOT COPY THIS OBJECT! RAII is not supported on this sub-object.
                      Use with care! Prefer to use dlib's convert() functions to convert to and back dlib objects.
            !*/

            AVFrame& get_frame();
            /*!
                requires
                    - is_empty() == false
                ensures
                    - Returns a non-const reference to the underlying AVFrame object. DO NOT COPY THIS OBJECT! RAII is not supported on this sub-object.
                      Use with care! Prefer to use dlib's convert() functions to convert to and back dlib objects.
            !*/

            void set_params(
                int                                     h,
                int                                     w,
                AVPixelFormat                           pixfmt,
                int                                     sample_rate,
                int                                     nb_samples,
                uint64_t                                channel_layout,
                AVSampleFormat                          samplefmt,
                std::chrono::system_clock::time_point   timestamp
            );
            /*!
                requires
                    - For images, set sample_rate = nb_samples = channel_layout = 0, and samplefmt = AV_SAMPLE_FMT_NONE
                    - For audio, set h = w = 0 and pixfmt = AV_PIX_FMT_NONE
                ensures
                    - Resizes the frame to the corresponding dims.
                    - is_empty()        == false
                    - is_image()        == (h > 0 && w > 0 && pixfmt != AV_PIX_FMT_NONE)
                    - is_audio()        == (sample_rate > 0 && nb_samples > 0 && channel_layout > 0 && samplefmt != AV_SAMPLE_FMT_NONE)
                    - height()          == h
                    - width()           == w
                    - pixfmt()          == pixfmt
                    - sample_rate()     == sample_rate
                    - layout()          == channel_layout
                    - samplefmt()       == samplefmt
                    - get_timestamp()   == timestamp
            !*/

            void clear();
            /*!
                ensures
                    - is_empty() == true
            !*/

        private:

            friend class details::resampler;
            friend class encoder;
            friend class decoder;

            void copy_from(const frame& other);

            details::av_ptr<AVFrame> f;
            std::chrono::system_clock::time_point timestamp;
        };

// ---------------------------------------------------------------------------------------------------

        template<class PixelType>
        struct pix_traits
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a type trait for converting a sample type to ffmpeg's AVPixelFormat obj.
            !*/
        };

        template<> struct pix_traits<uint8_t>           { constexpr static AVPixelFormat fmt = AV_PIX_FMT_GRAY8; };
        template<> struct pix_traits<rgb_pixel>         { constexpr static AVPixelFormat fmt = AV_PIX_FMT_RGB24; };
        template<> struct pix_traits<bgr_pixel>         { constexpr static AVPixelFormat fmt = AV_PIX_FMT_BGR24; };
        template<> struct pix_traits<rgb_alpha_pixel>   { constexpr static AVPixelFormat fmt = AV_PIX_FMT_RGBA;  };
        template<> struct pix_traits<bgr_alpha_pixel>   { constexpr static AVPixelFormat fmt = AV_PIX_FMT_BGRA;  };

// ---------------------------------------------------------------------------------------------------

        template<class SampleType>
        struct sample_traits 
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a type trait for converting a sample type to ffmpeg's AVSampleFormat obj.
            !*/
        };

        template<> struct sample_traits<uint8_t> { constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_U8; };
        template<> struct sample_traits<int16_t> { constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_S16; };
        template<> struct sample_traits<int32_t> { constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_S32; };
        template<> struct sample_traits<float>   { constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_FLT; };
        template<> struct sample_traits<double>  { constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_DBL; };

// ---------------------------------------------------------------------------------------------------

        template<class SampleType, std::size_t Channels>
        struct audio
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a typed audio buffer which can convert to and back dlib::ffmpeg::frame.
            !*/

            using sample = std::array<SampleType, Channels>;
            using format = SampleType;
            constexpr static std::size_t nchannels = Channels;

            std::vector<sample>                     samples;
            float                                   sample_rate{0};
            std::chrono::system_clock::time_point   timestamp{};
        };

// ---------------------------------------------------------------------------------------------------

        struct codec_details
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object informs on available codecs provided by the installation of ffmpeg dlib is linked against.
            !*/

            AVCodecID   codec_id{AV_CODEC_ID_NONE};
            std::string codec_name;
            bool supports_encoding{false};
            bool supports_decoding{false};
        };

        struct muxer_details
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object informs on available muxers provided by the installation of ffmpeg dlib is linked against.
            !*/

            std::string name;
            std::vector<codec_details> supported_codecs;
        };

        struct device_details
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object informs on available device types provided by the installation of ffmpeg dlib is linked against.
            !*/

            std::string device_type;
            bool        is_audio_type{false};
            bool        is_video_type{false};
        };

        struct device_instance
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object informs on the currently available device instances readable by ffmpeg.
            !*/

            std::string name;
            std::string description;
        };

        const std::vector<std::string>& list_protocols();
        /*!
            ensures
                - returns a list of all available ffmpeg protocols
        !*/

        const std::vector<std::string>& list_demuxers();
        /*!
            ensures
                - returns a list of all available ffmpeg demuxers
        !*/

        const std::vector<muxer_details>& list_muxers();
        /*!
            ensures
                - returns a list of all available ffmpeg muxers
        !*/
        
        const std::vector<codec_details>& list_codecs();
        /*!
            ensures
                - returns a list of all available ffmpeg codecs with information on whether decoding and/or encoding is supported.
                  Note that not all codecs support encoding, unless your installation of ffmpeg is built with third party library
                  dependencies like libx264, libx265, etc.
        !*/

        const std::vector<device_details>& list_input_device_types();
        /*!
            ensures
                - returns a list of all available ffmpeg input device types (e.g. alsa, v4l2, etc)
        !*/

        const std::vector<device_details>& list_output_device_types();
        /*!
            ensures
                - returns a list of all available ffmpeg output device types (e.g. alsa, v4l2, etc)
        !*/

        std::vector<device_instance> list_input_device_instances(const std::string& device_type);
        /*!
            ensures
                - returns a list of all available ffmpeg input device instances for device type *device_type (e.g. hw:0,0, /dev/video0, etc)
        !*/

        std::vector<device_instance> list_output_device_instances(const std::string& device_type);
        /*!
            ensures
                - returns a list of all available ffmpeg output device instances for device type *device_type (e.g. hw:0,0, /dev/video0, etc)
        !*/

// ---------------------------------------------------------------------------------------------------
    
        struct video_enabled_t
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a strong type which controls whether or not we want
                    to enable video decoding in demuxer or video encoding in muxer.

                    For example, you can now use the convenience constructor:

                        demuxer cap(filename, video_enabled, audio_disabled);
            !*/

            constexpr explicit video_enabled_t(bool enabled_) : enabled{enabled_} {}
            bool enabled{false};
        };

        constexpr video_enabled_t video_enabled{true};
        constexpr video_enabled_t video_disabled{false};

        struct audio_enabled_t
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a strong type which controls whether or not we want
                    to enable audio decoding in demuxer or audio encoding in muxer
            !*/

            constexpr explicit audio_enabled_t(bool enabled_) : enabled{enabled_} {}
            bool enabled{false};
        };

        constexpr audio_enabled_t audio_enabled{true};
        constexpr audio_enabled_t audio_disabled{false};

// ---------------------------------------------------------------------------------------------------

        template <
          class image_type, 
          is_image_check<image_type> = true
        >
        void convert(const frame& f, image_type& image);
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - f.is_image() == true
                - f.pixfmt() == pix_traits<pixel_type_t<image_type>>::fmt
            ensures
                - converts a frame object into array2d<rgb_pixel>
        !*/

        template <
          class image_type, 
          is_image_check<image_type> = true
        >
        void convert(const image_type& img, frame& f);
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h
            ensures
                - converts a dlib image into a frame object
        !*/

        template<class SampleFmt, std::size_t Channels>
        void convert(const frame& f, audio<SampleFmt, Channels>& obj);
        /*!
            requires
                - f.is_audio()  == true
                - f.samplefmt() == sample_traits<SampleFmt>::fmt
                - f.nchannels() == Channels
            ensures
                - converts a frame object into audio object
        !*/

        template<class SampleFmt, std::size_t Channels>
        void convert(const audio<SampleFmt, Channels>& audio, frame& b);
        /*!
            ensures
                - converts a dlib audio object into a frame object
        !*/

// ---------------------------------------------------------------------------------------------------

        struct resizing_args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments used for resizing an image frame.
                    When all arguments are zero or defaulted, then no resizing is undertaken,
                    and the src frame is memcpy'd to the dst frame.

                    Note:
                        - "fmt" can take any value starting with AV_PIX_FMT_
                          See libavutil/pixfmt.h for options.
            !*/

            int             h{0};
            int             w{0};
            AVPixelFormat   fmt{AV_PIX_FMT_NONE};
        };

// ---------------------------------------------------------------------------------------------------

        struct resampling_args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments used for resampling an audio frame.
                    When all arguments are zero or defaulted, then no resampling is undertaken,
                    and the src frame is memcpy'd to the dst frame.

                    Note:
                        - "channel_layout" can take values starting with AV_CH_LAYOUT_ 
                          See libavutil/channel_layout.h for options.
                        - "fmt" can take any value starting with AV_SAMPLE_FMT_
                          See libavutil/samplefmt.h for options.
            !*/

            int             sample_rate{0};
            uint64_t        channel_layout{0};
            AVSampleFormat  fmt{AV_SAMPLE_FMT_NONE};
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

// ---------------------------------------------------------------------------------------------------

        namespace details
        {
            template<class... Args>
            inline bool fail(Args&&... args)
            {
                auto ret = logger_dlib_wrapper() << LERROR;
#ifdef __cpp_fold_expressions
                ((ret << args),...);
#else
                (void)std::initializer_list<int>{((ret << args), 0)...};
#endif
                return false;
            }
        }

// ---------------------------------------------------------------------------------------------------

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

        inline std::string get_channel_layout_str(uint64_t channel_layout)
        {
            return details::get_channel_layout_str(channel_layout);
        }    

// ---------------------------------------------------------------------------------------------------

        inline dlib::logger& logger_ffmpeg()
        {
            details::register_ffmpeg();
            return details::logger_ffmpeg_private();
        }

        inline dlib::logger& logger_dlib_wrapper()
        {
            static dlib::logger GLOBAL("ffmpeg.dlib");
            return GLOBAL;
        }

// ---------------------------------------------------------------------------------------------------

        namespace details
        {
        
// ---------------------------------------------------------------------------------------------------

            class resizer
            {
            public:
                void resize(
                    const frame &src,
                    const int dst_h, const int dst_w, const AVPixelFormat dst_fmt,
                    frame &dst
                );

                void resize(
                    const frame &src,
                    frame &dst
                );

            private:
                av_ptr<SwsContext> imgConvertCtx;
            };

            inline void resizer::resize (
                const frame& src,
                const int dst_h, const int dst_w, const AVPixelFormat dst_fmt,
                frame& dst
            )
            {
                DLIB_CASSERT(src.is_image(), "src.is_image() == false");

                imgConvertCtx.reset(sws_getCachedContext(imgConvertCtx.release(),
                                                         src.width(), src.height(), src.pixfmt(),
                                                         dst_w,       dst_h,        dst_fmt,
                                                         SWS_FAST_BILINEAR, NULL, NULL, NULL));

                const bool is_same_object = &src == &dst;
                const bool is_copy = std::make_tuple(src.height(), src.width(), src.pixfmt()) == std::tie(dst_h, dst_w, dst_fmt);

                if (is_same_object && is_copy)
                    return;

                frame* ptr = &dst;
                frame tmp;

                if (is_same_object)
                    ptr = &tmp;
                
                ptr->set_params(dst_h, dst_w, dst_fmt, 0, 0, 0, AV_SAMPLE_FMT_NONE, src.get_timestamp());

                sws_scale(imgConvertCtx.get(),
                          src.get_frame().data,  src.get_frame().linesize, 0, src.height(),
                          ptr->get_frame().data, ptr->get_frame().linesize);

                if (is_same_object)
                    dst = std::move(tmp);
            }

            inline void resizer::resize(
                const frame& src,
                frame& dst
            )
            {
                resize(src, src.height(), src.width(), src.pixfmt(), dst);
            }

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

            inline void resampler::reset(
                const int src_sample_rate_, const uint64_t src_channel_layout_, const AVSampleFormat src_fmt_,
                const int dst_sample_rate_, const uint64_t dst_channel_layout_, const AVSampleFormat dst_fmt_
            )
            {
                using namespace details;

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
#if LIBSWRESAMPLE_VERSION_INT >= AV_VERSION_INT(4, 5, 100) 
                        AVChannelLayout layout_src = convert_layout(src_channel_layout);
                        AVChannelLayout layout_dst = convert_layout(dst_channel_layout);
                        
                        SwrContext* ptr{nullptr};
                        const int ret = swr_alloc_set_opts2(&ptr,
                            &layout_dst, dst_fmt, dst_sample_rate,
                            &layout_src, src_fmt, src_sample_rate,
                            0, nullptr
                        );
                        DLIB_CASSERT(ret == 0, "swr_alloc_set_opts2() failed : " << get_av_error(ret));
                        audioResamplerCtx.reset(ptr);
#else
                        audioResamplerCtx.reset(swr_alloc_set_opts(NULL,
                                                                   dst_channel_layout, dst_fmt_, dst_sample_rate_,
                                                                   src_channel_layout, src_fmt_, src_sample_rate_,
                                                                   0, NULL));
                        const int ret =  swr_init(audioResamplerCtx.get());
                        DLIB_CASSERT(ret == 0, "swr_init() failed : " << get_av_error(ret));
#endif
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
                using namespace details;
                using std::chrono::system_clock;

                DLIB_CASSERT(src.is_audio(), "src.is_audio() == false");

                const bool is_same_object = std::addressof(src) == std::addressof(dst);

                reset(src.sample_rate(),         src.layout(), src.samplefmt(),
                      dst_sample_rate_,  dst_channel_layout_,  dst_samplefmt_);

                if (audioResamplerCtx)
                {
                    av_ptr<AVFrame> tmp = make_avframe();
                    tmp->sample_rate    = dst_sample_rate;
                    tmp->format         = (int)dst_fmt;
                    set_layout(tmp.get(), dst_channel_layout);

                    const int ret = swr_convert_frame(audioResamplerCtx.get(), tmp.get(), &src.get_frame());
                    if (ret < 0)
                        throw std::runtime_error("swr_convert_frame() failed : " + get_av_error(ret));

                    dst.f           = std::move(tmp);
                    dst.f->pts      = tracked_samples;
                    dst.timestamp   = system_clock::time_point{system_clock::duration{av_rescale_q(tracked_samples,
                                                                                        {1, dst_sample_rate},
                                                                                        {system_clock::duration::period::num, system_clock::duration::period::den})}};
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
                using std::chrono::system_clock;
                DLIB_ASSERT(in.is_audio(), "this isn't an audio frame");

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
                        const system_clock::time_point timestamp{system_clock::duration{av_rescale_q(
                                sample_count,
                                {1, in.sample_rate()},
                                {system_clock::duration::period::num, system_clock::duration::period::den})}};

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
        inline void frame::clear()
        {
            f = nullptr;
            timestamp = std::chrono::system_clock::time_point{};
        }

        inline void frame::set_params(
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

            const int format = (h > 0 && w > 0) ? (int)pixfmt : (int)samplefmt;

            if (!f ||
                std::tie(f->height, f->width, f->sample_rate, f->nb_samples, f->format) != 
                std::tie(   h,         w,        sample_rate,    nb_samples,    format) ||
                get_layout(f.get()) != channel_layout)
            {
                f = make_avframe();
                f->height           = h;
                f->width            = w;
                f->sample_rate      = sample_rate;
                f->nb_samples       = nb_samples;
                f->format           = h > 0 && w > 0 ? (int)pixfmt : (int)samplefmt;
                set_layout(f.get(), channel_layout);

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
            }

            timestamp = timestamp_;
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
        )
        {
            set_params(h, w, fmt, 0, 0, 0, AV_SAMPLE_FMT_NONE, timestamp);
        }

        inline frame::frame(
            int             sample_rate,
            int             nb_samples,
            uint64_t        channel_layout,
            AVSampleFormat  fmt,
            std::chrono::system_clock::time_point timestamp
        )
        {
            set_params(0,0,AV_PIX_FMT_NONE, sample_rate, nb_samples, channel_layout, fmt, timestamp);
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
            using namespace details;

            if (ori.is_empty())
            {
                clear();
            }
            else
            {
                if (is_empty() ||
                    std::tie(    f->height,     f->width,     f->format,     f->sample_rate,     f->nb_samples) !=
                    std::tie(ori.f->height, ori.f->width, ori.f->format, ori.f->sample_rate, ori.f->nb_samples) ||
                    get_layout(f.get()) != get_layout(ori.f.get()))
                {
                    set_params(ori.f->height,
                               ori.f->width,
                               (AVPixelFormat)ori.f->format,
                               ori.f->sample_rate,
                               ori.f->nb_samples,
                               ori.layout(),
                               (AVSampleFormat)ori.f->format,
                               ori.timestamp);
                }

                av_frame_copy(f.get(), ori.f.get());
                av_frame_copy_props(f.get(), ori.f.get());
                // The following silences a warning message about too many b-frames.
                f->pict_type = AV_PICTURE_TYPE_NONE;
            }
        }

        inline bool frame::is_image() const noexcept
        {
            return f && f->width > 0 && f->height > 0 && f->format != AV_PIX_FMT_NONE;
        }

        inline bool frame::is_audio() const noexcept
        {
            return f && f->nb_samples > 0 && f->sample_rate > 0 && f->format != AV_SAMPLE_FMT_NONE && !details::channel_layout_empty(f.get());
        }

        inline bool                 frame::is_empty()   const noexcept { return !is_image() && !is_audio(); }
        inline AVPixelFormat        frame::pixfmt()     const noexcept { return is_image() ? (AVPixelFormat)f->format : AV_PIX_FMT_NONE; }
        inline int                  frame::height()     const noexcept { return is_image() ? f->height : 0; }
        inline int                  frame::width()      const noexcept { return is_image() ? f->width : 0; }
        inline AVSampleFormat       frame::samplefmt()  const noexcept { return is_audio() ? (AVSampleFormat)f->format : AV_SAMPLE_FMT_NONE; }
        inline int                  frame::nsamples()   const noexcept { return is_audio() ? f->nb_samples : 0; }
        inline int                  frame::sample_rate() const noexcept{ return is_audio() ? f->sample_rate : 0; }
        inline uint64_t             frame::layout()     const noexcept { return is_audio() ? details::get_layout(f.get()) : 0; }
        inline int                  frame::nchannels()  const noexcept { return is_audio() ? details::get_nchannels(f.get()) : 0; } 
        inline const AVFrame&       frame::get_frame()  const { DLIB_CASSERT(f, "is_empty() == true"); return *f; }
        inline AVFrame&             frame::get_frame()        { DLIB_CASSERT(f, "is_empty() == true"); return *f; }
        inline std::chrono::system_clock::time_point frame::get_timestamp() const noexcept { return timestamp; }

// ---------------------------------------------------------------------------------------------------

        inline const std::vector<std::string>& list_protocols()
        {
            const static auto protocols = []
            {
                details::register_ffmpeg();
                std::vector<std::string> protocols;
                void* opaque = nullptr;
                const char* name = 0;
                while ((name = avio_enum_protocols(&opaque, 0)))
                    protocols.emplace_back(name);

                opaque  = nullptr;
                name    = 0;

                while ((name = avio_enum_protocols(&opaque, 1)))
                    protocols.emplace_back(name);

                return protocols;
            }();

            return protocols;
        }

// ---------------------------------------------------------------------------------------------------

        inline const std::vector<std::string>& list_demuxers()
        {
            const static auto demuxers = []
            {
                details::register_ffmpeg();
                std::vector<std::string> demuxers;
                const AVInputFormat* demuxer = nullptr;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
                // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L86
                while ((demuxer = av_iformat_next(demuxer)))
#else
                void* opaque = nullptr;
                while ((demuxer = av_demuxer_iterate(&opaque)))
#endif
                    demuxers.push_back(demuxer->name);

                return demuxers;
            }();

            return demuxers;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<codec_details> list_codecs_for_muxer (
            const AVOutputFormat* oformat
        )
        {
            std::vector<codec_details> supported_codecs;

            for (const auto& codec : list_codecs())
                if (avformat_query_codec(oformat, codec.codec_id, FF_COMPLIANCE_STRICT) == 1)
                    supported_codecs.push_back(codec);
            
            return supported_codecs;
        }

// ---------------------------------------------------------------------------------------------------

        inline const std::vector<muxer_details>& list_muxers()
        {
            const static auto ret = []
            {
                details::register_ffmpeg();

                std::vector<muxer_details> all_details;
                const AVOutputFormat* muxer = nullptr;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
                // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L86
                while ((muxer = av_oformat_next(muxer)))
#else
                void* opaque = nullptr;
                while ((muxer = av_muxer_iterate(&opaque)))
#endif
                {
                    muxer_details details;
                    details.name                = muxer->name;
                    details.supported_codecs    = list_codecs_for_muxer(muxer);
                    all_details.push_back(details);
                }      
            
                return all_details;
            }();

            return ret;
        }

// ---------------------------------------------------------------------------------------------------

        inline const std::vector<codec_details>& list_codecs()
        {
            const static auto ret = []
            {
                details::register_ffmpeg();
                std::vector<codec_details> details;

        #if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 10, 100)
                // See https://github.com/FFmpeg/FFmpeg/blob/70d25268c21cbee5f08304da95be1f647c630c15/doc/APIchanges#L91
                AVCodec* codec = nullptr;
                while ((codec = av_codec_next(codec)))
        #else
                const AVCodec* codec = nullptr;
                void* opaque = nullptr;
                while ((codec = av_codec_iterate(&opaque)))
        #endif
                {
                    codec_details detail;
                    detail.codec_id     = codec->id;
                    detail.codec_name   = codec->name;
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
            }();

            return ret;
        }

// ---------------------------------------------------------------------------------------------------

        inline const std::vector<device_details>& list_input_device_types()
        {
            const static auto ret = []
            {
                details::register_ffmpeg();
                std::vector<device_details> devices;

#if LIBAVDEVICE_VERSION_INT < AV_VERSION_INT(59, 0, 100)
                using AVInputFormatPtr = AVInputFormat*;
#else
                using AVInputFormatPtr = const AVInputFormat*;
#endif

                AVInputFormatPtr device{nullptr};

                while ((device = av_input_audio_device_next(device)))
                {
                    device_details details;
                    details.device_type     = device->name;
                    details.is_audio_type   = true;
                    devices.push_back(std::move(details));
                }

                device = nullptr;

                while ((device = av_input_video_device_next(device)))
                {
                    device_details details;
                    details.device_type     = device->name;
                    details.is_video_type   = true;
                    devices.push_back(std::move(details));
                }

                return devices;
            }();

            return ret;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<device_instance> list_input_device_instances(const std::string& device_type)
        {
            const auto& types = list_input_device_types();
            auto ret = std::find_if(types.begin(), types.end(), [&](const auto& type) {return type.device_type == device_type;});
            if (ret == types.end())
                return {};

            std::vector<device_instance> instances;

            details::av_ptr<AVDeviceInfoList> managed;
            AVDeviceInfoList* device_list = nullptr;
            avdevice_list_input_sources(nullptr, ret->device_type.c_str(), nullptr, &device_list);
            managed.reset(device_list);

            if (device_list)
            {
                for (int i = 0 ; i < device_list->nb_devices ; ++i)
                {
                    device_instance instance;
                    instance.name        = std::string(device_list->devices[i]->device_name);
                    instance.description = std::string(device_list->devices[i]->device_description);
                    instances.push_back(std::move(instance));
                }
            }

            return instances;
        }

// ---------------------------------------------------------------------------------------------------

        inline const std::vector<device_details>& list_output_device_types()
        {
            const static auto ret = []
            {
                details::register_ffmpeg();
                std::vector<device_details> devices;

    #if LIBAVDEVICE_VERSION_INT < AV_VERSION_INT(59, 0, 100)
                using AVOutputFormatPtr = AVOutputFormat*;
    #else
                using AVOutputFormatPtr = const AVOutputFormat*;
    #endif

                AVOutputFormatPtr device{nullptr};

                while ((device = av_output_audio_device_next(device)))
                {
                    device_details details;
                    details.device_type     = std::string(device->name);
                    details.is_audio_type   = true;
                    devices.push_back(std::move(details));
                }

                device = nullptr;

                while ((device = av_output_video_device_next(device)))
                {
                    device_details details;
                    details.device_type     = std::string(device->name);
                    details.is_video_type   = true;
                    devices.push_back(std::move(details));
                }

                return devices;
            }();

            return ret;
        }

// ---------------------------------------------------------------------------------------------------

        inline std::vector<device_instance> list_output_device_instances(const std::string& device_type)
        {
            const auto& types = list_output_device_types();
            auto ret = std::find_if(types.begin(), types.end(), [&](const auto& type) {return type.device_type == device_type;});
            if (ret == types.end())
                return {};

            std::vector<device_instance> instances;

            details::av_ptr<AVDeviceInfoList> managed;
            AVDeviceInfoList* device_list = nullptr;
            avdevice_list_output_sinks(nullptr, ret->device_type.c_str(), nullptr, &device_list);
            managed.reset(device_list);

            if (device_list)
            {
                for (int i = 0 ; i < device_list->nb_devices ; ++i)
                {
                    device_instance instance;
                    instance.name        = std::string(device_list->devices[i]->device_name);
                    instance.description = std::string(device_list->devices[i]->device_description);
                    instances.push_back(std::move(instance));
                }
            }

            return instances;
        }

// ---------------------------------------------------------------------------------------------------

        template <
          class image_type, 
          is_image_check<image_type>
        >
        inline void convert(const frame& f, image_type& image)
        {
            using pixel = pixel_type_t<image_type>;
            
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
            
            DLIB_ASSERT(ret == (int)expsize, "av_image_copy_to_buffer() error : " << details::get_av_error(ret));
            (void)ret;
        }

// ---------------------------------------------------------------------------------------------------

        template<
          class image_type, 
          is_image_check<image_type>
        >
        inline void convert(const image_type& img, frame& f)
        {
            using pixel = pixel_type_t<image_type>;

            if (f.height() != img.nr() ||
                f.width()  != img.nc() ||
                f.pixfmt() != pix_traits<pixel>::fmt)
            {
                f.set_params(img.nr(), img.nc(), pix_traits<pixel>::fmt, 0, 0, 0, AV_SAMPLE_FMT_NONE, {});
            }

            const size_t imgsize            = img.nr()*img.nc()*sizeof(pixel);
            int         src_linesizes[4]    = {0};
            uint8_t*    src_pointers[4]     = {nullptr};

            const int ret = av_image_fill_arrays(src_pointers, src_linesizes, (uint8_t*)img.begin(), f.pixfmt(), f.width(), f.height(), 1);
            DLIB_ASSERT(ret == imgsize, "av_image_fill_arrays()  error : " << details::get_av_error(ret));
            (void)imgsize;
            (void)ret;
            
            av_image_copy(f.get_frame().data,
                          f.get_frame().linesize,
                          (const uint8_t**)src_pointers,
                          src_linesizes,
                          f.pixfmt(),
                          f.width(),
                          f.height());
        }

// ---------------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------------

        template<class SampleFmt, std::size_t Channels>
        inline void convert(const audio<SampleFmt, Channels>& obj, frame& f)
        {
            using namespace details;
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
