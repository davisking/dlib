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
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments passed to the encoder and muxer classes.
                    These must be set to non-zero or non-trivial values as they are used to configure 
                    the underlying codec and optionally, an internal image scaler.
                    Any frame that is pushed to encoder or muxer instances is resized to the codec's 
                    pre-configured settings if their dimensions or pixel format don't match.
                    For example, if the codec is configured to use height 512, width 384 and RGB format,
                    using the variables below, and the frames already have these settings when pushed, 
                    then no resizing is performed. If however they don't, then they are first resized. 
            !*/

            // Target height of codec.
            int h{0};

            // Target width of codec.
            int w{0};
            
            // Target pixel format of codec.
            AVPixelFormat fmt{AV_PIX_FMT_YUV420P};

            // Target framerate of codec/muxer
            int framerate{0};
        };

// ---------------------------------------------------------------------------------------------------

        struct encoder_audio_args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments passed to the encoder and muxer classes.
                    These must be set to non-zero or non-trivial values as they are used to configure 
                    the underlying codec and optionally, an internal audio resampler.
                    Any frame that is pushed to encoder or muxer instances is resampled to the codec's
                    pre-configured settings if their sample format, sample rate or channel layout, don't match.
            !*/

            // Target sample rate of codec
            int sample_rate{0};

            // Target channel layout of codec
            uint64_t channel_layout{AV_CH_LAYOUT_STEREO};

            // Target sample format of codec
            AVSampleFormat fmt{AV_SAMPLE_FMT_S16};
        };

// ---------------------------------------------------------------------------------------------------

        struct encoder_codec_args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments passed to the encoder and muxer classes.
                    Some of these must be set to non-zero or non-trivial values as they are used 
                    to configure the underlying codec. Others will only be used if non-zero or
                    non-trivial.
            !*/

            // Codec ID used to configure the encoder. Either codec or codec_name MUST be set.
            AVCodecID codec{AV_CODEC_ID_NONE};

            // Codec name used to configure the encoder. This is used if codec == AV_CODEC_ID_NONE.
            std::string codec_name;

            // A dictionary of AVCodecContext and codec-private options. Used by "avcodec_open2()"
            std::unordered_map<std::string, std::string> codec_options;

            // Sets AVCodecContext::bit_rate if non-negative.
            int64_t bitrate{-1};

            // Sets AVCodecContext::gop_size if non-negative.
            int gop_size{-1};

            // OR-ed with AVCodecContext::flags if non-negative.
            int flags{0};
        };

// ---------------------------------------------------------------------------------------------------

        class encoder
        {
        public:
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class is a libavcodec wrapper which encodes video or audio to raw memory.
                    Note, if you are creating a media file, it is easier to use the muxer object
                    as it also works with raw codec files like .h264 files.
                    This class is suitable for example if you need to send raw packets over a socket
                    or interface with another library that requires encoded data, not raw images
                    or raw audio samples.
            !*/

            struct args
            {
                /*!
                    WHAT THIS OBJECT REPRESENTS
                        This holds constructor arguments for encoder.
                !*/
                encoder_codec_args args_codec;
                encoder_image_args args_image;
                encoder_audio_args args_audio;
            };

            encoder() = default;
            /*!
                ensures
                    - is_open() == false
            !*/

            encoder(
                const args& a,
                std::function<bool(std::size_t, const char*)> sink
            );
            /*!
                requires
                    - a.args_codec.codec or a.args_codec.codec_name are set
                    - Either a.args_image or a.args_audio is fully set
                    - sink is set to a valid callback for writing packet data.
                      dlib/media/sink.h contains callback wrappers for
                      different buffer types.
                ensures
                    - Constructs encoder from args and sink
                    - is_open() == true
            !*/

            encoder(encoder&& other) = default;
            /*!
                ensures
                    - Move constructor
                    - other is in an empty but otherwise valid state after move
                    - other.is_open() == false after move
            !*/

            encoder& operator=(encoder&& other) = default;
            /*!
                ensures
                    - Move assignment operator
                    - other is in an empty but otherwise valid state after move
                    - other.is_open() == false after move
            !*/
            
            ~encoder();
            /*!
                ensures
                    - Destructor
                    - flush() is called if it hasn't been already
            !*/

            bool is_open() const noexcept;
            /*!
                ensures
                    - Returns true if the codec is open and user may call push()
            !*/

            bool is_image_encoder() const noexcept;
            /*!
                ensures
                    - Returns true if the codec is an image encoder.
            !*/

            bool is_audio_encoder() const noexcept;
            /*!
                ensures
                    - Returns true if the codec is an audio encoder.
            !*/

            AVCodecID get_codec_id() const noexcept;
            /*!
                requires
                    - is_open() == true
                ensures
                    - returns the codec id. See ffmpeg documentation or libavcodec/codec_id.h
            !*/

            std::string get_codec_name() const noexcept;
            /*!
                requires
                    - is_open() == true
                ensures
                    - returns string representation of codec id.
            !*/

            int height() const noexcept;
            /*!
                requires
                    - is_image_encoder() == true
                ensures
                    - returns the height of the configured codec, not necessarily the
                      height of frames passed to push(frame)
            !*/

            int width() const noexcept;
            /*!
                requires
                    - is_image_encoder() == true
                ensures
                    - returns the width of the configured codec, not necessarily the
                      width of frames passed to push(frame)
            !*/

            AVPixelFormat pixel_fmt() const noexcept;
            /*!
                requires
                    - is_image_encoder() == true
                ensures
                    - returns the pixel format of the configured codec, not necessarily the
                      pixel format of frames passed to push(frame)
            !*/

            int fps() const noexcept;
             /*!
                requires
                    - is_image_encoder() == true
                ensures
                    - returns the configured framerate of the codec.
            !*/

            int sample_rate() const noexcept;
            /*!
                requires
                    - is_audio_encoder() == true
                ensures
                    - returns the sample rate of the configured codec, not necessarily the
                      sample rate of frames passed to push(frame)
            !*/

            uint64_t channel_layout() const noexcept;
            /*!
                requires
                    - is_audio_encoder() == true
                ensures
                    - returns the channel layout of the configured codec, not necessarily the
                      channel layout of frames passed to push(frame).
                      e.g. AV_CH_LAYOUT_STEREO, AV_CH_LAYOUT_MONO etc.
            !*/

            AVSampleFormat sample_fmt() const noexcept;
            /*!
                requires
                    - is_audio_encoder() == true
                ensures
                    - returns the sample format of the configured codec, not necessarily the
                      sample format of frames passed to push(frame)
            !*/

            int nchannels() const noexcept;
            /*!
                requires
                    - is_audio_encoder() == true
                ensures
                    - returns the number of audio channels in the configured codec.
            !*/

            bool push(frame f);
            /*!
                requires
                    - if is_image_encoder() == true, then f.is_image() == true
                    - if is_audio_encoder() == true, then f.is_audio() == true
                ensures
                    - If f does not have matching settings to the codec, it is either
                      resized or resampled before being pushed to the codec and encoded.
                    - The callback passed to the constructor may or may not be invoked
                      as the underlying resampler, audio fifo and codec may buffer.
                    - Returns true if successfully encoded, even if callback wasn't invoked.
                    - Returns false if either EOF, i.e. flush() has been previously called,
                      or an error occurred, in which case is_open() == false.
            !*/

            void flush();
            /*!
                ensures
                    - Flushes the codec. Callback passed to constructor will likely be invoked.
                    - is_open() == false
                    - Becomes a no-op after the first time you call this.
            !*/

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

        class muxer
        {
        public:
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class is a libavformat wrapper which muxes video and/or audio streams to file.
                    It is analogous to OpenCV's cv::VideoWriter class but is more configurable, supports audio, 
                    devices (X11, speaker, ...) and network streams such as RTSP, HTTP, and more.
                    Note that a video file, e.g. MP4, can contain multiple streams: video, audio, subtitles, etc.
                    This class can encode both video and audio at the same time.
            !*/

            struct args
            {
                /*!
                    WHAT THIS OBJECT REPRESENTS
                        This holds constructor arguments for muxer.
                !*/

                args() = default;
                /*!
                    ensures
                        - Default constructor.
                !*/

                args(const std::string& filepath);
                /*!
                    ensures
                        - this->filepath = filepath
                !*/

                // Filepath, URL or device this object will write data to.
                std::string filepath;

                // Output format hint. e.g. 'rtsp', 'X11', 'alsa', etc. 99% of the time, users are not required to specify this as libavformat tries to guess it.
                std::string output_format;

                // A dictionary filled with AVFormatContext and muxer-private options. Used by "avformat_write_header()"".
                // Please see libavformat documentation for more details
                std::unordered_map<std::string, std::string> format_options;

                // An AVDictionary filled with protocol-private options. Used by avio_open2()
                std::unordered_map<std::string, std::string> protocol_options;

                // See documentation for AVFormatContext::max_delay
                int max_delay{-1};

                // Only relevant to network muxers such as RTSP, HTTP etc.
                // Connection/listening timeout. 
                std::chrono::milliseconds connect_timeout{std::chrono::milliseconds::max()};
                
                // Only relevant to network muxers such as RTSP, HTTP etc
                // Read timeout. 
                std::chrono::milliseconds read_timeout{std::chrono::milliseconds::max()};

                // Only relevant to network muxers such as RTSP, HTTP etc
                // Connection/listening interruption callback.
                // The constructor periodically calls interrupter() while waiting on the network. If it returns true, then
                // the connection is aborted and the muxer is closed.
                // So user may use this in conjunction with some thread-safe shared state to signal an abort/interrupt.
                std::function<bool()> interrupter;

                // Video stream arguments
                struct : encoder_codec_args, encoder_image_args{} args_image;

                // Audio stream arguments
                struct : encoder_codec_args, encoder_audio_args{} args_audio;
                
                // Whether or not to encode video stream.
                bool enable_image{true};

                // Whether or not to encode audio stream.
                bool enable_audio{true};
            };

            muxer() = default;
            /*!
                ensures
                    - is_open() == false
            !*/

            muxer(const args& a);
            /*!
                ensures
                    - Constructs muxer using args
            !*/

            muxer(muxer&& other) noexcept;
            /*!
                ensures
                    - Move constructor
                    - After move, other.is_open() == false
            !*/

            muxer& operator=(muxer&& other) noexcept;
            /*!
                ensures
                    - Move assignment
                    - After move, other.is_open() == false
            !*/

            ~muxer();
            /*!
                ensures
                    - Calls flush() if not already called
                    - Closes the underlying file/socket
            !*/

            bool is_open() const noexcept;
            /*!
                ensures
                    - Returns true if underlying container and codecs are open
                    - User may call push()
            !*/

            bool audio_enabled() const noexcept;
            /*!
                requires
                    - args.enable_audio == true
                ensures
                    - returns true if is_open() == true and an audio stream is available and open
            !*/

            bool video_enabled() const noexcept;
            /*!
                requires
                    - args.enable_video == true
                ensures
                    - returns true if is_open() == true and a video stream is available and open
            !*/

            int height() const noexcept;
            /*!
                ensures
                    - if (video_enabled())
                        - returns the height of the configured codec, not necessarily the height of
                          frames passed to push(frame).
                    - else
                        - returns 0
            !*/

            int width() const noexcept;
            /*!
                ensures
                    - if (video_enabled())
                        - returns the width of the configured codec, not necessarily the
                          width of frames passed to push(frame).
                    - else
                        - returns 0
            !*/

            AVPixelFormat pixel_fmt() const noexcept;
            /*!
                ensures
                    - if (video_enabled())
                        - returns the pixel format of the configured codec, not necessarily the
                          pixel format of frames passed to push(frame).
                    - else
                        - returns AV_PIX_FMT_NONE
            !*/

            float fps() const noexcept;
            /*!
                ensures
                    - if (video_enabled())
                        - returns the framerate of the configured codec
                    - else
                        - returns 0
            !*/

            int estimated_nframes() const noexcept;
            /*!
                ensures
                    - returns ffmpeg's estimation of how many video frames are in the file without reading any frames
                    - See ffmpeg's documentation for AVStream::nb_frames
                    - Note, this is known to not always work with ffmpeg v3 with certain files. Most of the time it does
                      Do not make your code rely on this
            !*/

            AVCodecID get_video_codec_id() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns codec ID of video stream
                    - else
                        - returns AV_CODEC_ID_NONE
            !*/

            std::string get_video_codec_name() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns codec name of video stream
                    - else
                        - returns ""
            !*/

            int sample_rate() const noexcept;
            /*!
                ensures
                    - if (audio_enabled())
                        - returns the sample rate of the configured codec, not necessarily the
                          sample rate of frames passed to push(frame).
                    - else
                        - returns 0
            !*/

            uint64_t channel_layout() const noexcept;
            /*!
                ensures
                    - if (audio_enabled())
                        - returns the channel layout of the configured codec, not necessarily the
                          channel layout of frames passed to push(frame).
                    - else
                        - returns 0
            !*/

            int nchannels() const noexcept;
            /*!
                ensures
                    - if (audio_enabled())
                        - returns the number of audio channels in the configured codec, not necessarily the
                          number of channels in frames passed to push(frame).
                    - else
                        - returns 0
            !*/

            AVSampleFormat sample_fmt() const noexcept;
            /*!
                ensures
                    - if (audio_enabled())
                        - returns the sample format of the configured codec, not necessarily the
                          sample format of frames passed to push(frame).
                    - else
                        - returns AV_SAMPLE_FMT_NONE
            !*/

            int estimated_total_samples() const noexcept;
            /*!
                ensures
                    - if (audio_enabled())
                        - returns an estimation fo the total number of audio samples in the audio stream
                    - else
                        - returns 0
            !*/

            AVCodecID get_audio_codec_id() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns codec ID of audio stream
                    - else
                        - returns AV_CODEC_ID_NONE
            !*/

            std::string get_audio_codec_name() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns codec name of audio stream
                    - else
                        - returns ""
            !*/

            bool push(frame f);
            /*!
                requires
                    - if is_image_encoder() == true, then f.is_image() == true
                    - if is_audio_encoder() == true, then f.is_audio() == true
                ensures
                    - If f does not have matching settings to the codec, it is either
                      resized or resampled before being pushed to the muxer.
                    - Encodes and writes the encoded data to file/socket
                    - Returns true if successfully encoded.
                    - Returns false if either EOF, i.e. flush() has been previously called,
                      or an error occurred, in which case is_open() == false.
            !*/

            void flush();
            /*!
                ensures
                    - Flushes the file.
                    - is_open() == false
            !*/

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
                        logger_dlib_wrapper() << LINFO 
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
                        logger_dlib_wrapper() << LINFO
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
                        logger_dlib_wrapper() << LINFO
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
                        logger_dlib_wrapper() << LINFO
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
                        logger_dlib_wrapper() << LINFO
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
                        logger_dlib_wrapper() << LINFO 
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

            details::register_ffmpeg();

            packet = make_avpacket();
            const AVCodec* pCodec = nullptr;

            if (args_.args_codec.codec != AV_CODEC_ID_NONE)
                pCodec = avcodec_find_encoder(args_.args_codec.codec);
            else if (!args_.args_codec.codec_name.empty())
                pCodec = avcodec_find_encoder_by_name(args_.args_codec.codec_name.c_str());

            if (!pCodec)
                return fail("Codec ",  avcodec_get_name(args_.args_codec.codec), " or ", args_.args_codec.codec_name, " not found");

            pCodecCtx.reset(avcodec_alloc_context3(pCodec));
            if (!pCodecCtx)
                return fail("AV : failed to allocate codec context for ", pCodec->name, " : likely ran out of memory");

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
                    return fail(pCodec->name, " is an image codec. height, width, fmt (pixel format) and framerate must be set");
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
                    return fail(pCodec->name, " is an audio codec. sample_rate, channel_layout and fmt (sample format) must be set");
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
                return fail("avcodec_open2() failed : ", get_av_error(ret));

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
                    logger_dlib_wrapper() << LERROR << "avcodec_send_frame() failed : " << get_av_error(ret);
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
                    logger_dlib_wrapper() << LERROR << "avcodec_receive_packet() failed : " << get_av_error(ret);
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
            st.args_ = a;

            if (!st.args_.enable_audio && !st.args_.enable_image)
                return fail("You need to set at least one of `enable_audio` or `enable_image`");

            {
                st.connecting_time = system_clock::now();
                st.connected_time  = system_clock::time_point::max();

                const char* const format_name   = st.args_.output_format.empty() ? nullptr : st.args_.output_format.c_str();
                const char* const filename      = st.args_.filepath.empty()      ? nullptr : st.args_.filepath.c_str();
                AVFormatContext* pFormatCtx = nullptr;
                int ret = avformat_alloc_output_context2(&pFormatCtx, nullptr, format_name, filename);

                if (ret < 0)
                    return fail("avformat_alloc_output_context2() failed : ", get_av_error(ret));

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
                        logger_dlib_wrapper() << LERROR << "av_interleaved_write_frame() failed : " << get_av_error(ret);
                    return ret == 0;
                };

                // Before we create the encoder, check the codec is supported by this muxer
                const auto supported_codecs = list_codecs_for_muxer(st.pFormatCtx->oformat);

                if (std::find_if(begin(supported_codecs), end(supported_codecs), [&](const auto& supported) {
                    return args.args_codec.codec != AV_CODEC_ID_NONE ? 
                                supported.codec_id   == args.args_codec.codec :
                                supported.codec_name == args.args_codec.codec_name;
                }) == end(supported_codecs))
                {
                    logger_dlib_wrapper() << LERROR
                        << "Codec " << avcodec_get_name(args.args_codec.codec) << " or " << args.args_codec.codec_name
                        << " cannot be stored in this file";
                    logger_dlib_wrapper() << LINFO 
                        << "List of supported codecs for muxer " << st.pFormatCtx->oformat->name << " in this installation of ffmpeg:";
                    for (const auto& supported : supported_codecs)
                        logger_dlib_wrapper() << LINFO << "    " << supported.codec_name;
                    return false;
                }

                // Codec is supported by muxer, so create encoder
                enc = encoder(args, handle_packet);

                if (!enc.is_open())
                    return false;

                AVStream* stream = avformat_new_stream(st.pFormatCtx.get(), enc.pCodecCtx->codec);

                if (!stream)
                    return fail("avformat_new_stream() failed");

                stream->id          = stream_counter;
                stream->time_base   = enc.pCodecCtx->time_base;
                ++stream_counter;

                int ret = avcodec_parameters_from_context(stream->codecpar, enc.pCodecCtx.get());

                if (ret < 0)
                    return fail("avcodec_parameters_from_context() failed : ", get_av_error(ret));

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
                    return fail("avio_open2() failed : ", get_av_error(ret));
            }

            av_dict opt = st.args_.format_options;

            int ret = avformat_write_header(st.pFormatCtx.get(), opt.get());

            if (ret < 0)
                return fail("avformat_write_header() failed : ", get_av_error(ret));

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
                    return fail("frame is an image type but image encoder is not initialized");

                return st.encoder_image.push(std::move(f));
            }

            else if (f.is_audio())
            {
                if (!st.encoder_audio.is_open())
                    return fail("frame is of audio type but audio encoder is not initialized");

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
                logger_dlib_wrapper() << LERROR << "av_write_trailer() failed : " << details::get_av_error(ret);

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
