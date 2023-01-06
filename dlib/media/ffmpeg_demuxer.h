// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_DEMUXER
#define DLIB_FFMPEG_DEMUXER

#include <queue>
#include <functional>
#include <unordered_map>
#include "ffmpeg_utils.h"

namespace dlib
{
    // ---------------------------------------------------------------------------------------------------

    struct decoder_image_args
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class groups a set of arguments passed to ffmpeg_decoder and ffmpeg_demuxer.
                Non-default values will configure an image resizer which will transform images
                from the decoder's/demuxer's format to the desired presentation format.
        !*/

        // Height of extracted frames. If 0, use whatever comes out decoder
        int             h   = 0;
        // Width of extracted frames. If 0, use whatever comes out decoder
        int             w   = 0;
        // Pixel format of extracted frames. If AV_PIX_FMT_NONE, use whatever comes out decoder
        AVPixelFormat   fmt = AV_PIX_FMT_NONE;
    };

    struct decoder_audio_args
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class groups a set of arguments passed to ffmpeg_decoder and ffmpeg_demuxer.
                Non-default values will configure an audio resampler which will transform audio frames
                from the decoder's/demuxer's format to the desired presentation format.
        !*/

        // Sample rate of audio frames. If 0, use whatever comes out decoder
        int             sample_rate     = 0;
        // Channel layout (mono, stereo) of audio frames
        uint64_t        channel_layout  = AV_CH_LAYOUT_STEREO;
        // Sample format of audio frames. If AV_SAMPLE_FMT_NONE, use whatever comes out decoder
        AVSampleFormat  fmt             = AV_SAMPLE_FMT_NONE;
    };

    struct decoder_codec_args
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class groups a set of arguments passed to ffmpeg_decoder and ffmpeg_demuxer.
                Non-default values will configure the codecs.
                Note, for ffmpeg_decoder, these are essential as they cannot be guessed.
                For ffmpeg_demuxer, these are derived from the input file.
                Note, for some demuxers, you may still want to set these. For example, network demuxers
                such as RTSP may require setting codec_options.
        !*/

        // Codec ID used to configure the decoder. Used by ffmpeg_decoder, NOT by ffmpeg_demuxer.
        AVCodecID   codec       = AV_CODEC_ID_NONE;
        // Codec name used to configure the decoder. This is used if codec == AV_CODEC_ID_NONE. Used by ffmpeg_decoder, NOT by ffmpeg_demuxer.
        std::string codec_name  = "";
        // Sets AVCodecContext::thread_count if non-negative. Otherwise, ffmpeg's default is used.
        int         nthreads    = -1;
        // Sets AVCodecContext::bit_rate if non-negative. Otherwise, ffmpeg's default is used.
        int64_t     bitrate     = -1;
        // ORed with AVCodecContext::flags if non-negative. Otherwise, ffmpeg's default is used.
        int         flags       = 0;
        // A dictionary of AVCodecContext and codec-private options. Used by avcodec_open2()
        // Note, when using ffmpeg_decoder, either codec or codec_name have to be specified.
        // When using ffmpeg_demuxer, codec and codec_name will be ignored.
        // This is less likely to be used when using ffmpeg_demuxer.
        std::unordered_map<std::string, std::string> codec_options;
    };

    // ---------------------------------------------------------------------------------------------------

    enum decoder_status
    {
        DECODER_CLOSED = -1,
        DECODER_EAGAIN,
        DECODER_FRAME_AVAILABLE
    };

    // ---------------------------------------------------------------------------------------------------

    class ffmpeg_decoder;
    class ffmpeg_demuxer;

    namespace details
    {
        class decoder_extractor
        {
        public:
            struct args
            {
                decoder_codec_args  args_codec;
                decoder_image_args  args_image;
                decoder_audio_args  args_audio;
                AVRational          time_base;
            };

            decoder_extractor() = default;

            decoder_extractor(
                const args& a,
                av_ptr<AVCodecContext> pCodecCtx_,
                const AVCodec* codec
            );

            bool            is_open()           const noexcept;
            bool            is_image_decoder()  const noexcept;
            bool            is_audio_decoder()  const noexcept;
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

            bool push(const av_ptr<AVPacket>& pkt);
            decoder_status read(Frame& dst_frame);

        private:
            friend class dlib::ffmpeg_decoder;
            friend class dlib::ffmpeg_demuxer;

            args                    args_;
            uint64_t                next_pts    = 0;
            int                     stream_id   = -1;
            av_ptr<AVCodecContext>  pCodecCtx;
            av_ptr<AVFrame>         frame;
            sw_image_resizer        resizer_image;
            sw_audio_resampler      resizer_audio;
            std::queue<Frame>       frame_queue;
        };
    }

    // ---------------------------------------------------------------------------------------------------
    
    class ffmpeg_decoder
    {
    public:
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class is a libavcodec wrapper which decodes video or audio from raw memory.
        !*/

        struct args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                   Constructor arguments for ffmpeg_decoder
            !*/

            decoder_codec_args args_codec;
            decoder_image_args args_image;
            decoder_audio_args args_audio;
        };

        ffmpeg_decoder() = default;
        /*!
            ensures
                - is_open() == false
        !*/

        explicit ffmpeg_decoder(const args &a);
        /*!
            ensures
                - Creates a decoder using args.
        !*/

        bool is_open() const noexcept;
        /*!
            ensures
                - returns true if codec is openened or nframes_available() > 0
        !*/

        bool is_image_decoder() const noexcept;
        /*!
            ensures
                - returns true if codec is openened and codec is an image/gif/video codec
        !*/

        bool is_audio_decoder() const noexcept;
        /*!
            ensures
                - returns true if codec is openened and codec is an audio codec
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

        /*! video properties !*/

        int height() const noexcept;
        /*!
            requires
                - is_image_decoder() == true
                - must have called push_encoded() enough times such that nframes_available() > 0
            ensures
                - returns height of images to be returned by read()
                - If decoder_image_args::h > 0, then frames returned by read() are automatically scaled.
        !*/

        int width() const noexcept;
        /*!
            requires
                - is_image_decoder() == true
                - must have called push_encoded() enough times such that nframes_available() > 0
            ensures
                - returns width of images to be returned by read()
                - If decoder_image_args::w > 0, then frames returned by read() are automatically scaled.
        !*/

        AVPixelFormat pixel_fmt() const noexcept;
        /*!
            requires
                - is_image_decoder() == true
                - must have called push_encoded() enough times such that nframes_available() > 0 
            ensures
                - returns pixel format of images to be returned by read()
                - If decoder_image_args::fmt > 0, then frames returned by read() are automatically scaled and converted.
        !*/

        /*! audio properties !*/

        int sample_rate() const noexcept;
        /*!
            requires
                - is_audio_decoder() == true
                - must have called push_encoded() enough times such that nframes_available() > 0
            ensures
                - returns sample rate of audio frames to be returned by read()
                - If decoder_audio_args::sample_rate > 0, then frames returned by read() are automatically resampled.
        !*/

        uint64_t channel_layout() const noexcept;
        /*!
            requires
                - is_audio_decoder() == true
                - must have called push_encoded() enough times such that nframes_available() > 0
            ensures
                - returns channel_layout of audio frames to be returned by read()
                - If decoder_audio_args::channel_layout > 0, then frames returned by read() are automatically resampled and converted.
                - See documentation of AVFrame::channel_layout
        !*/

        AVSampleFormat sample_fmt() const noexcept;
        /*!
            requires
                - is_audio_decoder() == true
                - must have called push_encoded() enough times such that nframes_available() > 0
            ensures
                - returns sample format (s16, u8, etc) of audio frames to be returned by read()
                - If decoder_audio_args::fmt > 0, then frames returned by read() are automatically resampled and converted.
        !*/

        int nchannels() const noexcept;
        /*!
            requires
                - is_audio_decoder() == true
                - must have called push_encoded() enough times such that nframes_available() > 0
            ensures
                - returns number of channels of audio frames to be returned by read()
                - If decoder_audio_args::channel_layout > 0, then frames returned by read() are automatically resampled and converted.
                - See documentation of AVFrame::channel_layout
        !*/

        bool push_encoded(const uint8_t *encoded, int nencoded);
        /*!
            requires
                - is_open() == true
            ensures
                - encodes data.
                - if (returns true)
                    return value of nframes_available() might change unless more input is required
                  else
                    decoder is closed but frames may still be available when calling read() so is_open() may still return true
        !*/

        void flush();
        /*!
            requires
                - is_open() == true
            ensures
                - calls push_encoded(nullptr, 0)
        !*/

        int nframes_available() const;
        /*!
            ensures
                - returns the number of frames available which can be retrieved by calling read()
        !*/

        decoder_status read(Frame& dst_frame);
        /*!
            ensures
                - if (DECODER_EAGAIN)
                    dst_frame.is_empty() == true
                    need to call push_encoded() again or flush()
                  else if (DECODER_FRAME_AVAILABLE)
                    dst_frame.is_empty() == false
                  else if (DECODER_CLOSED)
                    is_open() == false
                    dst_frame.is_empty() == true
        !*/

    private:
        bool push_encoded_padded(const uint8_t *encoded, int nencoded);

        std::vector<uint8_t>            encoded_buffer;
        av_ptr<AVCodecParserContext>    parser;
        av_ptr<AVPacket>                packet;
        details::decoder_extractor      extractor;
    };

    // ---------------------------------------------------------------------------------------------------

    class ffmpeg_demuxer
    {
    public:
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class is a libavformat wrapper which decodes video or audio from file.
                It is analoguous to OpenCV's cv::VideoCapture class but is more configurable,
                and supports audio, devices (X11, webcam, microphone, ...) and network streams
                such as RTSP, HTTP, and more.
                Note that a video file, e.g. MP4, can contain multiple streams: video, audio, subtitles, etc.
                This class can decode both video and audio at the same time.
                For audio files, e.g. MP3, FLAG, it only decodes audio. (obs)
        !*/

        struct args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                   Constructor arguments for ffmpeg_demuxer
            !*/

            args() = default;

            args(std::string filepath_) : filepath{std::move(filepath_)} {}
            /*!
                ensures
                    - Returns basic arguments using filepath only. Everything else is either defaulted or guessed by libavformat.
            !*/

            // Filepath, URL or device
            std::string filepath;

            // Input format hint. e.g. 'rtsp', 'X11', etc. 99% of the time, users are not required to specify this as libavformat tries to guess it.
            std::string input_format;

            // Sets AVFormatContext::probsize
            // Please see libavformat documentation for more details
            int probesize{-1};

            // Only relevant to network demuxers such as RTSP, HTTP etc.
            // Connection/listening timeout. 
            std::chrono::milliseconds connect_timeout{std::chrono::milliseconds::max()};
            
            // Only relevant to network demuxers such as RTSP, HTTP etc
            // Read timeout. 
            std::chrono::milliseconds read_timeout{std::chrono::milliseconds::max()};

            // Only relevant to network demuxers such as RTSP, HTTP etc
            // Connection/listening interruption callback.
            // The constructor periodially calls interrupter() while waiting on the network. If it returns true, then
            // the connection is aborted and the demuxer is closed.
            // So user may use this in conjunction with some thread-safe shared state to signal an abort/interrupt.
            std::function<bool()> interrupter;

            // A dictionary filled with AVFormatContext and demuxer-private options. Used by avformat_open_input().
            // Please see libavformat documentation for more details
            std::unordered_map<std::string, std::string> format_options;

            // Video stream arguments
            struct : decoder_codec_args, decoder_image_args{} image_options;
            // Audio stream arguments
            struct : decoder_codec_args, decoder_audio_args{} audio_options;
            // Whether or not to decode video stream
            bool enable_image = true;
            // Whether or not to decode audio stream
            bool enable_audio = true;
        };

        ffmpeg_demuxer() = default;
        ffmpeg_demuxer(const args& a);
        ffmpeg_demuxer(ffmpeg_demuxer&& other) noexcept;
        ffmpeg_demuxer& operator=(ffmpeg_demuxer&& other) noexcept;

        bool is_open()          const noexcept;
        bool audio_enabled()    const noexcept;
        bool video_enabled()    const noexcept;

        /*! video dims !*/
        int             height()                    const noexcept;
        int             width()                     const noexcept;
        AVPixelFormat   pixel_fmt()                 const noexcept;
        float           fps()                       const noexcept;
        int             estimated_nframes()         const noexcept;
        AVCodecID       get_video_codec_id()        const noexcept;
        std::string     get_video_codec_name()      const noexcept;
        /*!audio dims! */
        int             sample_rate()               const noexcept;
        uint64_t        channel_layout()            const noexcept;
        AVSampleFormat  sample_fmt()                const noexcept;
        int             nchannels()                 const noexcept;
        int             estimated_total_samples()   const noexcept;
        AVCodecID       get_audio_codec_id()        const noexcept;
        std::string     get_audio_codec_name()      const noexcept;
        float           duration()                  const noexcept;

        bool read(Frame& frame);

        /*! metadata! */
        std::unordered_map<int, std::unordered_map<std::string, std::string>> get_all_metadata() const noexcept;
        std::unordered_map<std::string,std::string> get_video_metadata() const noexcept;
        std::unordered_map<std::string,std::string> get_audio_metadata() const noexcept;
        float get_rotation_angle() const noexcept;

    private:
        bool open(const args& a);
        bool fill_queue();
        bool interrupt_callback();
        void populate_metadata();

        struct {
            args                    args_;
            bool                    opened = false;
            av_ptr<AVFormatContext> pFormatCtx;
            av_ptr<AVPacket>        packet;
            std::chrono::system_clock::time_point connecting_time{};
            std::chrono::system_clock::time_point connected_time{};
            std::chrono::system_clock::time_point last_read_time{};
            std::unordered_map<int, std::unordered_map<std::string, std::string>> metadata;
            details::decoder_extractor channel_video;
            details::decoder_extractor channel_audio;
            std::queue<Frame> frame_queue;
        } st;
    };

    // ---------------------------------------------------------------------------------------------------
}

#endif //DLIB_FFMPEG_DEMUXER