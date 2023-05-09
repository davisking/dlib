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
    namespace ffmpeg
    {
// ---------------------------------------------------------------------------------------------------

        struct decoder_image_args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments passed to the decoder and demuxer classes.
                    Non-default values will configure an image resizer which will transform images
                    from the decoder's/demuxer's internal format to the desired presentation format.
                    In a lot of codecs, RGB isn't the default format, usually YUV is. So,
                    it is usually necessary to reformat the frame into RGB or some other presentation
                    format. The ffmpeg object used to do this can simultaneously resize the image.
                    Therefore, the API allows users to optionally resize the image, as well as convert to RGB,
                    before being presented to user, as a possible optimization.

                    In the case of demuxer, if:
                        - h > 0
                        - w > 0
                        - and the demuxer is a device like v4l2 or xcbgrab
                    then we attempt to set the video size of the device before decoding.
                    Otherwise, the image dimensions set the bilinear resizer which resizes frames AFTER decoding.

                    Furthermore, in the case of demuxer, if:
                        - framerate > 0
                        - and the demuxer is a device like v4l2 or xcbgrab
                    then we attempt to set the framerate of the input device before decoding.
            !*/

            // Height of extracted frames. If 0, use whatever comes out decoder
            int h{0};

            // Width of extracted frames. If 0, use whatever comes out decoder
            int w{0};

            // Pixel format of extracted frames. If AV_PIX_FMT_NONE, use whatever comes out decoder. The default is AV_PIX_FMT_RGB24
            AVPixelFormat fmt{AV_PIX_FMT_RGB24};

            // Sets the output frame rate for any device that allows you to do so, e.g. webcam, x11grab, etc. Does not apply to files. If -1, ignored.
            int framerate{-1};
        };

// ---------------------------------------------------------------------------------------------------

        struct decoder_audio_args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments passed to the decoder and demuxer classes.
                    Non-default values will configure an audio resampler which will transform audio frames
                    from the decoder's/demuxer's format to the desired presentation format.
            !*/

            // Sample rate of audio frames. If 0, use whatever comes out decoder
            int sample_rate{0};
            
            // Channel layout (mono, stereo) of audio frames
            uint64_t channel_layout{AV_CH_LAYOUT_STEREO};

            // Sample format of audio frames. If AV_SAMPLE_FMT_NONE, use whatever comes out decoder. Default is AV_SAMPLE_FMT_S16
            AVSampleFormat fmt{AV_SAMPLE_FMT_S16};
        };

// ---------------------------------------------------------------------------------------------------

        struct decoder_codec_args
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class groups a set of arguments passed to the decoder and demuxer classes.
                    Non-default values will configure the codecs.
                    Note, for decoder, these are essential as they cannot be guessed.
                    For demuxer, these are derived from the input file or network connection.
                    Note, for some demuxers, you may still want to set these. For example, network demuxers
                    such as RTSP or HTTP may require setting codec_options.
            !*/

            // Codec ID used to configure the decoder. Used by decoder, IGNORED by demuxer.
            AVCodecID codec{AV_CODEC_ID_NONE};

            // Codec name used to configure the decoder. This is used if codec == AV_CODEC_ID_NONE. Used by decoder, IGNORED by demuxer.
            std::string codec_name;

            // A dictionary of AVCodecContext and codec-private options. Used by "avcodec_open2()"
            // This is less likely to be used when using demuxer objects, unless using network demuxers.
            std::unordered_map<std::string, std::string> codec_options;

            // Sets AVCodecContext::bit_rate if non-negative. Otherwise, ffmpeg's default is used.
            int64_t bitrate{-1};

            // OR-ed with AVCodecContext::flags if non-negative. Otherwise, ffmpeg's default is used.
            int flags{0};

            // Sets AVCodecContext::thread_count if non-negative. Otherwise, ffmpeg-s default is used. 
            // Some codecs can be multi-threaded. Setting this enables this and controls the size of the thread pool.
            // Not all codecs can be parallelised.
            int thread_count{-1};
        };

// ---------------------------------------------------------------------------------------------------

        enum decoder_status
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This enum describes the return value of decoder::read()
            !*/

            DECODER_CLOSED = -1,
            DECODER_EAGAIN,
            DECODER_FRAME_AVAILABLE
        };

// ---------------------------------------------------------------------------------------------------
        
        class decoder
        {
        public:
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class is a libavcodec wrapper which decodes video or audio from raw memory.
                    Note, if you are reading raw memory from file, it is easier to use demuxer
                    as it also works with raw codec files like .h264 files.
                    This class is suitable for example when reading raw encoded data from a socket,
                    or interfacing with another library that provides encoded data.
            !*/

            struct args
            {
                /*!
                    WHAT THIS OBJECT REPRESENTS
                        This holds constructor arguments for decoder.
                !*/

                decoder_codec_args args_codec;
                decoder_image_args args_image;
                decoder_audio_args args_audio;
            };

            decoder() = default;
            /*!
                ensures
                    - is_open() == false
            !*/

            explicit decoder(const args &a);
            /*!
                ensures
                    - Creates a decoder using args.
            !*/

            bool is_open() const noexcept;
            /*!
                ensures
                    - returns true if frames are available, i.e. read() == true
            !*/

            bool is_image_decoder() const noexcept;
            /*!
                ensures
                    - returns true if is_open() == true and codec is an image/gif/video codec
            !*/

            bool is_audio_decoder() const noexcept;
            /*!
                ensures
                    - returns true if is_open() == true and codec is an audio codec
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
                    - is_image_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                      The height cannot be deduced from codec only. It can only be deduced from decoded data.
                ensures 
                    - returns height of images to be returned by read()
                    - If decoder_image_args::h > 0, then frames returned by read() are automatically scaled.
            !*/

            int width() const noexcept;
            /*!
                requires
                    - is_image_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns width of images to be returned by read()
                    - If decoder_image_args::w > 0, then frames returned by read() are automatically scaled.
            !*/

            AVPixelFormat pixel_fmt() const noexcept;
            /*!
                requires
                    - is_image_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns pixel format of images to be returned by read()
                    - If decoder_image_args::fmt > 0, then frames returned by read() are automatically scaled and converted.
            !*/

            /*! audio properties !*/

            int sample_rate() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns sample rate of audio frames to be returned by read()
                    - If decoder_audio_args::sample_rate > 0, then frames returned by read() are automatically resampled.
            !*/

            uint64_t channel_layout() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns channel_layout of audio frames to be returned by read()
                    - If decoder_audio_args::channel_layout > 0, then frames returned by read() are automatically resampled and converted.
                    - See documentation of AVFrame::channel_layout
            !*/

            AVSampleFormat sample_fmt() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns sample format (s16, u8, etc) of audio frames to be returned by read()
                    - If decoder_audio_args::fmt > 0, then frames returned by read() are automatically resampled and converted.
            !*/

            int nchannels() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
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
                    - Flushes the decoder. This must be called when there is no more data to be decoded. Last remaining frames will be available.
                    - calls push_encoded(nullptr, 0)
            !*/

            decoder_status read(frame& dst_frame);
            /*!
                ensures
                    - Attempts to read a frame, storing the result in dst_frame.
                    - If it is successful then returns DECODER_FRAME_AVAILABLE and 
                      dst_frame.is_empty() == false.  Otherwise, returns one of the 
                      following:
                        - DECODER_EAGAIN: this indicates more encoded data is required to decode
                          additional frames.  In particular, you will need to keep calling
                          push_encoded() until this function returns DECODER_FRAME_AVAILABLE.
                          Alternatively, if there is no more encoded data, call flush(). This will
                          flush the decoder, resulting in additional frames being available.  After
                          the decoder is flushed, this function function will return
                          DECODER_FRAME_AVAILABLE until it finally calls DECODER_CLOSED.
                        - DECODER_CLOSED: This indicates there aren't any more frames.  If this happens
                          then it also means that is_open() == false and you can no longer retrieve anymore frames.
            !*/

        private:
            friend class demuxer;

            bool open (
                const args&                     a,
                details::av_ptr<AVCodecContext> pCodecCtx_,
                const AVCodec*                  codec,
                AVRational                      timebase_
            );

            bool push_encoded_padded(const uint8_t *encoded, int nencoded);
            bool push(const details::av_ptr<AVPacket>& pkt);

            args                                    args_;
            AVRational                              timebase;
            details::av_ptr<AVCodecParserContext>   parser;
            details::av_ptr<AVCodecContext>         pCodecCtx;
            details::av_ptr<AVPacket>               packet;
            details::av_ptr<AVFrame>                avframe;
            details::resizer                        resizer_image;
            details::resampler                      resizer_audio;
            std::vector<uint8_t>                    encoded_buffer;
            std::queue<frame>                       frame_queue;
            uint64_t                                next_pts{0};
        };

// ---------------------------------------------------------------------------------------------------

        class demuxer
        {
        public:
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This class is a libavformat wrapper which demuxes video and/or audio streams from file and decodes them.
                    It is analogous to OpenCV's cv::VideoCapture class but is more configurable, supports audio, 
                    devices (X11, webcam, microphone, ...) and network streams such as RTSP, HTTP, and more.
                    Note that a video file, e.g. MP4, can contain multiple streams: video, audio, subtitles, etc.
                    This class can decode both video and audio at the same time.
                    For audio files, e.g. MP3, FLAG, it only decodes audio. (obs)
            !*/

            struct args
            {
                /*!
                    WHAT THIS OBJECT REPRESENTS
                        This holds constructor arguments for demuxer.
                !*/

                args() = default;
                /*!
                    ensures
                        - Default constructor
                !*/

                args(const std::string& filepath);
                /*!
                    ensures
                        - this->filepath = filepath
                !*/

                args(const std::string& filepath, video_enabled_t video_on, audio_enabled_t audio_on);
                /*!
                    ensures
                        - this->filepath        = filepath
                        - this->enable_image    = video_on.enabled
                        - this->enable_audio    = audio_on.enabled
                !*/

                // Filepath, URL or device
                std::string filepath;

                // Input format hint. e.g. 'rtsp', 'X11', 'alsa', etc. 99% of the time, users are not required to specify this as libavformat tries to guess it.
                std::string input_format;

                // A dictionary filled with AVFormatContext and demuxer-private options. Used by "avformat_open_input()"".
                // Please see libavformat documentation for more details
                std::unordered_map<std::string, std::string> format_options;

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
                // The constructor periodically calls interrupter() while waiting on the network. If it returns true, then
                // the connection is aborted and the demuxer is closed.
                // So user may use this in conjunction with some thread-safe shared state to signal an abort/interrupt.
                std::function<bool()> interrupter;

                // Video stream arguments
                struct : decoder_codec_args, decoder_image_args{} args_image;

                // Audio stream arguments
                struct : decoder_codec_args, decoder_audio_args{} args_audio;
                
                // Whether or not to decode video stream.
                bool enable_image{true};

                // Whether or not to decode audio stream.
                bool enable_audio{true};
            };

            demuxer() = default;
            /*!
                ensures
                    - is_open() == false
            !*/

            demuxer(const args& a);
            /*!
                ensures
                    - Creates a demuxer using args.
            !*/

            demuxer(demuxer&& other) noexcept;
            /*!
                ensures
                    - Move constructor
                    - After move, other.is_open() == false
            !*/

            demuxer& operator=(demuxer&& other) noexcept;
            /*!
                ensures
                    - Move assignment
                    - After move, other.is_open() == false
            !*/

            bool is_open() const noexcept;
            /*!
                ensures
                    - returns true if frames are available, i.e. read() == true
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
                    - args.enable_image == true
                ensures
                    - returns true if is_open() == true and an video stream is available and open.
            !*/

            int height() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns height of images to be returned by read()
                    - else
                        - returns 0
            !*/

            int width() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns width of images to be returned by read()
                    - else
                        - returns 0
            !*/

            AVPixelFormat pixel_fmt() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns pixel format of images to be returned by read()
                    - else
                        - returns AV_PIX_FMT_NONE
            !*/

            float fps() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns frames per second (FPS) of video stream (if available)
                    - else
                        - returns 0
            !*/

            int estimated_nframes() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - estimates and returns number of frames in video stream
                    - else
                        - returns 0
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
                        - returns sample rate of audio stream
                    - else
                        - returns 0
            !*/

            uint64_t channel_layout() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns channel layout of audio stream (e.g. AV_CH_LAYOUT_STEREO)
                    - else
                        - returns 0
            !*/

            AVSampleFormat sample_fmt() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns sample format of audio stream (e.g. AV_SAMPLE_FMT_S16)
                    - else
                        - returns AV_SAMPLE_FMT_NONE
            !*/

            int nchannels() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns number of audio channels in audio stream (e.g. 1 for mono, 2 for stereo)
                    - else
                        - returns 0
            !*/

            int estimated_total_samples() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns estimated number of samples in audio stream
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

            float duration() const noexcept;
            /*!
                ensures 
                    - if (is_open())
                        - returns an estimated duration of file in seconds
                    - else
                        - returns 0
            !*/

            const std::unordered_map<std::string, std::string>& get_metadata() const noexcept;
            /*!
                ensures 
                    - if (is_open())
                        - returns metadata in file
            !*/

            float get_rotation_angle() const noexcept;
            /*!
                ensures 
                    - if (is_open())
                        - returns video rotation angle from metadata if available
                    - else
                        - returns 0
            !*/

            bool read(frame& frame);
            /*!
                ensures 
                    - if (is_open())
                        - returns true and frame.is_empty() == false
                    - else
                        - returns false and frame.is_empty() == true
            !*/

        private:
            bool open(const args& a);
            bool object_alive() const noexcept;
            bool fill_queue();
            bool interrupt_callback();
            void populate_metadata();

            struct {
                args                                    args_;
                details::av_ptr<AVFormatContext>        pFormatCtx;
                details::av_ptr<AVPacket>               packet;
                std::chrono::system_clock::time_point   connecting_time{};
                std::chrono::system_clock::time_point   connected_time{};
                std::chrono::system_clock::time_point   last_read_time{};
                std::unordered_map<std::string, std::string> metadata;
                decoder                                 channel_video;
                decoder                                 channel_audio;
                int                                     stream_id_video{-1};
                int                                     stream_id_audio{-1};
                std::queue<frame>                       frame_queue;
            } st;
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class image_type,
          is_image_check<image_type> = true
        >
        void load_frame(
            image_type& image,
            const std::string& file_name
        );
        /*!
            requires
                - image_type must be a type conforming to the generic image interface.
            ensures
                - reads the first frame of the image or video pointed by file_name and
                  loads into image.
        !*/
    }

// ---------------------------------------------------------------------------------------------------

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

        inline decoder::decoder(const args &a)
        {
            using namespace details;

            DLIB_ASSERT(a.args_codec.codec != AV_CODEC_ID_NONE || a.args_codec.codec_name != "", "At least args_codec.codec or args_codec.codec_name must be set");
            
            register_ffmpeg();
            
            const AVCodec* pCodec = nullptr;

            if (a.args_codec.codec != AV_CODEC_ID_NONE)
                pCodec = avcodec_find_decoder(a.args_codec.codec);
            else if (!a.args_codec.codec_name.empty())
                pCodec = avcodec_find_decoder_by_name(a.args_codec.codec_name.c_str());

            if (!pCodec)
            {
                logger_dlib_wrapper() << LERROR 
                    << "Codec "
                    << avcodec_get_name(a.args_codec.codec)
                    << " / "
                    << a.args_codec.codec_name
                    << " not found.";
                return;
            }

            av_ptr<AVCodecContext> pCodecCtx_{avcodec_alloc_context3(pCodec)};

            if (!pCodecCtx_)
            {
                logger_dlib_wrapper() << LERROR << "avcodec_alloc_context3() failed to allocate codec context for " << pCodec->name;
                return;
            }

            if (pCodecCtx_->codec_id == AV_CODEC_ID_AAC)
                pCodecCtx_->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;


            timebase = pCodecCtx_->time_base;

            if (!open({a.args_codec, a.args_image, a.args_audio}, std::move(pCodecCtx_), pCodec,  timebase))
                return;

            // It's very likely ALL the PCM codecs don't require a parser
            const bool no_parser_required = pCodec->id == AV_CODEC_ID_PCM_S16LE ||
                                            pCodec->id == AV_CODEC_ID_PCM_U8;

            if (!no_parser_required)
            {
                parser.reset(av_parser_init(pCodec->id));
                if (!parser)
                {
                    logger_dlib_wrapper() << LERROR << "av_parser_init() failed codec " << pCodec->name << " not found";
                    pCodecCtx = nullptr;
                    return;
                }
            }
        }

// ---------------------------------------------------------------------------------------------------

        inline bool decoder::open (
            const args&                     a,
            details::av_ptr<AVCodecContext> pCodecCtx_,
            const AVCodec*                  codec,
            AVRational                      timebase_
        )
        {
            using namespace details;

            args_    = a;
            timebase = timebase_;
            packet   = make_avpacket();
            avframe  = make_avframe();

            if (args_.args_codec.bitrate > 0)
                pCodecCtx_->bit_rate = args_.args_codec.bitrate;
            if (args_.args_codec.flags > 0)
                pCodecCtx_->flags |= args_.args_codec.flags;
            if (args_.args_codec.thread_count > 0)
                pCodecCtx_->thread_count = args_.args_codec.thread_count;

            av_dict opt = args_.args_codec.codec_options;
            int ret = avcodec_open2(pCodecCtx_.get(), codec, opt.get());

            if (ret < 0)
                return fail("avcodec_open2() failed : ", get_av_error(ret));
            
            pCodecCtx = std::move(pCodecCtx_);

            // Set image scaler if possible
            if (pCodecCtx->height > 0 &&
                pCodecCtx->width  > 0 &&
                pCodecCtx->pix_fmt != AV_PIX_FMT_NONE)
            {
                resizer_image.reset(
                    pCodecCtx->height,
                    pCodecCtx->width,
                    pCodecCtx->pix_fmt,
                    args_.args_image.h > 0                  ? args_.args_image.h   : pCodecCtx->height,
                    args_.args_image.w > 0                  ? args_.args_image.w   : pCodecCtx->width,
                    args_.args_image.fmt != AV_PIX_FMT_NONE ? args_.args_image.fmt : pCodecCtx->pix_fmt
                );   
            }

            const uint64_t pCodecCtx_channel_layout = details::get_layout(pCodecCtx.get());

            // Set audio resampler if possible
            if (pCodecCtx->sample_rate > 0                  &&
                pCodecCtx->sample_fmt != AV_SAMPLE_FMT_NONE &&
                pCodecCtx_channel_layout > 0)
            {
                resizer_audio.reset(
                    pCodecCtx->sample_rate,
                    pCodecCtx_channel_layout,
                    pCodecCtx->sample_fmt,
                    args_.args_audio.sample_rate > 0            ? args_.args_audio.sample_rate      : pCodecCtx->sample_rate,
                    args_.args_audio.channel_layout > 0         ? args_.args_audio.channel_layout   : pCodecCtx_channel_layout,
                    args_.args_audio.fmt != AV_SAMPLE_FMT_NONE  ? args_.args_audio.fmt              : pCodecCtx->sample_fmt
                );
            }

            return true;
        }

// ---------------------------------------------------------------------------------------------------

        enum extract_state
        {
            EXTRACT_SEND_PACKET,
            EXTRACT_READ_FRAME_THEN_DONE,
            EXTRACT_READ_FRAME_THEN_SEND_PACKET,
            EXTRACT_DONE,
            EXTRACT_ERROR = -1
        };

        inline bool decoder::push(const details::av_ptr<AVPacket>& pkt)
        {
            using namespace std::chrono;
            using namespace details;

            const auto send_packet = [&](extract_state& state)
            {
                const int ret = avcodec_send_packet(pCodecCtx.get(), pkt.get());

                if (ret >= 0) {
                    state   = EXTRACT_READ_FRAME_THEN_DONE;
                } else if (ret == AVERROR(EAGAIN)) {
                    state   = EXTRACT_READ_FRAME_THEN_SEND_PACKET;
                } else if (ret == AVERROR_EOF) {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_DONE;
                } else {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_ERROR;
                    logger_dlib_wrapper() << LERROR << "avcodec_send_packet() failed : " << get_av_error(ret);
                }
            };

            const auto recv_frame = [&](extract_state& state, bool resend)
            {
                const int ret = avcodec_receive_frame(pCodecCtx.get(), avframe.get());

                if (ret == AVERROR(EAGAIN) && resend)
                    state   = EXTRACT_SEND_PACKET;
                else if (ret == AVERROR(EAGAIN))
                    state   = EXTRACT_DONE;
                else if (ret == AVERROR_EOF) {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_DONE;
                }
                else if (ret < 0)
                {
                    pCodecCtx = nullptr;
                    state   = EXTRACT_ERROR;
                    logger_dlib_wrapper() << LERROR << "avcodec_receive_frame() failed : " << get_av_error(ret);
                }
                else
                {
                    const bool is_video         = pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
                    const AVRational tb         = is_video ? timebase : AVRational{1, avframe->sample_rate};
                    const uint64_t pts          = is_video ? avframe->pts : next_pts;
                    const uint64_t timestamp_ns = av_rescale_q(pts, tb, {1,1000000000});
                    next_pts                    += is_video ? 1 : avframe->nb_samples;

                    frame decoded;
                    frame src;
                    src.f           = std::move(avframe); //make sure you move it back when you're done
                    src.timestamp   = system_clock::time_point{duration_cast<system_clock::duration>(nanoseconds{timestamp_ns})};

                    if (src.is_image())
                    {
                        resizer_image.resize(
                            src,
                            args_.args_image.h > 0                  ? args_.args_image.h :   src.height(),
                            args_.args_image.w > 0                  ? args_.args_image.w :   src.width(),
                            args_.args_image.fmt != AV_PIX_FMT_NONE ? args_.args_image.fmt : src.pixfmt(),
                            decoded);
                    }
                    else
                    {
                        resizer_audio.resize(
                            src,
                            args_.args_audio.sample_rate > 0            ? args_.args_audio.sample_rate      : src.sample_rate(),
                            args_.args_audio.channel_layout > 0         ? args_.args_audio.channel_layout   : src.layout(),
                            args_.args_audio.fmt != AV_SAMPLE_FMT_NONE  ? args_.args_audio.fmt              : src.samplefmt(),
                            decoded);
                    }

                    avframe = std::move(src.f); // move back where it was

                    if (!decoded.is_empty())
                        frame_queue.push(std::move(decoded));
                }
            };

            extract_state state = pCodecCtx ? EXTRACT_SEND_PACKET : EXTRACT_ERROR;

            while (state != EXTRACT_ERROR && state != EXTRACT_DONE)
            {
                switch(state)
                {
                    case EXTRACT_SEND_PACKET:                   send_packet(state);         break;
                    case EXTRACT_READ_FRAME_THEN_DONE:          recv_frame(state, false);   break;
                    case EXTRACT_READ_FRAME_THEN_SEND_PACKET:   recv_frame(state, true);    break;
                    default: break;
                }
            }

            return state != EXTRACT_ERROR;
        }

        inline bool decoder::push_encoded_padded(const uint8_t *encoded, int nencoded)
        {
            using namespace std;
            using namespace details;

            if (!is_open())
                return false;

            const auto parse = [&]
            {
                if (parser)
                {
                    const int ret = av_parser_parse2(
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
                        return fail("AV : error while parsing encoded buffer");

                    encoded  += ret;
                    nencoded -= ret;
                } else
                {
                    /*! Codec does not require parser !*/
                    packet->data = const_cast<uint8_t *>(encoded);
                    packet->size = nencoded;
                    encoded      += nencoded;
                    nencoded     = 0;
                }

                return true;
            };

            const bool flushing = encoded == nullptr && nencoded == 0;
            bool ok = true;

            while (ok && (nencoded > 0 || flushing))
            {
                // Parse data OR flush parser
                ok = parse();

                // If data is available, decode
                if (ok && packet->size > 0)
                    ok = push(packet);
                
                // If flushing, only flush parser once, so break
                if (flushing)
                    break;
            }

            if (flushing)
            {
                // Flush codec. After this, pCodecCtx == nullptr since AVERROR_EOF will be returned at some point.
                ok = push(nullptr);
            }
        
            return ok;
        }

        inline bool decoder::push_encoded(const uint8_t *encoded, int nencoded)
        {
            bool ok = true;

            if (encoded == nullptr && nencoded == 0)
            {
                ok = push_encoded_padded(nullptr, 0);
            }
            else
            {
                if (nencoded > AV_INPUT_BUFFER_PADDING_SIZE)
                {
                    const int blocksize = nencoded - AV_INPUT_BUFFER_PADDING_SIZE;

                    ok = push_encoded_padded(encoded, blocksize);
                    encoded  += blocksize;
                    nencoded -= blocksize; // == AV_INPUT_BUFFER_PADDING_SIZE
                }

                if (ok)
                {
                    encoded_buffer.resize(nencoded + AV_INPUT_BUFFER_PADDING_SIZE);
                    std::memcpy(encoded_buffer.data(), encoded, nencoded);
                    ok = push_encoded_padded(encoded_buffer.data(), nencoded);
                }
            }

            return ok;
        }

        inline void decoder::flush()
        {
            push_encoded(nullptr, 0);
        }

        inline decoder_status decoder::read(frame &dst_frame)
        {
            if (!frame_queue.empty())
            {
                dst_frame = std::move(frame_queue.front());
                frame_queue.pop();
                return DECODER_FRAME_AVAILABLE;
            }

            if (!is_open())
                return DECODER_CLOSED;

            return DECODER_EAGAIN;
        }

        inline bool             decoder::is_open()          const noexcept { return pCodecCtx != nullptr || !frame_queue.empty(); }
        inline bool             decoder::is_image_decoder() const noexcept { return pCodecCtx != nullptr && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO; }
        inline bool             decoder::is_audio_decoder() const noexcept { return pCodecCtx != nullptr && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO; }
        inline AVCodecID        decoder::get_codec_id()     const noexcept { return pCodecCtx != nullptr ? pCodecCtx->codec_id : AV_CODEC_ID_NONE; }
        inline std::string      decoder::get_codec_name()   const noexcept { return pCodecCtx != nullptr ? avcodec_get_name(pCodecCtx->codec_id) : "NONE"; }
        inline int              decoder::height()           const noexcept { return resizer_image.get_dst_h(); }
        inline int              decoder::width()            const noexcept { return resizer_image.get_dst_w(); }
        inline AVPixelFormat    decoder::pixel_fmt()        const noexcept { return resizer_image.get_dst_fmt(); }
        inline int              decoder::sample_rate()      const noexcept { return resizer_audio.get_dst_rate(); }
        inline AVSampleFormat   decoder::sample_fmt()       const noexcept { return resizer_audio.get_dst_fmt(); }
        inline uint64_t         decoder::channel_layout()   const noexcept { return resizer_audio.get_dst_layout(); }
        inline int              decoder::nchannels()        const noexcept { return details::get_nchannels(channel_layout()); }

// ---------------------------------------------------------------------------------------------------

        inline demuxer::args::args(const std::string& filepath_)
        : filepath{filepath_}
        {
        }

        inline demuxer::args::args(
            const std::string& filepath_, 
            video_enabled_t video_on, 
            audio_enabled_t audio_on
        ) : filepath{filepath_},
            enable_image{video_on.enabled},
            enable_audio{audio_on.enabled}
        {
        }

        inline demuxer::demuxer(const args &a)
        {
            if (!open(a))
                st.pFormatCtx = nullptr;
        }

        inline demuxer::demuxer(demuxer &&other) noexcept
        : st{std::move(other.st)}
        {
            if (st.pFormatCtx)
                st.pFormatCtx->opaque = this;
        }

        inline demuxer& demuxer::operator=(demuxer &&other) noexcept
        {
            st = std::move(other.st);
            if (st.pFormatCtx)
                st.pFormatCtx->opaque = this;
            return *this;
        }

        inline bool demuxer::open(const args& a)
        {
            using namespace std;
            using namespace std::chrono;
            using namespace details;

            details::register_ffmpeg();

            st = {};
            st.args_ = a;

            AVFormatContext* pFormatCtx = avformat_alloc_context();
            pFormatCtx->opaque = this;
            pFormatCtx->interrupt_callback.opaque   = pFormatCtx;
            pFormatCtx->interrupt_callback.callback = [](void* ctx) -> int {
                AVFormatContext* pFormatCtx = (AVFormatContext*)ctx;
                demuxer* me = (demuxer*)pFormatCtx->opaque;
                return me->interrupt_callback();
            };

            if (st.args_.probesize > 0)
                pFormatCtx->probesize = st.args_.probesize;

            // Hacking begins. 
            if (st.args_.args_image.h > 0 && 
                st.args_.args_image.w > 0 && 
                st.args_.format_options.find("video_size") == st.args_.format_options.end())
            {
                // See if format supports "video_size"
                st.args_.format_options["video_size"] = std::to_string(st.args_.args_image.w) + "x" + std::to_string(st.args_.args_image.h);
            }

            if (st.args_.args_image.framerate > 0 &&
                st.args_.format_options.find("framerate") == st.args_.format_options.end())
            {
                // See if format supports "framerate"
                st.args_.format_options["framerate"] = std::to_string(st.args_.args_image.framerate);
            }

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(59, 0, 100)
            using AVInputputFormatPtr   = AVInputFormat*;
            using AVCodecPtr            = AVCodec*;
#else
            using AVInputputFormatPtr   = const AVInputFormat*;
            using AVCodecPtr            = const AVCodec*;
#endif

            av_dict opts = st.args_.format_options;
            AVInputputFormatPtr input_format = st.args_.input_format.empty() ? nullptr : av_find_input_format(st.args_.input_format.c_str());

            st.connecting_time = system_clock::now();
            st.connected_time  = system_clock::time_point::max();

            int ret = avformat_open_input(&pFormatCtx,
                                        st.args_.filepath.c_str(),
                                        input_format,
                                        opts.get());

            if (ret != 0)
                return fail("avformat_open_input() failed with error : ", get_av_error(ret));

            if (opts.size() > 0)
            {
                printf("demuxer::args::format_options ignored:\n");
                opts.print();
            }

            st.connected_time = system_clock::now();
            st.pFormatCtx.reset(std::exchange(pFormatCtx, nullptr));

            ret = avformat_find_stream_info(st.pFormatCtx.get(), NULL);

            if (ret < 0)
                return fail("avformat_find_stream_info() failed with error : ", get_av_error(ret));

            const auto setup_stream = [&](bool is_video)
            {
                const AVMediaType media_type = is_video ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;

                AVCodecPtr pCodec = nullptr;
                const int stream_id = av_find_best_stream(st.pFormatCtx.get(), media_type, -1, -1, &pCodec, 0);

                if (stream_id == AVERROR_STREAM_NOT_FOUND)
                    return true; //You might be asking for both video and audio but only video is available. That's OK. Just provide video.

                else if (stream_id == AVERROR_DECODER_NOT_FOUND)
                    return fail("av_find_best_stream() : decoder not found for stream type : ", av_get_media_type_string(media_type));

                else if (stream_id < 0)
                    return fail("av_find_best_stream() failed : ", get_av_error(stream_id));

                av_ptr<AVCodecContext> pCodecCtx{avcodec_alloc_context3(pCodec)};

                if (!pCodecCtx)
                    return fail("avcodec_alloc_context3() failed to allocate codec context for ", pCodec->name);

                const int ret = avcodec_parameters_to_context(pCodecCtx.get(), st.pFormatCtx->streams[stream_id]->codecpar);
                if (ret < 0)
                    return fail("avcodec_parameters_to_context() failed : ", get_av_error(ret));

                if (pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO)
                {
                    if (pCodecCtx->height   == 0 ||
                        pCodecCtx->width    == 0 ||
                        pCodecCtx->pix_fmt  == AV_PIX_FMT_NONE)
                        return fail("Codec parameters look wrong : (h,w,pixel_fmt) : (",
                                pCodecCtx->height, ",",
                                pCodecCtx->width,  ",",
                                get_pixel_fmt_str(pCodecCtx->pix_fmt), ")");
                }
                else if (pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO)
                {
                    check_layout(pCodecCtx.get());

                    if (pCodecCtx->sample_rate == 0 ||
                        pCodecCtx->sample_fmt  == AV_SAMPLE_FMT_NONE ||
                        channel_layout_empty(pCodecCtx.get()))
                        return fail("Codec parameters look wrong :",
                            " sample_rate : ", pCodecCtx->sample_rate,
                            " sample format : ", get_audio_fmt_str(pCodecCtx->sample_fmt),
                            " channel layout : ", get_channel_layout_str(pCodecCtx.get()));
                }
                else
                    return fail("Unrecognized media type ", pCodecCtx->codec_type);

                if (is_video)
                {
                    st.channel_video.open({
                        st.args_.args_image,
                        st.args_.args_image,
                        {}},
                        std::move(pCodecCtx), 
                        pCodec,
                        st.pFormatCtx->streams[stream_id]->time_base
                    );

                    st.stream_id_video = stream_id;
                }
                else
                {
                    st.channel_audio.open({
                        st.args_.args_audio,
                        {},
                        st.args_.args_audio},
                        std::move(pCodecCtx), 
                        pCodec,
                        st.pFormatCtx->streams[stream_id]->time_base
                    );

                    st.stream_id_audio = stream_id;
                }

                return true;
            };

            if (st.args_.enable_image && !setup_stream(true))
                return false;

            if (st.args_.enable_audio && !setup_stream(false))
                return false;

            if (!st.channel_audio.is_open() && !st.channel_video.is_open())
                return fail("At least one of video and audio channels must be enabled");

            populate_metadata();

            st.packet = make_avpacket();
            return true;
        }

        inline bool demuxer::object_alive() const noexcept
        {
            return st.pFormatCtx != nullptr && (st.channel_video.is_open() || st.channel_audio.is_open());
        }

        inline bool demuxer::is_open() const noexcept
        {
            return object_alive() || !st.frame_queue.empty();
        }

        inline bool demuxer::interrupt_callback()
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

        inline void demuxer::populate_metadata()
        {
            AVDictionaryEntry *tag = nullptr;
            while ((tag = av_dict_get(st.pFormatCtx->metadata, "", tag, AV_DICT_IGNORE_SUFFIX)))
                st.metadata.emplace(tag->key, tag->value);
            
            tag = nullptr;
            for (unsigned int i = 0; i < st.pFormatCtx->nb_streams; i++)
                while ((tag = av_dict_get(st.pFormatCtx->streams[i]->metadata, "", tag, AV_DICT_IGNORE_SUFFIX)))
                    st.metadata.emplace(std::string("stream_") + std::to_string(i) + "_" + std::string(tag->key), tag->value);
        }

        inline bool demuxer::fill_queue()
        {
            using namespace std;
            using namespace details;

            if (!st.frame_queue.empty())
                return true;

            decoder* channel{nullptr};

            const auto parse = [&]
            {
                channel = nullptr;
                av_packet_unref(st.packet.get());

                const int ret = av_read_frame(st.pFormatCtx.get(), st.packet.get());

                if (ret == AVERROR_EOF)
                    return false;
   
                else if (ret < 0)
                    return fail("av_read_frame() failed : ", get_av_error(ret));
 
                if (st.packet->stream_index == st.stream_id_video)
                    channel = &st.channel_video;

                else if (st.packet->stream_index == st.stream_id_audio)
                    channel = &st.channel_audio;

                return true;
            };

            bool ok{true};

            while (object_alive() && st.frame_queue.empty() && ok)
            {
                ok = parse();

                if (ok && st.packet->size > 0)
                {
                    // Decode
                    ok = channel->push(st.packet);

                    // Pull all frames from extractor to (*this).st.frame_queue
                    decoder_status suc;
                    frame frame;

                    while ((suc = channel->read(frame)) == DECODER_FRAME_AVAILABLE)
                        st.frame_queue.push(std::move(frame));
                }
            }

            if (!ok)
            {
                // Flush
                st.channel_video.push(nullptr);
                st.channel_audio.push(nullptr);

                // Pull remaining frames
                decoder_status suc;
                frame frame;

                while ((suc = st.channel_video.read(frame)) == DECODER_FRAME_AVAILABLE)
                    st.frame_queue.push(std::move(frame));   
                
                while ((suc = st.channel_audio.read(frame)) == DECODER_FRAME_AVAILABLE)
                    st.frame_queue.push(std::move(frame)); 
            }

            return !st.frame_queue.empty();
        }

        inline bool demuxer::read(frame& dst_frame)
        {
            if (!fill_queue())
                return false;

            if (!st.frame_queue.empty())
            {
                dst_frame = std::move(st.frame_queue.front());
                st.frame_queue.pop();
                return true;
            }

            return false;
        }

        inline bool            demuxer::video_enabled()         const noexcept { return st.channel_video.is_image_decoder(); }
        inline bool            demuxer::audio_enabled()         const noexcept { return st.channel_audio.is_audio_decoder(); }
        inline int             demuxer::height()                const noexcept { return st.channel_video.height(); }
        inline int             demuxer::width()                 const noexcept { return st.channel_video.width(); }
        inline AVPixelFormat   demuxer::pixel_fmt()             const noexcept { return st.channel_video.pixel_fmt(); }
        inline AVCodecID       demuxer::get_video_codec_id()    const noexcept { return st.channel_video.get_codec_id(); }
        inline std::string     demuxer::get_video_codec_name()  const noexcept { return st.channel_video.get_codec_name(); }

        inline int             demuxer::sample_rate()           const noexcept { return st.channel_audio.sample_rate(); }
        inline uint64_t        demuxer::channel_layout()        const noexcept { return st.channel_audio.channel_layout(); }
        inline AVSampleFormat  demuxer::sample_fmt()            const noexcept { return st.channel_audio.sample_fmt(); }
        inline int             demuxer::nchannels()             const noexcept { return st.channel_audio.nchannels(); }
        inline AVCodecID       demuxer::get_audio_codec_id()    const noexcept { return st.channel_audio.get_codec_id(); }
        inline std::string     demuxer::get_audio_codec_name()  const noexcept { return st.channel_audio.get_codec_name(); }

        inline float demuxer::fps() const noexcept
        {
            /*!
                Do we need to adjust _pFormatCtx->fps_probe_size ?
                Do we need to adjust _pFormatCtx->max_analyze_duration ?
            !*/
            if (st.channel_video.is_image_decoder() && st.pFormatCtx)
            {
                const float num = st.pFormatCtx->streams[st.stream_id_video]->avg_frame_rate.num;
                const float den = st.pFormatCtx->streams[st.stream_id_video]->avg_frame_rate.den;
                return num / den;
            }

            return 0.0f;
        }

        inline int demuxer::estimated_nframes() const noexcept
        {
            return st.channel_video.is_image_decoder() ? st.pFormatCtx->streams[st.stream_id_video]->nb_frames : 0;
        }

        inline int demuxer::estimated_total_samples() const noexcept
        {
            if (st.channel_audio.is_audio_decoder())
            {
                const AVRational src_time_base = st.pFormatCtx->streams[st.stream_id_audio]->time_base;
                const AVRational dst_time_base = {1, sample_rate()};
                return av_rescale_q(st.pFormatCtx->streams[st.stream_id_audio]->duration, src_time_base, dst_time_base);
            }
            return 0;
        }

        inline float demuxer::duration() const noexcept
        {
            return st.pFormatCtx ? (float)av_rescale_q(st.pFormatCtx->duration, {1, AV_TIME_BASE}, {1, 1000000}) * 1e-6 : 0.0f;
        }

        inline const std::unordered_map<std::string, std::string>& demuxer::get_metadata() const noexcept
        {
            return st.metadata;
        }

        inline float demuxer::get_rotation_angle() const noexcept
        {
            const auto it = st.metadata.find("rotate");
            return it != st.metadata.end() ? std::stof(it->second) : 0;
        }

// ---------------------------------------------------------------------------------------------------

        template <typename image_type>
        std::enable_if_t<is_image_type<image_type>::value, void>
        load_frame(image_type& image, const std::string& file_name)
        {
            demuxer reader(file_name);
            frame f;

            if (!reader.is_open() || !reader.read(f) || !f.is_image())
                throw error("ffmpeg::load_frame: error while loading " + file_name);

            convert(f, image);
        }

    }
}

#endif //DLIB_FFMPEG_DEMUXER
