// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_FFMPEG_DEMUXER
#define DLIB_FFMPEG_DEMUXER

#include <queue>
#include <functional>
#include <unordered_map>
#include "ffmpeg_utils.h"
#include "../functional.h"
#include "../constexpr_if.h"

namespace dlib
{
    namespace ffmpeg
    {

// ---------------------------------------------------------------------------------------------------

        template <
          class Callback,
          std::enable_if_t<dlib::is_invocable<Callback, frame&>::value, bool> = true
        >
        auto wrap (
            Callback&&              clb,
            const resizing_args&    args_image = resizing_args{},
            const resampling_args&  args_audio = resampling_args{}
        );
        /*!
            requires
                - Callback is a callback type with signature void(frame&)
            ensures
                - returns a new conforming callback which can be passed to
                    decoder::push()
                    decoder::flush()
                - The callback returned by wrap() captures `clb`, `args_image` and `args_audio` by VALUE
                - When a new frame object `f` is decoded it is optionally resized or resampled then 
                  the callback `clb` is invoked by passing `f` by reference like so: clb(f)
                - It is safe for the callback `clb` to swap or move `f` when passed by reference in clb(f).
                - Resizing is performed if:
                    - the decoded frame `f` satisfies f.is_image() == true
                    - args_image is non-empty
                    - args_image.h      != f.height() or 
                      args_image.w      != f.width() or 
                      args_image.fmt    != f.pixfmt() 
                - Resampling is performed if:
                    - the decoded frame `f` satisfies f.is_audio() == true
                    - args_audio is non-empty
                    - args_audio.sample_rate    != f.sample_rate() or 
                      args_audio.channel_layout != f.layout() or 
                      args_audio.fmt            != f.samplefmt()
        !*/

// ---------------------------------------------------------------------------------------------------

        template <
          class Callback,
          std::enable_if_t<is_callable<Callback>::value, bool> = true,
          std::enable_if_t<callable_nargs<Callback>::value == 1, bool> = true,
          std::enable_if_t<is_image_type<std::remove_reference_t<callable_arg<0, Callback>>>::value, bool> = true
        >
        auto wrap (
            Callback&& clb
        );
        /*!
            requires
                - Callback is a callback type with signature void(image_type&)
                  where image_type is an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h
            ensures
                - returns a new conforming callback which can be passed to
                    decoder::push()
                    decoder::flush()
                - The callback returned by wrap() captures `clb` by VALUE.
                - When a new frame object is decoded, it is converted to an image_type object `img`,
                  then the callback `clb` is invoked like so: `clb(img)`
        !*/

// ---------------------------------------------------------------------------------------------------

        auto wrap (
            std::queue<frame>& queue,
            const resizing_args& args_image = resizing_args{},
            const resampling_args& args_audio = resampling_args{}
        );
        /*!
            ensures
                - returns a new conforming callback which can be passed to
                    decoder::push()
                    decoder::flush()
                - The callback returned by wrap() captures `queue` by REFERENCE, 
                  both `args_image` and `args_audio` by VALUE.
                - When a new frame object `f` is decoded, it is optionally resized or resampled,
                  then added to `queue`.
                - Resizing is performed if:
                    - the decoded frame `f` satisfies f.is_image() == true
                    - args_image is non-empty
                    - args_image.h != f.height() or 
                      args_image.w != f.width() or 
                      args_image.fmt != f.pixfmt() 
                - Resampling is performed if:
                    - the decoded frame `f` satisfied f.is_audio() == true
                    - args_audio is non-empty
                    - args_audio.sample_rate != f.sample_rate() or 
                      args_audio.channel_layout != f.layout() or 
                      args_audio.fmt != f.samplefmt()
        !*/

// ---------------------------------------------------------------------------------------------------

        template<class image_type>
        auto wrap (
            std::queue<image_type>& queue
        );
        /*!
            requires
                - image_type is an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h
            ensures
                - returns a new conforming callback which can be passed to
                    decoder::push()
                    decoder::flush()
                - The callback returned by wrap() captures `queue` by REFERENCE.
                - When a new frame object is decoded, it is converted to an image_type object `img`,
                  then added to `queue`
        !*/

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

            using args = decoder_codec_args;
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This holds constructor arguments for decoder.
            !*/

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
                    - returns height of encoded images
            !*/

            int width() const noexcept;
            /*!
                requires
                    - is_image_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns width of encoded images
            !*/

            AVPixelFormat pixel_fmt() const noexcept;
            /*!
                requires
                    - is_image_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns pixel format of encoded images
            !*/

            /*! audio properties !*/

            int sample_rate() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns sample rate of encoded audio frames
            !*/

            uint64_t channel_layout() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns channel layout of encoded audio frames
            !*/

            AVSampleFormat sample_fmt() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns sample format of encoded audio frames
            !*/

            int nchannels() const noexcept;
            /*!
                requires
                    - is_audio_decoder() == true
                    - must have called push_encoded() enough times such that a call to read() would have returned a frame.
                ensures
                    - returns the number of channels (e.g. 2 if stereo, 1 if mono) of encoded audio frames
            !*/

            template <class Callback> 
            bool push (
                const uint8_t*  encoded, 
                int             nencoded,
                Callback&&      clb
            );
            /*!
                requires
                    - is_open() == true
                    - clb is a valid callback created using one of the dlib::ffmpeg::wrap() global functions
                    - clb must not call decoder::push() or decoder::flush(), i.e. the callback does not exhibit recursion.
                ensures
                    - encodes data
                    - clb may get invoked with new frames
            !*/

            template <class Callback>
            void flush(Callback&& clb);
            /*!
                requires
                    - is_open() == true
                    - clb is a valid callback created using one of the dlib::ffmpeg::wrap() global functions
                    - clb must not call decoder::push() or decoder::flush(), i.e. the callback does not exhibit recursion.
                ensures
                    - Flushes the decoder. This must be called when there is no more data to be decoded. Last remaining frames will be available.
                    - calls push_encoded(nullptr, 0)
            !*/

        private:
            friend class demuxer;

            bool open (
                const args&                     a,
                details::av_ptr<AVCodecContext> pCodecCtx_,
                const AVCodec*                  codec,
                AVRational                      timebase_
            );

            template <class Callback> bool push_padded(const uint8_t *encoded, int nencoded, Callback&& clb);
            template <class Callback> bool push(const details::av_ptr<AVPacket>& pkt, Callback&& clb);

            args                                    args_;
            bool                                    open_{false};
            bool                                    decoding{false};
            AVRational                              timebase;
            details::av_ptr<AVCodecParserContext>   parser;
            details::av_ptr<AVCodecContext>         pCodecCtx;
            details::av_ptr<AVPacket>               packet;
            frame                                   avframe;
            details::resizer                        resizer_image;
            details::resampler                      resizer_audio;
            std::vector<uint8_t>                    encoded_buffer;
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

                // Video stream codec arguments
                decoder_codec_args args_codec_image;

                // Audio stream codec arguments
                decoder_codec_args args_codec_audio;
                
                // Whether or not to decode video stream.
                bool enable_image{true};

                // Whether or not to decode audio stream.
                bool enable_audio{true};

                // Sets the output frame rate for any device that allows you to do so, e.g. webcam, x11grab, etc. Does not apply to files. If -1, ignored.
                int framerate{-1};

                // Sets output height for any device that allows you to do so, e.g. webcam, x11grab, etc. Dot not apply to files. If -1, ignored.
                int height{-1};

                // Sets output width for any device that allows you to do so, e.g. webcam, x11grab, etc. Dot not apply to files. If -1, ignored.
                int width{-1};
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
                        - returns height of encoded images
                    - else
                        - returns 0
            !*/

            int width() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns width of encoded images
                    - else
                        - returns 0
            !*/

            AVPixelFormat pixel_fmt() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns pixel format of encoded images
                    - else
                        - returns AV_PIX_FMT_NONE
            !*/

            float fps() const noexcept;
            /*!
                ensures 
                    - if (video_enabled())
                        - returns frame rate (frames per second) of video
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
                        - returns sample rate of encoded audio frames
                    - else
                        - returns 0
            !*/

            uint64_t channel_layout() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns channel layout of encoded audio frames (e.g. AV_CH_LAYOUT_STEREO)
                    - else
                        - returns 0
            !*/

            AVSampleFormat sample_fmt() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns sample format of encoded audio frames (e.g. AV_SAMPLE_FMT_S16)
                    - else
                        - returns AV_SAMPLE_FMT_NONE
            !*/

            int nchannels() const noexcept;
            /*!
                ensures 
                    - if (audio_enabled())
                        - returns the number of channels in the encoded audio frames (e.g. 1 for mono, 2 for stereo)
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

            bool read (
                frame& frame,
                const resizing_args&   args_image = resizing_args{}, 
                const resampling_args& args_audio = resampling_args{}
            );
            /*!
                ensures 
                    - if (is_open())
                        - returns true
                        - frame.is_empty() == false
                        - optionally converts frame using args_image if frame.is_image() == true, or args_audio if frame.is_audio() == true
                    - else
                        - returns false and frame.is_empty() == true
            !*/

            template <
              class image_type,
              is_image_check<image_type> = true
            >
            bool read (
                image_type& img
            );
            /*!
                requires
                    - image_type == an image object that implements the interface defined in
                      dlib/image_processing/generic_image.h

                ensures 
                    - keeps reading the file until one of the following is true:
                        - is_open() == false, in which case return false;
                        - an image is found, in which case it is converted to image_type appropriately and returns true
                    - If num_rows(img) > 0 or num_cols(img) > 0 :
                        then the frame is resized to fit those dimensions.
                    - If num_rows(img) == 0 and num_cols(img) == 0:
                        then the frame is copied to "img"
            !*/

        private:
            bool open(const args& a);
            bool object_alive() const noexcept;
            bool interrupt_callback();
            void populate_metadata();
            bool fill_queue();

            struct {
                args                                    args_;
                details::av_ptr<AVFormatContext>        pFormatCtx;
                details::av_ptr<AVPacket>               packet;
                decoder                                 channel_video;
                decoder                                 channel_audio;
                int                                     stream_id_video{-1};
                int                                     stream_id_audio{-1};
                std::chrono::system_clock::time_point   connecting_time{};
                std::chrono::system_clock::time_point   connected_time{};
                std::chrono::system_clock::time_point   last_read_time{};
                std::unordered_map<std::string, std::string> metadata;
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

        namespace details
        {
            inline void convert (
                const frame&            src_frame,
                frame&                  dst_frame,
                resizer&                resizer,
                resampler&              resampler,
                const resizing_args&    args_image, 
                const resampling_args&  args_audio
            )
            {
                if (src_frame.is_image())
                {
                    resizer.resize(
                        src_frame,
                        args_image.h > 0                  ? args_image.h :   src_frame.height(),
                        args_image.w > 0                  ? args_image.w :   src_frame.width(),
                        args_image.fmt != AV_PIX_FMT_NONE ? args_image.fmt : src_frame.pixfmt(),
                        dst_frame);
                }
                else
                {
                    resampler.resize(
                        src_frame,
                        args_audio.sample_rate > 0            ? args_audio.sample_rate      : src_frame.sample_rate(),
                        args_audio.channel_layout > 0         ? args_audio.channel_layout   : src_frame.layout(),
                        args_audio.fmt != AV_SAMPLE_FMT_NONE  ? args_audio.fmt              : src_frame.samplefmt(),
                        dst_frame);
                }
            }
        }

// ---------------------------------------------------------------------------------------------------

        template <
          class Callback,
          std::enable_if_t<dlib::is_invocable<Callback, frame&>::value, bool>
        >
        inline auto wrap (
            Callback&&              clb,
            const resizing_args&    args_image,
            const resampling_args&  args_audio
        )
        {
            using namespace details;

            return [=, pclb = std::forward<Callback>(clb)] (
                frame&      f, 
                resizer&    resizer, 
                resampler&  resampler
            ) mutable
            {
                convert(f, f, resizer, resampler, args_image, args_audio);
                pclb(f);
            };
        }

// ---------------------------------------------------------------------------------------------------

        template <
          class Callback,
          std::enable_if_t<is_callable<Callback>::value, bool>,
          std::enable_if_t<callable_nargs<Callback>::value == 1, bool>,
          std::enable_if_t<is_image_type<std::remove_reference_t<callable_arg<0, Callback>>>::value, bool>
        >
        inline auto wrap (
            Callback&& clb
        )
        {
            using namespace details;
            using image_type = std::remove_reference_t<callable_arg<0, Callback>>;

            return [img = image_type{}, pclb = std::forward<Callback>(clb)] (
                frame&      f, 
                resizer&    resizer, 
                resampler&  resampler
            ) mutable
            {
                if (f.is_image())
                {
                    convert (
                        f, f,
                        resizer,
                        resampler,
                        {0,0,pix_traits<pixel_type_t<image_type>>::fmt},
                        {}
                    );

                    convert(f, img);
                    pclb(img);
                } 
            };
        }

// ---------------------------------------------------------------------------------------------------

        inline auto wrap (
            std::queue<frame>& queue,
            const resizing_args& args_image,
            const resampling_args& args_audio
        )
        {
            using namespace details;

            return [&queue, args_image, args_audio] (
                frame&      f, 
                resizer&    resizer, 
                resampler&  resampler
            ) mutable
            {
                convert(f, f, resizer, resampler, args_image, args_audio);
                queue.push(std::move(f));
            };
        }

// ---------------------------------------------------------------------------------------------------

        template<class image_type>
        inline auto wrap (
            std::queue<image_type>& queue
        )
        {
            using namespace details;

            return [&queue] (
                frame&      f, 
                resizer&    resizer, 
                resampler&  resampler
            ) mutable
            {
                if (f.is_image())
                {
                    convert (
                        f, f,
                        resizer,
                        resampler,
                        {0,0,pix_traits<pixel_type_t<image_type>>::fmt},
                        {}
                    );

                    image_type img;
                    convert(f, img);
                    queue.push(std::move(img));
                } 
            };
        }

// ---------------------------------------------------------------------------------------------------

        inline decoder::decoder(const args &a)
        {
            using namespace details;

            DLIB_ASSERT(a.codec != AV_CODEC_ID_NONE || a.codec_name != "", "At least args_codec.codec or args_codec.codec_name must be set");
            
            register_ffmpeg();
            
            const AVCodec* pCodec = nullptr;

            if (a.codec != AV_CODEC_ID_NONE)
                pCodec = avcodec_find_decoder(a.codec);
            else if (!a.codec_name.empty())
                pCodec = avcodec_find_decoder_by_name(a.codec_name.c_str());

            if (!pCodec)
            {
                logger_dlib_wrapper() << LERROR 
                    << "Codec "
                    << avcodec_get_name(a.codec)
                    << " / "
                    << a.codec_name
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

            if (!open(a, std::move(pCodecCtx_), pCodec,  timebase))
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

            args_       = a;
            timebase    = timebase_;
            packet      = make_avpacket();

            if (args_.bitrate > 0)
                pCodecCtx_->bit_rate = args_.bitrate;
            if (args_.flags > 0)
                pCodecCtx_->flags |= args_.flags;
            if (args_.thread_count > 0)
                pCodecCtx_->thread_count = args_.thread_count;

            av_dict opt = args_.codec_options;
            int ret = avcodec_open2(pCodecCtx_.get(), codec, opt.get());

            if (ret < 0)
                return fail("avcodec_open2() failed : ", get_av_error(ret));
            
            pCodecCtx = std::move(pCodecCtx_);
            open_ = true;
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

        template<class Callback>
        inline bool decoder::push (
            const details::av_ptr<AVPacket>& pkt,
            Callback&& clb
        )
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
                    open_ = false;
                    state = EXTRACT_DONE;
                } else {
                    open_ = false;
                    state = EXTRACT_ERROR;
                    logger_dlib_wrapper() << LERROR << "avcodec_send_packet() failed : " << get_av_error(ret);
                }
            };

            const auto recv_frame = [&](extract_state& state, bool resend)
            {
                if (!avframe.f)
                    avframe.f = make_avframe();
                const int ret = avcodec_receive_frame(pCodecCtx.get(), avframe.f.get());

                if (ret == AVERROR(EAGAIN) && resend)
                    state   = EXTRACT_SEND_PACKET;
                else if (ret == AVERROR(EAGAIN))
                    state   = EXTRACT_DONE;
                else if (ret == AVERROR_EOF) {
                    open_ = false;
                    state = EXTRACT_DONE;
                }
                else if (ret < 0)
                {
                    open_ = false;
                    state = EXTRACT_ERROR;
                    logger_dlib_wrapper() << LERROR << "avcodec_receive_frame() failed : " << get_av_error(ret);
                }
                else
                {
                    const AVRational tb         = avframe.is_image() ? timebase : AVRational{1, avframe.sample_rate()};
                    const uint64_t pts          = avframe.is_image() ? avframe.f->pts : next_pts;
                    avframe.timestamp           = system_clock::time_point{duration_cast<system_clock::duration>(nanoseconds{av_rescale_q(pts, tb, {1,1000000000})})};
                    next_pts                    += avframe.is_image() ? 1 : avframe.f->nb_samples;
                    avframe.f->pict_type = AV_PICTURE_TYPE_NONE;
                    std::forward<Callback>(clb)(avframe, resizer_image, resizer_audio);
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

        template <class Callback>
        inline bool decoder::push_padded (
            const uint8_t*  encoded, 
            int             nencoded,
            Callback&&      clb
        )
        {
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
                    ok = push(packet, std::forward<Callback>(clb));
                
                // If flushing, you keep parsing until you get an empty packet
                if (packet->size == 0 && flushing)
                    break;
            }

            if (flushing)
            {
                // Flush codec. After this, pCodecCtx == nullptr since AVERROR_EOF will be returned at some point.
                ok = push(nullptr, std::forward<Callback>(clb));
            }
        
            return ok;
        }

        template <class Callback>
        inline bool decoder::push (
            const uint8_t*  encoded, 
            int             nencoded,
            Callback&&      clb
        )
        {
            DLIB_ASSERT(!decoding, "Recursion in push() no supported");
            decoding = true;

            bool ok = true;

            if (encoded == nullptr && nencoded == 0)
            {
                ok = push_padded(nullptr, 0, std::forward<Callback>(clb));
            }
            else
            {
                if (nencoded > AV_INPUT_BUFFER_PADDING_SIZE)
                {
                    const int blocksize = nencoded - AV_INPUT_BUFFER_PADDING_SIZE;

                    ok = push_padded(encoded, blocksize, std::forward<Callback>(clb));
                    encoded  += blocksize;
                    nencoded -= blocksize; // == AV_INPUT_BUFFER_PADDING_SIZE
                }

                if (ok)
                {
                    encoded_buffer.resize(nencoded + AV_INPUT_BUFFER_PADDING_SIZE);
                    std::memcpy(encoded_buffer.data(), encoded, nencoded);
                    ok = push_padded(encoded_buffer.data(), nencoded, std::forward<Callback>(clb));
                }
            }

            decoding = false;
            return ok;
        }

        template<class Callback>
        inline void decoder::flush(Callback&& clb)
        {
            push(nullptr, 0, std::forward<Callback>(clb));
        }

        inline bool             decoder::is_open()          const noexcept { return pCodecCtx && open_; }
        inline bool             decoder::is_image_decoder() const noexcept { return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO; }
        inline bool             decoder::is_audio_decoder() const noexcept { return pCodecCtx && pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO; }
        inline AVCodecID        decoder::get_codec_id()     const noexcept { return pCodecCtx ? pCodecCtx->codec_id : AV_CODEC_ID_NONE; }
        inline std::string      decoder::get_codec_name()   const noexcept { return pCodecCtx ? avcodec_get_name(pCodecCtx->codec_id) : "NONE"; }
        inline int              decoder::height()           const noexcept { return pCodecCtx ? pCodecCtx->height       : 0; }
        inline int              decoder::width()            const noexcept { return pCodecCtx ? pCodecCtx->width        : 0; }
        inline AVPixelFormat    decoder::pixel_fmt()        const noexcept { return pCodecCtx ? pCodecCtx->pix_fmt      : AV_PIX_FMT_NONE; }
        inline int              decoder::sample_rate()      const noexcept { return pCodecCtx ? pCodecCtx->sample_rate  : 0; }
        inline AVSampleFormat   decoder::sample_fmt()       const noexcept { return pCodecCtx ? pCodecCtx->sample_fmt   : AV_SAMPLE_FMT_NONE; }
        inline uint64_t         decoder::channel_layout()   const noexcept { return details::get_layout(pCodecCtx.get()); }
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
            if (st.args_.height > 0 && 
                st.args_.width > 0 && 
                st.args_.format_options.find("video_size") == st.args_.format_options.end())
            {
                // See if format supports "video_size"
                st.args_.format_options["video_size"] = std::to_string(st.args_.width) + "x" + std::to_string(st.args_.height);
            }

            if (st.args_.framerate > 0 &&
                st.args_.format_options.find("framerate") == st.args_.format_options.end())
            {
                // See if format supports "framerate"
                st.args_.format_options["framerate"] = std::to_string(st.args_.framerate);
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
                    st.channel_video.open(
                        st.args_.args_codec_image,
                        std::move(pCodecCtx), 
                        pCodec,
                        st.pFormatCtx->streams[stream_id]->time_base
                    );

                    st.stream_id_video = stream_id;
                }
                else
                {
                    st.channel_audio.open(
                        st.args_.args_codec_audio,
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

                if (ok && channel && st.packet->size > 0)
                {
                    // Decode
                    ok = channel->push(st.packet, wrap(st.frame_queue));
                }
            }

            if (!ok)
            {
                // Flush
                st.channel_video.push(nullptr, wrap(st.frame_queue));
                st.channel_audio.push(nullptr, wrap(st.frame_queue));
            }

            return !st.frame_queue.empty();
        }

        inline bool demuxer::read (
            frame&                  dst_frame,
            const resizing_args&    args_image, 
            const resampling_args&  args_audio
        )
        {
            using namespace details;

            if (!fill_queue())
                return false;

            dst_frame = std::move(st.frame_queue.front());
            st.frame_queue.pop();

            convert (
                dst_frame,
                dst_frame,
                st.channel_video.resizer_image,
                st.channel_audio.resizer_audio,
                args_image,
                args_audio
            );

            return true;
        }

        template <
          class image_type,
          is_image_check<image_type>
        >
        inline bool demuxer::read (
            image_type& img
        )
        {
            using namespace details;

            while (fill_queue())
            {
                frame f = std::move(st.frame_queue.front());
                st.frame_queue.pop();

                if (f.is_image())
                {
                    convert (
                        f,
                        f,
                        st.channel_video.resizer_image,
                        st.channel_audio.resizer_audio,
                        {img.nr() > 0 ? (int)img.nr() : f.height(),
                         img.nc() > 0 ? (int)img.nc() : f.width(),
                         pix_traits<pixel_type_t<image_type>>::fmt},
                        {}
                    );

                    convert(f, img);
                    
                    return true;
                } 
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

        template <
          class image_type,
          is_image_check<image_type>
        >
        inline void load_frame(image_type& image, const std::string& file_name)
        {
            if (!demuxer({file_name, video_enabled, audio_disabled}).read(image))
                throw error(EIMAGE_LOAD, "ffmpeg::load_frame: error while loading " + file_name);
        }

// ---------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_FFMPEG_DEMUXER
