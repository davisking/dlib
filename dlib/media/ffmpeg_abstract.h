// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#undef DLIB_FFMPEG_ABSTRACT
#ifdef DLIB_FFMPEG_ABSTRACT

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
                std::chrono::system_clock::time_point   timestamp
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

        private:
            // Implementation details
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

        template<> struct pix_traits<uint8_t>           {constexpr static AVPixelFormat fmt = AV_PIX_FMT_GRAY8; };
        template<> struct pix_traits<rgb_pixel>         {constexpr static AVPixelFormat fmt = AV_PIX_FMT_RGB24; };
        template<> struct pix_traits<bgr_pixel>         {constexpr static AVPixelFormat fmt = AV_PIX_FMT_BGR24; };
        template<> struct pix_traits<rgb_alpha_pixel>   {constexpr static AVPixelFormat fmt = AV_PIX_FMT_RGBA;  };
        template<> struct pix_traits<bgr_alpha_pixel>   {constexpr static AVPixelFormat fmt = AV_PIX_FMT_BGRA;  };

// ---------------------------------------------------------------------------------------------------

        template<class SampleType>
        struct sample_traits 
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a type trait for converting a sample type to ffmpeg's AVSampleFormat obj.
            !*/
        };

        template<> struct sample_traits<uint8_t> {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_U8; };
        template<> struct sample_traits<int16_t> {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_S16; };
        template<> struct sample_traits<int32_t> {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_S32; };
        template<> struct sample_traits<float>   {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_FLT; };
        template<> struct sample_traits<double>  {constexpr static AVSampleFormat fmt = AV_SAMPLE_FMT_DBL; };

// ---------------------------------------------------------------------------------------------------

        template<class SampleType, std::size_t Channels>
        struct audio
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is a typed audio buffer which can convert to and back dlib::ffmpeg::frame.
            !*/
            using sample = std::array<SampleType, Channels>;

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
            AVCodecID   codec_id;
            std::string codec_name;
            bool supports_encoding;
            bool supports_decoding;
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
                    This object informs on available devices provided by the installation of ffmpeg dlib is linked against.
            !*/

            struct instance
            {
                std::string name;
                std::string description;
            };

            std::string device_type;
            std::vector<instance> devices;
        };
        
        std::vector<std::string> list_protocols();
        /*!
            ensures
                - returns a list of all registered ffmpeg protocols
        !*/

        std::vector<std::string> list_demuxers();
        /*!
            ensures
                - returns a list of all registered ffmpeg demuxers
        !*/

        std::vector<muxer_details> list_muxers();
        /*!
            ensures
                - returns a list of all registered ffmpeg muxers
        !*/
        
        std::vector<codec_details> list_codecs();
        /*!
            ensures
                - returns a list of all registered ffmpeg codecs with information on whether decoding and/or encoding is supported.
                  Note that not all codecs support encoding, unless your installation of ffmpeg is built with third party library
                  dependencies like libx264, libx265, etc.
        !*/

        std::vector<device_details> list_input_devices();
        /*!
            ensures
                - returns a list of all registered ffmpeg input devices and available instances of those devices
        !*/

        std::vector<device_details> list_output_devices();
        /*!
            ensures
                - returns a list of all registered ffmpeg output devices and available instances of those devices
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

            constexpr explicit video_enabled_t(bool enabled_);
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

        template <class image_type>
        void convert(const frame& f, image_type& image)
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - f.is_image() == true
                - f.pixfmt() == pix_traits<pixel_type_t<image_type>>::fmt
            ensures
                - converts a frame object into array2d<rgb_pixel>
        !*/

        template <class image_type>
        void convert(const image_type& img, frame& f)
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

            bool read(frame& frame);
            /*!
                ensures 
                    - if (is_open())
                        - returns true and frame.is_empty() == false
                    - else
                        - returns false and frame.is_empty() == true
            !*/

            /*! metadata! */
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
        };

// ---------------------------------------------------------------------------------------------------

        template <typename image_type>
        std::enable_if_t<is_image_type<image_type>::value, void>
        load_frame(
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

// ---------------------------------------------------------------------------------------------------

        template <
          class Byte, 
          class Allocator,
          std::enable_if_t<is_byte<Byte>::value, bool> = true
        >
        auto sink(std::vector<Byte, Allocator>& buf);
        /*!
            requires
                - Byte must be a byte type, e.g. char, int8_t or uint8_t
            ensures
                - returns a function object with signature bool(std::size_t N, const char* data).  When
                  called that function appends the first N bytes pointed to by data onto the end of buf.
                - The returned function is valid only as long as buf exists.
                - The function always returns true.        
        !*/

// ---------------------------------------------------------------------------------------------------

        auto sink(std::ostream& out);
        /*!
            ensures
                - returns a function object with signature bool(std::size_t N, const char* data).  When
                  called that function writes the first N bytes pointed to by data to out.
                - The returned view is valid only as long as out exists.
                - Returns out.good(). I.e. returns true if the write to the stream succeeded and false otherwise.       
        !*/

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
        };

// ---------------------------------------------------------------------------------------------------

    }
}

#endif //DLIB_FFMPEG_ABSTRACT
