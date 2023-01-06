#include <stdexcept>
#include <cassert>
#include <algorithm>
#include "ffmpeg_utils.h"
#include "../assert.h"

namespace dlib
{
    namespace details
    {
        bool ffmpeg_initialize_all()
        {
            avdevice_register_all();
            return true;
        }

        const bool FFMPEG_INITIALIZED = ffmpeg_initialize_all();

        std::string get_av_error(int ret)
        {
            char buf[128] = {0};
            int suc = av_strerror(ret, buf, sizeof(buf));
            return suc == 0 ? buf : "couldn't set error";
        }

        std::string get_pixel_fmt_str(AVPixelFormat fmt)
        {
            const char* name = av_get_pix_fmt_name(fmt);
            return name ? std::string(name) : std::string("unknown");
        }

        std::string get_audio_fmt_str(AVSampleFormat fmt)
        {
            const char* name = av_get_sample_fmt_name(fmt);
            return name ? std::string(name) : std::string("unknown");
        }

        std::string get_channel_layout_str(uint64_t layout)
        {
            std::string buf(32, '\0');
            av_get_channel_layout_string(&buf[0], buf.size(), 0, layout);
            return buf;
        }

        float rational::get() { return float(num) / denom; }

        void av_deleter::operator()(AVFrame *ptr)               const { if (ptr) av_frame_free(&ptr); }
        void av_deleter::operator()(AVPacket *ptr)              const { if (ptr) av_packet_free(&ptr); }
        void av_deleter::operator()(AVAudioFifo *ptr)           const { if (ptr) av_audio_fifo_free(ptr); }
        void av_deleter::operator()(SwsContext *ptr)            const { if (ptr) sws_freeContext(ptr); }
        void av_deleter::operator()(SwrContext *ptr)            const { if (ptr) swr_free(&ptr); }
        void av_deleter::operator()(AVCodecContext *ptr)        const { if (ptr) avcodec_free_context(&ptr); }
        void av_deleter::operator()(AVCodecParserContext *ptr)  const { if (ptr) av_parser_close(ptr); }
        void av_deleter::operator()(AVFormatContext *ptr)       const { if (ptr) avformat_close_input(&ptr); }

        av_dict::av_dict(const std::unordered_map<std::string, std::string>& options)
        {
            int ret = 0;

            for (const auto& opt : options) {
                if ((ret = av_dict_set(&avdic, opt.first.c_str(), opt.second.c_str(), 0)) < 0) {
                    printf("av_dict_set() failed : %s\n", get_av_error(ret).c_str());
                    break;
                }
            }
        }

        av_dict::av_dict(const av_dict& ori)
        {
            av_dict_copy(&avdic, ori.avdic, 0);
        }

        av_dict& av_dict::operator=(const av_dict& ori)
        {
            *this = std::move(av_dict{ori});
            return *this;
        }

        av_dict::av_dict(av_dict &&ori) noexcept
        : avdic{std::exchange(ori.avdic, nullptr)}
        {
        }

        av_dict &av_dict::operator=(av_dict &&ori) noexcept
        {
            if (this != &ori)
                avdic = std::exchange(ori.avdic, nullptr);
            return *this;
        }

        av_dict::~av_dict()
        {
            if (avdic)
                av_dict_free(&avdic);
        }

        AVDictionary** av_dict::get()
        {
            return avdic ? &avdic: nullptr;
        }

        av_ptr<AVFrame> make_avframe()
        {
            av_ptr<AVFrame> obj(av_frame_alloc());
            if (!obj)
                throw std::runtime_error("Failed to allocate AVFrame");
            return obj;
        }

        av_ptr<AVPacket> make_avpacket()
        {
            av_ptr<AVPacket> obj(av_packet_alloc());
            if (!obj)
                throw std::runtime_error("Failed to allocate AVPacket");
            return obj;
        }

    }

    Frame Frame::make(
        int             h,
        int             w,
        AVPixelFormat   pixfmt,
        int             sample_rate,
        int             nb_samples,
        uint64_t        channel_layout,
        AVSampleFormat  samplefmt,
        std::chrono::system_clock::time_point timestamp
    )
    {
        using namespace details;

        Frame obj;
        obj.frame = make_avframe();
        obj.frame->height           = h;
        obj.frame->width            = w;
        obj.frame->sample_rate      = sample_rate;
        obj.frame->channel_layout   = channel_layout;
        obj.frame->nb_samples       = nb_samples;
        obj.frame->format           = h > 0 && w > 0 ? (int)pixfmt : (int)samplefmt;
        obj.timestamp               = timestamp;

        int ret = av_frame_get_buffer(obj.frame.get(), 0); //use default alignment, which is likely 32
        if (ret < 0)
        {
            obj.frame = nullptr;
            throw std::runtime_error("av_frame_get_buffer() failed : " + get_av_error(ret));
        }

//        ret = av_frame_make_writable(obj.frame.get());
//        if (ret < 0)
//        {
//            obj.frame.reset(nullptr);
//            throw std::runtime_error("av_frame_make_writable() failed : " + get_av_error(ret));
//        }

        if (obj.is_audio())
            obj.frame->pts = av_rescale_q(obj.timestamp.time_since_epoch().count(),
                                        {decltype(obj.timestamp)::period::num, (decltype(obj.timestamp)::period::den)},
                                        {1, obj.frame->sample_rate});

        return obj;
    }

    Frame Frame::make_image(
        int h,
        int w,
        AVPixelFormat fmt,
        std::chrono::system_clock::time_point timestamp_us
    )
    {
        return make(h, w, fmt, 0, 0, 0, AV_SAMPLE_FMT_NONE, timestamp_us);
    }

    Frame Frame::make_audio(
        int             sample_rate,
        int             nb_samples,
        uint64_t        channel_layout,
        AVSampleFormat  fmt,
        std::chrono::system_clock::time_point timestamp_us
    )
    {
        return make(0,0,AV_PIX_FMT_NONE, sample_rate, nb_samples, channel_layout, fmt, timestamp_us);
    }

    Frame::Frame(const Frame &ori)
    {
        copy_from(ori);
    }

    Frame& Frame::operator=(const Frame& ori)
    {
        copy_from(ori);
        return *this;
    }

    void Frame::copy_from(const Frame &ori)
    {
        if (ori.is_empty())
        {
            Frame empty{std::move(*this)};
        }
        else
        {
            const bool same_image =
                    is_image() && ori.is_image() &&
                    std::tie(    frame->height,     frame->width,     frame->format) ==
                    std::tie(ori.frame->height, ori.frame->width, ori.frame->format);

            const bool same_audio =
                    is_audio() && ori.is_audio() &&
                    std::tie(    frame->sample_rate,     frame->nb_samples,     frame->channel_layout,     frame->format) ==
                    std::tie(ori.frame->sample_rate, ori.frame->nb_samples, ori.frame->channel_layout, ori.frame->format);

            if (!same_image && !same_audio)
            {
                Frame tmp = make(ori.frame->height,
                                ori.frame->width,
                                (AVPixelFormat)ori.frame->format,
                                ori.frame->sample_rate,
                                ori.frame->nb_samples,
                                ori.frame->channel_layout,
                                (AVSampleFormat)ori.frame->format,
                                ori.timestamp);
                *this = std::move(tmp);
            }

            av_frame_copy(frame.get(), ori.frame.get());
            av_frame_copy_props(frame.get(), ori.frame.get());
        }
    }

    bool Frame::is_empty() const noexcept
    {
        return !is_image() && !is_audio();
    }

    bool Frame::is_image() const noexcept
    {
        return frame && frame->width > 0 && frame->height > 0 && frame->format >= 0;
    }

    bool Frame::is_audio() const noexcept
    {
        return frame && frame->nb_samples > 0 && frame->channel_layout > 0 && frame->sample_rate > 0;
    }

    AVPixelFormat Frame::pixfmt() const noexcept
    {
        return is_image() ? (AVPixelFormat)frame->format : AV_PIX_FMT_NONE;
    }

    int Frame::height() const noexcept
    {
        return is_image() ? frame->height : 0;
    }

    int Frame::width() const noexcept
    {
        return is_image() ? frame->width : 0;
    }

    AVSampleFormat Frame::samplefmt() const noexcept
    {
        return is_audio() ? (AVSampleFormat)frame->format : AV_SAMPLE_FMT_NONE;
    }

    int Frame::nsamples() const noexcept
    {
        return is_audio() ? frame->nb_samples : 0;
    }

    uint64_t Frame::layout() const noexcept
    {
        return frame->channel_layout;
    }

    int Frame::nchannels() const noexcept
    {
        return is_audio() ? av_get_channel_layout_nb_channels(frame->channel_layout) : 0;
    }

    int Frame::sample_rate() const noexcept
    {
        return is_audio() ? frame->sample_rate : 0;
    }

    std::chrono::system_clock::time_point Frame::get_timestamp() const noexcept
    {
        return timestamp;
    }

    const AVFrame& Frame::get_frame() const noexcept
    {
        return *frame;
    }

    namespace details
    {
        int sw_image_resizer::get_src_h() const {
            return _src_h;
        }

        int sw_image_resizer::get_src_w() const {
            return _src_w;
        }

        AVPixelFormat sw_image_resizer::get_src_fmt() const {
            return _src_fmt;
        }

        int sw_image_resizer::get_dst_h() const {
            return _dst_h;
        }

        int sw_image_resizer::get_dst_w() const {
            return _dst_w;
        }

        AVPixelFormat sw_image_resizer::get_dst_fmt() const {
            return _dst_fmt;
        }

        void sw_image_resizer::reset(
            int src_h, int src_w, AVPixelFormat src_fmt,
            int dst_h, int dst_w, AVPixelFormat dst_fmt)
        {
            auto this_params = std::tie(_src_h, _src_w, _src_fmt, _dst_h, _dst_w, _dst_fmt);
            auto new_params  = std::tie( src_h,  src_w,  src_fmt,  dst_h,  dst_w,  dst_fmt);

            if (this_params != new_params)
            {
                this_params = new_params;

                _imgConvertCtx.reset(nullptr);

                if (_dst_h != _src_h ||
                    _dst_w != _src_w ||
                    _dst_fmt != _src_fmt)
                {
                    _imgConvertCtx.reset(sws_getContext(_src_w, _src_h, _src_fmt,
                                                        _dst_w, _dst_h, _dst_fmt,
                                                        SWS_FAST_BILINEAR, NULL, NULL, NULL));
                }
            }
        }

        void sw_image_resizer::resize (
            const Frame& src,
            int dst_h, int dst_w, AVPixelFormat dst_pixfmt,
            Frame& dst
        )
        {
            assert(src.is_image());

            const bool is_same_object = std::addressof(src) == std::addressof(dst);

            reset(src.frame->height, src.frame->width, src.pixfmt(),
                dst_h, dst_w, dst_pixfmt);

            if (_imgConvertCtx)
            {
                Frame tmp;
                Frame* ptr = std::addressof(dst);

                if (is_same_object ||
                    !dst.is_image() ||
                    std::tie(dst.frame->height, dst.frame->width, dst.frame->format) !=
                    std::tie(_dst_h, _dst_w, _dst_fmt))
                {
                    tmp = Frame::make_image(_dst_h, _dst_w, _dst_fmt, src.timestamp);
                    ptr = std::addressof(tmp);
                }

                sws_scale(_imgConvertCtx.get(),
                        src.frame->data, src.frame->linesize, 0, src.frame->height,
                        ptr->frame->data, ptr->frame->linesize);

                if (ptr != std::addressof(dst))
                    dst = std::move(tmp);
            }
            else if (!is_same_object)
            {
                dst = src;
            }
        }

        void sw_image_resizer::resize(
            const Frame& src,
            Frame& dst
        )
        {
            resize(src, _dst_h, _dst_w, _dst_fmt, dst);
        }

        void sw_audio_resampler::reset(
            int src_sample_rate, uint64_t src_channel_layout, AVSampleFormat src_fmt,
            int dst_sample_rate, uint64_t dst_channel_layout, AVSampleFormat dst_fmt
        )
        {
            auto this_params = std::tie(src_sample_rate_,
                                        src_channel_layout_,
                                        src_fmt_,
                                        dst_sample_rate_,
                                        dst_channel_layout_,
                                        dst_fmt_);
            auto new_params  = std::tie(src_sample_rate,
                                        src_channel_layout,
                                        src_fmt,
                                        dst_sample_rate,
                                        dst_channel_layout,
                                        dst_fmt);

            if (this_params != new_params)
            {
                this_params = new_params;

                audioResamplerCtx_.reset(nullptr);

                if (src_sample_rate_    != dst_sample_rate_ ||
                    src_channel_layout_ != dst_channel_layout_ ||
                    src_fmt_            != dst_fmt_)
                {
                    audioResamplerCtx_.reset(swr_alloc_set_opts(NULL,
                                                            dst_channel_layout_, dst_fmt_, dst_sample_rate_,
                                                            src_channel_layout_, src_fmt_, src_sample_rate_,
                                                            0, NULL));
                    int ret = 0;
                    if ((ret = swr_init(audioResamplerCtx_.get())) < 0)
                    {
                        audioResamplerCtx_.reset(nullptr);
                        throw std::runtime_error("swr_init() failed : " + get_av_error(ret));
                    }
                }
            }
        }

        void sw_audio_resampler::resize(
            const Frame&    src,
            int             dst_sample_rate,
            uint64_t        dst_channel_layout,
            AVSampleFormat  dst_samplefmt,
            Frame&          dst
        )
        {
            using namespace std::chrono;
            assert(src.is_audio());

            const bool is_same_object = std::addressof(src) == std::addressof(dst);

            reset(src.frame->sample_rate, src.frame->channel_layout, (AVSampleFormat)src.frame->format,
                dst_sample_rate, dst_channel_layout, dst_samplefmt);

            if (audioResamplerCtx_)
            {
                Frame tmp;
                Frame* ptr = std::addressof(dst);

                const int64_t delay       = swr_get_delay(audioResamplerCtx_.get(), src_sample_rate_);
                const auto dst_nb_samples = av_rescale_rnd(delay + src.frame->nb_samples, dst_sample_rate_, src_sample_rate_, AV_ROUND_UP);

                if (is_same_object ||
                    !dst.is_audio() ||
                    std::tie(dst.frame->sample_rate, dst.frame->channel_layout, dst.frame->format, dst.frame->nb_samples) !=
                    std::tie(dst_sample_rate_, dst_channel_layout_, dst_fmt_, dst_nb_samples))
                {
                    tmp = Frame::make_audio(dst_sample_rate_, dst_nb_samples, dst_channel_layout_, dst_fmt_, std::chrono::system_clock::time_point{});
                    ptr = std::addressof(tmp);
                }

                int ret = swr_convert(audioResamplerCtx_.get(),
                                    ptr->frame->data, ptr->frame->nb_samples,
                                    (const uint8_t**)src.frame->data, src.frame->nb_samples);
                if (ret < 0)
                    throw std::runtime_error("swr_convert() failed : " + get_av_error(ret));

                ptr->frame->nb_samples = ret;
                ptr->frame->pts   = tracked_samples_;
                ptr->timestamp    = system_clock::time_point{nanoseconds{av_rescale_q(tracked_samples_,
                                                                                    {1, dst_sample_rate_},
                                                                                    {nanoseconds::period::num, nanoseconds::period::den})}};
                tracked_samples_ += ptr->frame->nb_samples;

                if (ptr!= std::addressof(dst))
                    dst = std::move(tmp);
            }
            else if (!is_same_object)
            {
                dst = src;
            }
        }

        void sw_audio_resampler::resize(
            const Frame& src,
            Frame& dst
        )
        {
            resize(src, dst_sample_rate_, dst_channel_layout_, dst_fmt_, dst);
        }

        int sw_audio_resampler::get_src_rate() const
        {
            return src_sample_rate_;
        }

        uint64_t sw_audio_resampler::get_src_layout() const
        {
            return src_channel_layout_;
        }

        AVSampleFormat sw_audio_resampler::get_src_fmt() const
        {
            return src_fmt_;
        }

        int sw_audio_resampler::get_dst_rate() const
        {
            return dst_sample_rate_;
        }

        uint64_t sw_audio_resampler::get_dst_layout() const
        {
            return dst_channel_layout_;
        }

        AVSampleFormat sw_audio_resampler::get_dst_fmt() const
        {
            return dst_fmt_;
        }

        sw_audio_fifo::sw_audio_fifo(
            int codec_frame_size,
            int sample_format,
            int nchannels
        ) : frame_size(codec_frame_size),
            fmt(sample_format),
            channels(nchannels)
        {
            if (frame_size > 0)
            {
                fifo.reset(av_audio_fifo_alloc((AVSampleFormat)fmt, channels, frame_size));
                if (!fifo)
                    throw std::runtime_error("av_audio_fifo_alloc() failed");
            }
        }

        std::vector<Frame> sw_audio_fifo::push_pull(
            Frame&& in
        )
        {
            using namespace std::chrono;
            assert(in.is_audio());

            std::vector<Frame> outs;

            //check that the configuration hasn't suddenly changed this would be exceptional
            const int nchannels = in.nchannels();
            auto current_params = std::tie(fmt, channels);
            auto new_params     = std::tie(in.frame->format, nchannels);

            if (current_params != new_params)
                throw std::runtime_error("new audio frame params differ from first ");

            if (frame_size == 0)
            {
                outs.push_back(std::move(in));
            }
            else
            {
                if (av_audio_fifo_write(fifo.get(), (void**)in.frame->data, in.frame->nb_samples) != in.frame->nb_samples)
                    throw std::runtime_error("av_audio_fifo_write() failed to write all samples");

                while (av_audio_fifo_size(fifo.get()) >= frame_size)
                {
                    const system_clock::time_point timestamp{nanoseconds{av_rescale_q(
                            sample_count,
                            {1, in.frame->sample_rate},
                            {nanoseconds::period::num, nanoseconds::period::den})}};

                    Frame out = Frame::make_audio(in.frame->sample_rate, frame_size, in.frame->channel_layout, in.samplefmt(), timestamp);

                    if (av_audio_fifo_read(fifo.get(), (void**)out.frame->data, out.frame->nb_samples) != out.frame->nb_samples)
                        throw std::runtime_error("av_audio_fifo_read() failed to read all requested samples");

                    sample_count += out.frame->nb_samples;
                    outs.push_back(std::move(out));
                }
            }

            return outs;
        }
    }

    std::vector<std::string> ffmpeg_list_protocols()
    {
        std::vector<std::string> protocols;
        void* opaque = NULL;
        const char* name = 0;
        while ((name = avio_enum_protocols(&opaque, 0)))
            protocols.emplace_back(name);

        opaque  = NULL;
        name    = 0;

        while ((name = avio_enum_protocols(&opaque, 1)))
            protocols.emplace_back(name);

        return protocols;
    }

    std::vector<std::string> ffmpeg_list_demuxers()
    {
        std::vector<std::string> demuxers;
        void* opaque = nullptr;
        const AVInputFormat* demuxer = NULL;
        while ((demuxer = av_demuxer_iterate(&opaque)))
            demuxers.push_back(demuxer->name);
        return demuxers;
    }

    std::vector<std::string> ffmpeg_list_muxers()
    {
        std::vector<std::string> muxers;
        void* opaque = nullptr;
        const AVOutputFormat* muxer = NULL;
        while ((muxer = av_muxer_iterate(&opaque)))
            muxers.push_back(muxer->name);
        return muxers;
    }

    std::vector<codec_details> ffmpeg_list_available_codecs()
    {
        std::vector<codec_details> details;
        const AVCodec* codec = NULL;

#if LIBAVCODEC_VERSION_MAJOR >= 58 && LIBAVCODEC_VERSION_MINOR >= 10 && LIBAVCODEC_VERSION_MICRO >= 100
        void* opaque = nullptr;
        while ((codec = av_codec_iterate(&opaque)))
#else
        while ((codec = av_codec_iterate(codec)))
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
        auto it = details.begin() + 1;
        while (it != details.end())
        {
            auto prev = it - 1;

            if (it->codec_name == prev->codec_name)
            {
                prev->supports_encoding |= it->supports_encoding;
                prev->supports_decoding |= it->supports_decoding;
                it = details.erase(it);
            }
            else
                it++;
        }
        return details;
    }

    type_safe_union<array2d<rgb_pixel>, audio_frame> convert(const Frame& f)
    {
        type_safe_union<array2d<rgb_pixel>, audio_frame> obj;

        if (f.is_image())
        {
            DLIB_ASSERT(frame.pixfmt() == AV_PIX_FMT_RGB24, "frame isn't RGB image. Make sure your decoder/demuxer/encoder/muxer has correct args passed to constructor");
            
            array2d<rgb_pixel> image(f.height(), f.width());

            for (int row = 0 ; row < f.height() ; row++)
            {
                memcpy(image.begin() + row * f.width(),
                       f.get_frame().data[0] + row * f.get_frame().linesize[0],
                       f.width()*3);
            }
            
            obj = std::move(image);
        }
        else if (f.is_audio())
        {
            DLIB_ASSERT(frame.samplefmt() == AV_SAMPLE_FMT_S16, "audio buffer requires s16 format. Make sure correct args are passed to constructor of decoder/demuxer/encoder/muxer");

            audio_frame audio;
            audio.sample_rate = f.sample_rate();
            audio.samples.resize(f.nsamples());

            if (f.nchannels() == 1)
            {
                for (int i = 0 ; i < f.nsamples() ; ++i)
                {
                    memcpy(&audio.samples[i].ch1, f.get_frame().data[i], sizeof(int16_t));
                    audio.samples[i].ch2 = audio.samples[i].ch1;
                }  
            }
            else if (f.nchannels() == 2)
            {
                memcpy(audio.samples.data(), f.get_frame().data[0], audio.samples.size()*sizeof(audio_frame::sample));
            }

            obj = std::move(audio);
        }

        return obj;
    }

    Frame convert(const array2d<rgb_pixel>& frame)
    {
        Frame f = Frame::make_image(frame.nr(), frame.nc(), AV_PIX_FMT_RGB24, {});

        for (int row = 0 ; row < f.height() ; row++)
        {
            memcpy(f.get_frame().data[0] + row * f.get_frame().linesize[0],
                   frame.begin() + row * f.width(),
                   f.width()*3);
        }

        return f;
    }

    Frame convert(const audio_frame& frame)
    {
        Frame f = Frame::make_audio(frame.sample_rate, frame.samples.size(), AV_CH_LAYOUT_STEREO, AV_SAMPLE_FMT_S16, frame.timestamp);
        memcpy(f.frame->data[0], frame.samples.data(), frame.samples.size()*sizeof(audio_frame::sample));
        return f;
    }
}