#include "../utility.h" // for dlib::exchange
#include "ffmpeg_helpers.h"

namespace dlib
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

    float rational::get()
    {
        return float(num) / denom;
    }

    void av_deleter::operator()(AVFrame *ptr) const
    {
        if (ptr)
            av_frame_free(&ptr);
    }

    void av_deleter::operator()(AVPacket *ptr) const
    {
        if (ptr)
            av_packet_free(&ptr);
    }

    void av_deleter::operator()(AVAudioFifo *ptr) const
    {
        if (ptr)
            av_audio_fifo_free(ptr);
    }

    void av_deleter::operator()(SwsContext *ptr) const
    {
        if (ptr)
            sws_freeContext(ptr);
    }

    void av_deleter::operator()(SwrContext *ptr) const
    {
        if (ptr)
            swr_free(&ptr);
    }

    void av_deleter::operator()(AVCodecContext *ptr) const
    {
        if (ptr)
            avcodec_free_context(&ptr);
    }

    void av_deleter::operator()(AVCodecParserContext *ptr) const
    {
        if (ptr)
            av_parser_close(ptr);
    }

    void av_deleter::operator()(AVFormatContext *ptr) const
    {
        if (ptr)
            avformat_close_input(&ptr);
    }

    av_dict::av_dict(const std::map<std::string, std::string>& options)
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
        av_dict tmp(ori);
        *this = std::move(tmp);
        return *this;
    }

    av_dict::av_dict(av_dict &&ori)
    : avdic{dlib::exchange(ori.avdic, nullptr)}
    {
    }

    av_dict &av_dict::operator=(av_dict &&ori)
    {
        std::swap(avdic, ori.avdic);
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

    Frame Frame::make(
        int             h,
        int             w,
        AVPixelFormat   pixfmt,
        int             sample_rate,
        int             nb_samples,
        uint64_t        channel_layout,
        AVSampleFormat  samplefmt,
        uint64_t        timestamp_us
    )
    {
        Frame obj;
        obj.frame = make_avframe();
        obj.frame->height           = h;
        obj.frame->width            = w;
        obj.frame->sample_rate      = sample_rate;
        obj.frame->channel_layout   = channel_layout;
        obj.frame->nb_samples       = nb_samples;
        obj.frame->format           = h > 0 && w > 0 ? (int)pixfmt : (int)samplefmt;
        obj.timestamp_us            = timestamp_us;

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
            obj.frame->pts = av_rescale_q(obj.timestamp_us, {1,1000000}, {1, obj.frame->sample_rate});

        return obj;
    }

    Frame Frame::make_image(
        int h,
        int w,
        AVPixelFormat fmt,
        uint64_t timestamp_us
    )
    {
        return make(h, w, fmt, 0, 0, 0, AV_SAMPLE_FMT_NONE, timestamp_us);
    }

    Frame Frame::make_audio(
        int             sample_rate,
        int             nb_samples,
        uint64_t        channel_layout,
        AVSampleFormat  fmt,
        uint64_t        timestamp_us
    )
    {
        return make(0,0,AV_PIX_FMT_NONE, sample_rate, nb_samples, channel_layout, fmt, timestamp_us);
    }

    Frame::Frame(const Frame &ori)
    {
        copy(ori);
    }

    Frame& Frame::operator=(const Frame& ori)
    {
        copy(ori);
        return *this;
    }

    void Frame::copy(const Frame &ori)
    {
        if (ori.is_empty())
        {
            Frame empty(std::move(*this));
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
                                 ori.timestamp_us);
                *this = std::move(tmp);
            }

            av_frame_copy(frame.get(), ori.frame.get());
            av_frame_copy_props(frame.get(), ori.frame.get());
        }
    }

    bool Frame::is_empty() const
    {
        return !is_image() && !is_audio();
    }

    bool Frame::is_image() const
    {
        return frame && frame->width > 0 && frame->height > 0 && frame->format >= 0;
    }

    bool Frame::is_audio() const
    {
        return frame && frame->nb_samples > 0 && frame->channel_layout > 0 && frame->sample_rate > 0;
    }

    AVPixelFormat Frame::pixfmt() const
    {
        return is_image() ? (AVPixelFormat)frame->format : AV_PIX_FMT_NONE;
    }

    AVSampleFormat Frame::samplefmt() const
    {
        return is_audio() ? (AVSampleFormat)frame->format : AV_SAMPLE_FMT_NONE;
    }

    int Frame::nchannels() const
    {
        return is_audio() ? av_get_channel_layout_nb_channels(frame->channel_layout) : 0;
    }

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
//                printf("sw_image_resizer::reset() (h,w,pixel_fmt) : (%i,%i,%s) -> (%i,%i,%s)\n",
//                         _src_w, _src_h, av_get_pix_fmt_name(_src_fmt),
//                         _dst_w, _dst_h, av_get_pix_fmt_name(_dst_fmt));

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
        DLIB_ASSERT(src.is_image());

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
                tmp = Frame::make_image(_dst_h, _dst_w, _dst_fmt, src.timestamp_us);
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

            _audioResamplerCtx.reset(nullptr);

            if (src_sample_rate_    != dst_sample_rate_ ||
                src_channel_layout_ != dst_channel_layout_ ||
                src_fmt_            != dst_fmt_)
            {
//                printf("sw_audio_resampler::reset() (sr, layout, pixel_fmt) : (%i,%s,%s}) -> (%i,%s,%s)\n",
//                         src_sample_rate_, get_channel_layout_str(src_channel_layout_).c_str(), av_get_sample_fmt_name(src_fmt_),
//                         dst_sample_rate_, get_channel_layout_str(dst_channel_layout_).c_str(), av_get_sample_fmt_name(dst_fmt_));

                _audioResamplerCtx.reset(swr_alloc_set_opts(NULL,
                                                        dst_channel_layout_, dst_fmt_, dst_sample_rate_,
                                                        src_channel_layout_, src_fmt_, src_sample_rate_,
                                                        0, NULL));
                int ret = 0;
                if ((ret = swr_init(_audioResamplerCtx.get())) < 0)
                {
                    _audioResamplerCtx.reset(nullptr);
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
        DLIB_ASSERT(src.is_audio());

        const bool is_same_object = std::addressof(src) == std::addressof(dst);

        reset(src.frame->sample_rate, src.frame->channel_layout, (AVSampleFormat)src.frame->format,
              dst_sample_rate, dst_channel_layout, dst_samplefmt);

        if (_audioResamplerCtx)
        {
            Frame tmp;
            Frame* ptr = std::addressof(dst);

            const int64_t delay       = swr_get_delay(_audioResamplerCtx.get(), src_sample_rate_);
            const auto dst_nb_samples = av_rescale_rnd(delay + src.frame->nb_samples, dst_sample_rate_, src_sample_rate_, AV_ROUND_UP);

            if (is_same_object ||
                !dst.is_audio() ||
                std::tie(dst.frame->sample_rate, dst.frame->channel_layout, dst.frame->format, dst.frame->nb_samples) !=
                std::tie(dst_sample_rate_, dst_channel_layout_, dst_fmt_, dst_nb_samples))
            {
                tmp = Frame::make_audio(dst_sample_rate_, dst_nb_samples, dst_channel_layout_, dst_fmt_, 0);
                ptr = std::addressof(tmp);
            }

            int ret = swr_convert(_audioResamplerCtx.get(),
                                  ptr->frame->data, ptr->frame->nb_samples,
                                  (const uint8_t**)src.frame->data, src.frame->nb_samples);
            if (ret < 0)
                throw std::runtime_error("swr_convert() failed : " + get_av_error(ret));

            ptr->frame->nb_samples = ret;
            ptr->frame->pts   = _tracked_samples;
            ptr->timestamp_us = av_rescale_q(_tracked_samples, {1, dst_sample_rate_}, {1,1000000});
            _tracked_samples += ptr->frame->nb_samples;

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
        DLIB_ASSERT(in.is_audio());

        std::vector<Frame> outs;

        //check that the configuration hasn't suddenly changed this would be exceptional
        const int nchannels = in.nchannels();
        auto current_params = std::tie(fmt, channels);
        auto new_params = std::tie(in.frame->format, nchannels);

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
                const AVRational tb1 = {1, in.frame->sample_rate};
                const AVRational tb2 = {1, 1000000};
                const uint64_t timestamp_us = av_rescale_q(sample_count, tb1, tb2);
                Frame out = Frame::make_audio(in.frame->sample_rate, frame_size, in.frame->channel_layout, in.samplefmt(), timestamp_us);

                if (av_audio_fifo_read(fifo.get(), (void**)out.frame->data, out.frame->nb_samples) != out.frame->nb_samples)
                    throw std::runtime_error("av_audio_fifo_read() failed to read all requested samples");

                sample_count += out.frame->nb_samples;
                outs.push_back(std::move(out));
            }
        }

        return outs;
    }
}