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
        std::string error_str(100, '\0');
        int suc = av_strerror(ret, &error_str[0], sizeof(error_str));
        if (suc == 0)
            error_str.resize(strlen(&error_str[0]));
        else
            error_str = "couldn't set error";
        return error_str;
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

    av_dict::av_dict(const av_dict& ori)
    {
        av_dict_copy(&avdic, ori.avdic, 0);
    }

    av_dict& av_dict::operator=(const av_dict& ori)
    {
        if (this != &ori)
        {
            reset();
            av_dict_copy(&avdic, ori.avdic, 0);
        }
        return *this;
    }

    av_dict::av_dict(av_dict&& ori)
    {
        std::swap(avdic, ori.avdic);
    }

    av_dict& av_dict::operator=(av_dict&& ori)
    {
        std::swap(avdic, ori.avdic);
        return *this;
    }

    av_dict::av_dict(const std::map<std::string, std::string>& options)
    {
        int ret = 0;

        for (const auto& opt : options) {
            if ((ret = av_dict_set(&avdic, opt.first.c_str(), opt.second.c_str(), 0)) < 0)
                throw std::runtime_error("av_dict_set() failed : " + get_av_error(ret));
        }
    }

    av_dict::~av_dict()
    {
        reset();
    }

    void av_dict::reset()
    {
        if (avdic) {
            av_dict_free(&avdic);
            avdic = nullptr;
        }
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

    AVPacketRefLock::AVPacketRefLock(
        av_ptr<AVPacket> &packet
    ) : _packet(packet)
    {
    }

    AVPacketRefLock::~AVPacketRefLock()
    {
        if (_packet)
            av_packet_unref(_packet.get());
    }

    AVFrameRefLock::AVFrameRefLock(
        av_ptr<AVFrame>& frame
    ) : _frame(frame)
    {
    }

    AVFrameRefLock::~AVFrameRefLock()
    {
        if (_frame)
            av_frame_unref(_frame.get());
    }

    sw_frame::sw_frame(
        const av_ptr<AVFrame>& frame,
        uint64_t timestamp_us_
    ) : sw_frame()
    {
        copy(
            (const uint8_t**)frame->data,
            frame->linesize,
            frame->format,
            frame->height,
            frame->width,
            frame->sample_rate,
            frame->nb_samples,
            frame->channel_layout,
            timestamp_us_
        );
    }

    sw_frame::sw_frame(const sw_frame& other) : sw_frame()
    {
        copy(
            (const uint8_t**)other.st.data,
            other.st.linesize,
            other.is_video() ? (int) other.st.pixfmt : (int) other.st.samplefmt,
            other.st.h,
            other.st.w,
            other.st.sample_rate,
            other.st.nb_samples,
            other.st.channel_layout,
            other.st.timestamp_us
        );
    }

    sw_frame& sw_frame::operator=(const sw_frame& other)
    {
        sw_frame tmp(other);
        std::swap(st, tmp.st);
        return *this;
    }

    sw_frame::sw_frame(sw_frame&& ori) noexcept
    : sw_frame()
    {
        std::swap(st, ori.st);
    }

    sw_frame& sw_frame::operator=(sw_frame&& ori) noexcept
    {
        std::swap(st, ori.st);
        return *this;
    }

    sw_frame::~sw_frame()
    {
        if (st.data[0])
            av_freep(&st.data[0]);
    }

    void sw_frame::copy(
        const uint8_t** data_,
        const int* linesize_,
        int format_,
        int h_,
        int w_,
        int sample_rate_,
        int nb_samples_,
        int channel_layout_,
        uint64_t timestamp_us_
    )
    {
        if (h_ > 0 && w_ > 0)
        {
            resize_image(
                h_,
                w_,
                (AVPixelFormat)format_,
                timestamp_us_
            );

            av_image_copy(st.data,  st.linesize,
                          data_, linesize_,
                          st.pixfmt, st.w, st.h);
        }
        else if (sample_rate_ > 0 && nb_samples_ > 0 && channel_layout_ > 0)
        {
            resize_audio(
                    sample_rate_,
                    nb_samples_,
                    channel_layout_,
                    (AVSampleFormat)format_,
                    timestamp_us_
            );

            av_samples_copy(st.data, (uint8_t* const*)data_,
                            0, 0,
                            st.nb_samples, nchannels(), st.samplefmt);
        }
    }

    av_ptr<AVFrame> sw_frame::copy_to_avframe() const
    {
        av_ptr<AVFrame> frame;

        if (is_video() || is_audio())
        {
            frame = make_avframe();

            if (frame)
            {
                if (is_video())
                {
                    frame->height = st.h;
                    frame->width  = st.w;
                    frame->format = (int)st.pixfmt;
                }
                else if (is_audio())
                {
                    frame->sample_rate      = st.sample_rate;
                    frame->nb_samples       = st.nb_samples;
                    frame->channel_layout   = st.channel_layout;
                    frame->format           = (int)st.samplefmt;
                }

                int ret = av_frame_get_buffer(frame.get(), 1);
                if (ret < 0)
                {
                    frame.reset(nullptr);
                    throw std::runtime_error("sw_frame::copy_to_avframe() : av_frame_get_buffer() failed");
                }
                else
                {
                    if (is_video())
                    {
                        av_image_copy(frame->data, frame->linesize,
                                      (const uint8_t**)st.data, st.linesize,
                                      st.pixfmt, st.w, st.h);
                    }
                    else if (is_audio())
                    {
                        frame->pts = av_rescale_q(st.timestamp_us, {1,1000000}, {1, frame->sample_rate});
                        av_samples_copy(frame->data, st.data, 0, 0, st.nb_samples, nchannels(), st.samplefmt);
                    }
                }
            }
        }
        else
        {
        }

        return frame;
    }

    void sw_frame::resize_image(
        int srch,
        int srcw,
        AVPixelFormat srcfmt,
        uint64_t timestamp_us_
    )
    {
        constexpr auto params_zero = std::make_tuple(0,0,AV_PIX_FMT_NONE);
        auto params_this = std::tie(st.h, st.w, st.pixfmt);
        auto params_new  = std::tie(srch, srcw, srcfmt);

        if (params_this != params_new)
        {
            sw_frame empty;
            std::swap(st, empty.st);
            params_this = params_new;

            if (params_new != params_zero)
            {
                int ret = av_image_alloc(st.data, st.linesize, st.w, st.h, st.pixfmt, 1);
                if (ret < 0)
                    throw std::runtime_error("av_image_alloc() failed : " + get_av_error(ret));
            }
        }

        st.timestamp_us = timestamp_us_;
    }

    void sw_frame::resize_audio(
        int             sample_rate_,
        int             nb_samples_,
        uint64_t        channel_layout_,
        AVSampleFormat  samplefmt_,
        uint64_t        timestamp_us_
    )
    {
        constexpr auto params_zero = std::make_tuple(0,0,0,AV_SAMPLE_FMT_NONE);
        auto params_this = std::tie(st.sample_rate,st.nb_samples, st.channel_layout , st.samplefmt);
        auto params_new  = std::tie(sample_rate_, nb_samples_, channel_layout_, samplefmt_);

        if (params_this != params_new)
        {
            sw_frame empty;
            std::swap(st, empty.st);
            params_this = params_new;

            if (params_new != params_zero)
            {
                int ret = av_samples_alloc(st.data, st.linesize, nchannels(), st.nb_samples, st.samplefmt, 1);
                if (ret < 0)
                    throw std::runtime_error("av_samples_alloc() failed : " + get_av_error(ret));
            }
        }

        st.timestamp_us = timestamp_us_;
    }

    bool sw_frame::is_video() const
    {
        return st.h > 0 && st.w > 0 && st.pixfmt != AV_PIX_FMT_NONE;
    }

    bool sw_frame::is_audio() const
    {
        return st.sample_rate > 0 && st.nb_samples > 0 && st.channel_layout > 0 && st.samplefmt != AV_SAMPLE_FMT_NONE;
    }

    int sw_frame::nchannels() const
    {
        return av_get_channel_layout_nb_channels(st.channel_layout);
    }

    int sw_frame::size() const
    {
        if (is_video())
            return av_image_get_buffer_size(st.pixfmt, st.w, st.h, 1);
        else if (is_audio())
            return av_samples_get_buffer_size(nullptr, nchannels(), st.nb_samples, st.samplefmt, 1);
        return 0;
    }

    std::string sw_frame::description() const
    {
        std::string str(256, '\0');
        if (is_video())
        {
            int ret = snprintf(&str[0], str.size(), "video (h,w,fmt) : (%i,%i,%s) - size : %i - timestamp_us %lu",
                               st.h, st.w, get_pixel_fmt_str(st.pixfmt).c_str(), size(), st.timestamp_us);
            str.resize(ret);
        }
        else if (is_audio())
        {
            int ret = snprintf(&str[0], str.size(), "audio (sr, nsamples, layout, fmt) : (%i,%i,%lu,%s) - size : %i - timestamp_us %lu",
                               st.sample_rate, st.nb_samples, st.channel_layout, get_audio_fmt_str(st.samplefmt).c_str(), size(), st.timestamp_us);
            str.resize(ret);
        }
        else
        {
            str = "empty frame";
        }

        return str;
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
//                printf("sw_image_resizer::reset() (h,w,fmt) : (%i,%i,%s) -> (%i,%i,%s)\n",
//                         _src_w, _src_h, av_get_pix_fmt_name(_src_fmt),
//                         _dst_w, _dst_h, av_get_pix_fmt_name(_dst_fmt));

                _imgConvertCtx.reset(sws_getContext(_src_w, _src_h, _src_fmt,
                                                    _dst_w, _dst_h, _dst_fmt,
                                                    SWS_FAST_BILINEAR, NULL, NULL, NULL));
            }
        }
    }

    void sw_image_resizer::resize(
        const sw_frame& src,
        int dst_h, int dst_w, AVPixelFormat dst_pixfmt,
        sw_frame& dst
    )
    {
        reset(src.st.h, src.st.w, src.st.pixfmt,
              dst_h, dst_w, dst_pixfmt);

        if (_imgConvertCtx)
        {
            dst.resize_image(_dst_h, _dst_w, _dst_fmt, src.st.timestamp_us);

            sws_scale(_imgConvertCtx.get(),
                      src.st.data, src.st.linesize, 0, src.st.h,
                      dst.st.data, dst.st.linesize);
        }
        else
        {
            dst = src;
        }
    }

    void sw_image_resizer::resize(
        const sw_frame& src,
        sw_frame& dst
    )
    {
        resize(src, _dst_h, _dst_w, _dst_fmt, dst);
    }

    void sw_image_resizer::resize_inplace(
        int dst_h, int dst_w, AVPixelFormat dst_pixfmt,
        sw_frame& src
    )
    {
        reset(src.st.h, src.st.w, src.st.pixfmt,
              dst_h, dst_w, dst_pixfmt);

        if (_imgConvertCtx)
        {
            sw_frame dst;
            dst.resize_image(_dst_h, _dst_w, _dst_fmt, src.st.timestamp_us);

            sws_scale(_imgConvertCtx.get(),
                      src.st.data, src.st.linesize, 0, src.st.h,
                      dst.st.data, dst.st.linesize);

            src = std::move(dst);
        }
    }

    void sw_image_resizer::resize_inplace(
        sw_frame& f
    )
    {
        resize_inplace(_dst_h, _dst_w, _dst_fmt, f);
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
//                printf("sw_audio_resampler::reset() (sr, layout, fmt) : (%i,%s,%s}) -> (%i,%s,%s)\n",
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

    void sw_audio_resampler::unchecked_resize(
        const sw_frame &src,
        sw_frame &dst
    )
    {
        const int64_t delay       = swr_get_delay(_audioResamplerCtx.get(), src_sample_rate_);
        const auto dst_nb_samples = av_rescale_rnd(delay + src.st.nb_samples, dst_sample_rate_, src_sample_rate_, AV_ROUND_UP);
        dst.resize_audio(dst_sample_rate_, dst_nb_samples, dst_channel_layout_, dst_fmt_, src.st.timestamp_us); //could put 0 as timestamp, wouldn't matter, we use tracked samples instead

        int ret = swr_convert(_audioResamplerCtx.get(), dst.st.data, dst.st.nb_samples, (const uint8_t**)src.st.data, src.st.nb_samples);
        if (ret < 0)
            throw std::runtime_error("swr_convert() failed : " + get_av_error(ret));

        dst.st.nb_samples   = ret;
        dst.st.timestamp_us = av_rescale_q(_tracked_samples, {1, dst_sample_rate_}, {1,1000000});
        _tracked_samples    += dst.st.nb_samples;
    }

    void sw_audio_resampler::resize(
        const sw_frame& src,
        int             dst_sample_rate,
        uint64_t        dst_channel_layout,
        AVSampleFormat  dst_samplefmt,
        sw_frame&       dst
    )
    {
        reset(src.st.sample_rate, src.st.channel_layout, src.st.samplefmt,
              dst_sample_rate, dst_channel_layout, dst_samplefmt);

        if (_audioResamplerCtx)
        {
            unchecked_resize(src, dst);
        }
        else
        {
            dst = src;
        }
    }

    void sw_audio_resampler::resize_inplace(
            int             dst_sample_rate,
            uint64_t        dst_channel_layout,
            AVSampleFormat  dst_samplefmt,
            sw_frame&       src
    )
    {
        reset(src.st.sample_rate, src.st.channel_layout, src.st.samplefmt,
              dst_sample_rate, dst_channel_layout, dst_samplefmt);

        if (_audioResamplerCtx)
        {
            sw_frame dst;
            unchecked_resize(src,dst);
            src = std::move(dst);
        }
    }

    void sw_audio_resampler::resize(
        const sw_frame& src,
        sw_frame& dst
    )
    {
        resize(src, dst_sample_rate_, dst_channel_layout_, dst_fmt_, dst);
    }

    void sw_audio_resampler::resize_inplace(
        sw_frame& f
    )
    {
        resize_inplace(dst_sample_rate_, dst_channel_layout_, dst_fmt_, f);
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
            if (fifo)
                throw std::runtime_error("av_audio_fifo_alloc() failed");
        }
    }

    std::vector<sw_frame> sw_audio_fifo::push_pull(
        sw_frame&& in
    )
    {
        std::vector<sw_frame> outs;

        //check that the configuration hasn't suddenly changed this would be exceptional
        const int nchannels = in.nchannels();
        auto current_params = std::tie(         fmt,  channels);
        auto new_params     = std::tie(in.st.samplefmt, nchannels);

        if (current_params != new_params)
            throw std::runtime_error("new audio frame params differ from first ");

        if (frame_size == 0)
        {
            outs.push_back(std::move(in));
        }
        else
        {
            if (av_audio_fifo_write(fifo.get(), (void**)in.st.data, in.st.nb_samples) != in.st.nb_samples)
                throw std::runtime_error("av_audio_fifo_write() failed to write all samples");

            while (av_audio_fifo_size(fifo.get()) >= frame_size)
            {
                const AVRational tb1 = {1, in.st.sample_rate};
                const AVRational tb2 = {1, 1000000};
                const uint64_t timestamp_us = av_rescale_q(sample_count, tb1, tb2);
                sw_frame out;
                out.resize_audio(in.st.sample_rate, frame_size, in.st.channel_layout, in.st.samplefmt, timestamp_us);

                if (av_audio_fifo_read(fifo.get(), (void**)out.st.data, out.st.nb_samples) != out.st.nb_samples)
                    throw std::runtime_error("av_audio_fifo_read() failed to read all requested samples");

                sample_count += out.st.nb_samples;
                outs.push_back(std::move(out));
            }
        }

        return outs;
    }
}