// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_VIDEO_DEMUXER_IMPL
#define DLIB_VIDEO_DEMUXER_IMPL

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <queue>
#include <thread>
#include "../string.h"
#include "../type_safe_union.h"
#include "../array2d.h"
#include "../audio.h"
#include "ffmpeg_helpers.h"

extern "C" {
#include <libavformat/avformat.h>
}

namespace dlib
{
    struct video_demuxer_args
    {
        struct image_args
        {
            /*
             * Height of decoded images.
             * If 0, then use whatever comes out the decoder.
             * Otherwise, frames will be resized.
             */
            int h = 0;
            
            /*
             * Width of decoded images.
             * If 0, then use whatever comes out the decoder.
             * Otherwise, frames will be resized.
             */
            int w = 0;
        };
        
        struct audio_args
        {
            /*
             * Sample rate of decoded audio frames. 
             * If 0, then use whatever comes out the decoder.
             * Otherwise, frames will be resampled.
             */
            int sample_rate = 0;
        };
        
        /*
         * This can be:
         *  - video filepath    (eg *.mp4)
         *  - audio filepath    (eg *.mp3, *.wav, ...)
         *  - video device      (eg /dev/video0)
         *  - screen buffer     (eg X11 screen grap) you need to set input_format to "X11" for this to work
         *  - audio device      (eg hw:0,0) you need to set input_format to "ALSA" for this to work
         */
        std::string filepath;
        
        /*
         * If empty (default), the format is guessed by libavformat.
         * For certain things, this is required. For example, for audio device,
         * this must be set to "ALSA". For screen grab, this needs to be set
         * to "X11" (TODO: check this)
         * If the filepath is random (like if you used mkstemp()), then libavformat
         * might struggle to guess the format. In which case, you might need
         * to set this to "mp4" for example if the file is an MP4 file.
         */
        std::string input_format = "";
        
        /*
         * Number of threads used per decoder.
         * -1 means std::thread::hardware_concurrency() / 2 (just in case you
         * are decoding both video and audio, each can have half the total 
         * number of threads)
         */
        int nthreads = -1;
        
        /*
         * Basically a workspace reserved for decoded frames which allow to
         * determine heuristics. 
         * -1 means use demuxer's default.
         */
        int probesize = -1;
        
        /*
         * Timeout on read (only really relevant for network demuxers (RTSP, TCP)
         * -1 overflows to something very large which means the demuxer will be
         * blocking.
         */
        uint64_t read_timeout_ms = -1;
        
        /*
         * These are options specific to the demuxer, not the decoders. 
         * For example, for an MP4 file, these are options that relate to the 
         * MP4 container, not the H264 or H265 encoded streams contained within.
         * Note that for video files, you rarely have to modify this.
         * This is only used in anger with network (de)muxers like RTSP, TCP, etc.
         * For example, this allows you to set TCP timeout, whether this is a 
         * listening, or connecting socket, things like that :)
         * 
         * To be exact, these are the options passed to avformat_open_input()
         * in the avformat.h
         */
        std::vector<std::pair<std::string,std::string>> format_options;
        
        /*
         * This enables video/image decoding.
         * You may or may not need to amend the defaults in image_options.
         */
        bool enable_image = true;
        
        /*
         * Image/video decoding options.
         */
        image_args image_options;
        
        /*
         * This enables audio decoding.
         * You may or may not need to amend the defaults in audio_options.
         */
        bool enable_audio = false;
        
        /*
         * Audio decoding options.
         */
        audio_args audio_options;
    };

    class video_demuxer
    {
    public:
        video_demuxer(
            const video_demuxer_args& args
        );
        
        ~video_demuxer();
        
        bool is_open() const;
        bool audio_enabled() const;
        bool video_enabled() const;
        
        /*video dims*/
        int   height() const;
        int   width() const;
        float fps() const;
        int   video_frame_number() const;
        
        /*audio dims*/
        int sample_rate() const;
        
        std::chrono::milliseconds duration() const;

        bool read(
            type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
            uint64_t& timestamp_us
        );
        
        /*metadata*/
        std::map<int,std::map<std::string,std::string>> get_all_metadata() const;
        std::map<std::string,std::string> get_video_metadata() const;
        float get_rotation_angle() const;
        
    private:
        
        bool connect();  
        bool interrupt_callback();
        static int interrupt_callback_static(void* ctx);
        void populate_metadata();
        bool fill_decoded_buffer();
    
        struct channel
        {
            ~channel();
            void close();
            bool is_enabled() const;
            void swap(channel& other);

            AVCodecContext* _pCodecCtx  = nullptr;
            int             _stream_id  = -1;
            uint64_t        _next_pts   = 0;
            ffmpeg::sw_image_resizer    _resizer_image;
            ffmpeg::sw_audio_resampler  _resizer_audio;
        };

        const video_demuxer_args _args;
        bool                _connected      = false;
        AVFormatContext*    _pFormatCtx     = nullptr;
        channel             _channel_video;
        channel             _channel_audio;

        uint64_t            _last_read_time_ms  = 0;
        std::queue<ffmpeg::sw_frame> _src_frame_buffer;
        std::map<int,std::map<std::string,std::string>> _metadata;
    };
    
    inline video_demuxer::video_demuxer(
        const video_demuxer_args& args
    ) : _args(args)
    {
        _connected = connect();
    }
    
    inline video_demuxer::~video_demuxer()
    {
        if (_pFormatCtx)
        {
            avformat_close_input(&_pFormatCtx);
            avformat_free_context(_pFormatCtx);
        }
    }
    
    inline bool video_demuxer::is_open() const
    {
        return _connected;
    }
    
    inline bool video_demuxer::audio_enabled() const
    {
        return _channel_audio.is_enabled();
    }

    inline bool video_demuxer::video_enabled() const
    {
        return _channel_video.is_enabled();
    }

    /*video dims*/
    inline int video_demuxer::height() const
    {
        return _channel_video.is_enabled() ? _channel_video._resizer_image.get_dst_h() : -1;
    }

    inline int video_demuxer::width() const
    {
        return _channel_video.is_enabled() ? _channel_video._resizer_image.get_dst_w() : -1;
    }
    
    inline float video_demuxer::fps() const
    {
        //Might need to adjust _pFormatCtx->fps_probe_size before calling
        //avformat_find_stream_info() to determine framerate.
        //Might also need to adjust _pFormatCtx->max_analyze_duration
        //This could be a potential solution to a future bug.
        return _channel_video.is_enabled() && _pFormatCtx ? 
               (float)_pFormatCtx->streams[_channel_video._stream_id]->avg_frame_rate.num / 
               (float)_pFormatCtx->streams[_channel_video._stream_id]->avg_frame_rate.den : 0;
    }
    
    inline int video_demuxer::video_frame_number() const
    {
        return _channel_video.is_enabled() ? _channel_video._pCodecCtx->frame_number : -1;
    }

    inline std::chrono::milliseconds video_demuxer::duration() const
    {
        return std::chrono::milliseconds(_pFormatCtx ? av_rescale(_pFormatCtx->duration, 1000, AV_TIME_BASE) : 0);
    }
    
    inline int video_demuxer::sample_rate() const
    {
        return _channel_audio.is_enabled() ? _channel_audio._resizer_audio.get_dst_rate() : -1;
    }
    
    inline bool video_demuxer::read(
        type_safe_union<array2d<rgb_pixel>, audio_frame>& frame,
        uint64_t& timestamp_us
    )
    {
        if (!fill_decoded_buffer())
            return false;

        ffmpeg::sw_frame f = std::move(_src_frame_buffer.front());
        _src_frame_buffer.pop();

        ffmpeg::sw_frame tmp;

        if (f.is_video())
        {
            _channel_video._resizer_image.resize(f, tmp);
            array2d<rgb_pixel> frame_image(tmp.frame->height, tmp.frame->width);

            for (int row = 0 ; row < tmp.frame->height ; row++)
            {
                memcpy(frame_image.begin() + row * tmp.frame->width, 
                       tmp.frame->data[0]  + row * tmp.frame->linesize[0], 
                       tmp.frame->width*3);
            }

            frame = std::move(frame_image);
        }
        else
        {
            _channel_audio._resizer_audio.resize(f, tmp);
            audio_frame frame_audio;
            frame_audio.sample_rate = tmp.frame->sample_rate;
            frame_audio.samples.resize(tmp.frame->nb_samples);
            memcpy(frame_audio.samples.data(), tmp.frame->data[0], frame_audio.samples.size()*sizeof(decltype(frame_audio.samples)::value_type));
            frame = std::move(frame_audio);
        }

        timestamp_us = tmp.timestamp_us;

        return true;
    }
    
    inline std::map<int,std::map<std::string,std::string>> video_demuxer::get_all_metadata() const
    {
        const static std::map<int,std::map<std::string,std::string>> empty;
        return _pFormatCtx ? _metadata : empty;
    }

    inline std::map<std::string,std::string> video_demuxer::get_video_metadata() const
    {
        const static std::map<std::string,std::string> empty;
        return _pFormatCtx && 
               _channel_video.is_enabled() &&
               _metadata.find(_channel_video._stream_id) != _metadata.end() ? 
                    _metadata.at(_channel_video._stream_id) : 
                    empty;
    }

    inline float video_demuxer::get_rotation_angle() const
    {
        const auto metadata = get_video_metadata();
        const auto it = metadata.find("rotate");
        return it != metadata.end() ? std::stof(it->second) : 0;
    }
    
    inline bool video_demuxer::connect()
    {
        ffmpeg::av_dict opts = _args.format_options;    
        _pFormatCtx = avformat_alloc_context();
        _pFormatCtx->interrupt_callback = {video_demuxer::interrupt_callback_static, this};

        if (_args.probesize > 0)
            _pFormatCtx->probesize = _args.probesize;

        AVInputFormat* input_format = _args.input_format.empty() ? nullptr : av_find_input_format(_args.input_format.c_str());

        int ret = avformat_open_input(&_pFormatCtx, 
                                      _args.filepath.c_str(),
                                      input_format, 
                                      opts.avdic ? &opts.avdic : NULL);

        if (ret != 0)
        {
            std::cerr << "avformat_open_input() failed with error `" << ffmpeg::get_av_error(ret) << "`\n";
            return false;
        }

        if ((ret = avformat_find_stream_info(_pFormatCtx, NULL)) < 0)
        {
            std::cerr << "avformat_find_stream_info() failed with error `" << ffmpeg::get_av_error(ret) << "`\n";
            return false;
        }

        auto setup_stream = [&](bool is_video, channel& ch) -> bool
        {    
            const AVMediaType media_type = is_video ? AVMEDIA_TYPE_VIDEO : AVMEDIA_TYPE_AUDIO;

            AVCodec* pCodec = 0;
            ch._stream_id = av_find_best_stream(_pFormatCtx, media_type, -1, -1, &pCodec, 0);

            if (ch._stream_id < 0 || ch._stream_id == AVERROR_STREAM_NOT_FOUND || ch._stream_id == AVERROR_DECODER_NOT_FOUND) 
            {
                if (ch._stream_id == AVERROR_STREAM_NOT_FOUND)
                {
                    std::cout << "av_find_best_stream() : stream not found for stream type `" << av_get_media_type_string(media_type) << "`\n";
                    ch.close();
                    return true; //You might be asking for both video and audio but only video is available. That's OK. Just provide video.
                }
                else if (ch._stream_id == AVERROR_DECODER_NOT_FOUND)
                {
                    std::cerr << "av_find_best_stream() : decoder not found for stream type `" << av_get_media_type_string(media_type) << "`\n";
                    std::cerr << "Check that your ffmpeg build is correct" << std::endl;
                }
                else
                {
                    std::cerr << "av_find_best_stream() failed : `" << ffmpeg::get_av_error(ch._stream_id) << "`\n";
                }
                return false;
            }

            /* create decoding context */
            ch._pCodecCtx = avcodec_alloc_context3(pCodec);
            if (!ch._pCodecCtx) 
            {
                std::cerr << "avcodec_alloc_context3() failed\n";
                return false;
            }

            int ret = avcodec_parameters_to_context(ch._pCodecCtx, _pFormatCtx->streams[ch._stream_id]->codecpar);
            if (ret < 0) 
            {
                std::cerr << "avcodec_parameters_to_context() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                return false;
            }

            ch._pCodecCtx->thread_count = _args.nthreads > 0 ? _args.nthreads : std::thread::hardware_concurrency() / 2;

            /* init the decoder. Note: codec-private options could go here.*/
            ret = avcodec_open2(ch._pCodecCtx, pCodec, NULL);
            if (ret < 0)
            {
                std::cerr << "avcodec_open2() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                return false;
            }

            if (ch._pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                if (ch._pCodecCtx->height   == 0 || 
                    ch._pCodecCtx->width    == 0 || 
                    ch._pCodecCtx->pix_fmt  == AV_PIX_FMT_NONE)
                {
                    std::cerr << "Codec parameters look wrong : (h,w) fmt : (" 
                              << ch._pCodecCtx->height << "," 
                              << ch._pCodecCtx->width << ") " 
                              << ch._pCodecCtx->pix_fmt << std::endl;
                    return false;
                }

                ch._resizer_image.reset(
                    ch._pCodecCtx->height,
                    ch._pCodecCtx->width,
                    ch._pCodecCtx->pix_fmt,
                    _args.image_options.h > 0 ? _args.image_options.h : ch._pCodecCtx->height,
                    _args.image_options.w > 0 ? _args.image_options.w : ch._pCodecCtx->width,
                    AV_PIX_FMT_RGB24  
                );
            }
            else if (ch._pCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO)
            {
                if (ch._pCodecCtx->sample_rate == 0 || 
                    ch._pCodecCtx->sample_fmt  == AV_SAMPLE_FMT_NONE)
                {
                    std::cerr << "Codec parameters look wrong:"
                              << " sample_rate : " << ch._pCodecCtx->sample_rate 
                              << " fmt : "  << ch._pCodecCtx->sample_fmt << std::endl;
                    return false;
                }

                if (ch._pCodecCtx->channel_layout == 0)
                    ch._pCodecCtx->channel_layout = av_get_default_channel_layout(ch._pCodecCtx->channels);

                ch._resizer_audio.reset(
                    ch._pCodecCtx->sample_rate,
                    ch._pCodecCtx->channel_layout,
                    ch._pCodecCtx->sample_fmt,
                    _args.audio_options.sample_rate > 0 ? _args.audio_options.sample_rate: ch._pCodecCtx->sample_rate,
                    AV_CH_LAYOUT_STEREO,
                    AV_SAMPLE_FMT_S16
                );
            }
            else
            {
                std::cerr << "Unrecognised media type " << ch._pCodecCtx->codec_type << std::endl;
                return false;
            }

            return true;
        };

        if (_args.enable_image)
        {
            if (!setup_stream(true, _channel_video))
                return false;
        }

        if (_args.enable_audio)
        {
            if (!setup_stream(false, _channel_audio))
                return false;
        }

        if (!_channel_audio.is_enabled() && !_channel_video.is_enabled())
        {
            std::cerr << "At least one of video and audio channels must be enabled" << std::endl;
            return false;
        }

        populate_metadata();

        return true;
    }
    
    inline bool video_demuxer::interrupt_callback()
    {
        if (_args.read_timeout_ms > 0 && _last_read_time_ms > 0)
        {
            const uint64_t now_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            const uint64_t diff_ms = now_ms - _last_read_time_ms;
            return diff_ms > _args.read_timeout_ms;
        }
        return false;
    }
    
    inline int video_demuxer::interrupt_callback_static(void* ctx)
    {
        video_demuxer* me = (video_demuxer*)ctx;
        return me->interrupt_callback();
    }
    
    inline void video_demuxer::populate_metadata()
    {
        for (unsigned int i = 0; i < _pFormatCtx->nb_streams; i++) 
        {
            std::string metadata_str;
            {
                char* charbuf = 0;
                av_dict_get_string(_pFormatCtx->streams[i]->metadata, &charbuf, ',', ';');
                metadata_str = std::string(charbuf);
                free(charbuf);
            }

            std::vector<std::string> keyvals = dlib::split(metadata_str, ";");
            for (size_t kv = 0; kv < keyvals.size(); kv++) {
                std::vector<std::string> kv_item = dlib::split(keyvals[kv], ",");
                assert(kv_item.size() == 2);
                _metadata[i][kv_item[0]] = dlib::trim(kv_item[1]);
            }
        }
    }
    
    inline bool video_demuxer::fill_decoded_buffer()
    {
        if (!_connected)
            return false;

        if (!_src_frame_buffer.empty())
            return true;

        bool ok = false;
        bool do_read = true;

        while(do_read) 
        {
            AVPacket* packet = av_packet_alloc();
            if (packet == nullptr)
                throw std::runtime_error("av_packet_alloc() failed");

            if (av_read_frame(_pFormatCtx, packet) < 0)
            {
                std::cout << "av_read_frame() failed. Probably EOF" << std::endl;
                do_read = false;
            }
            else
            {        
                if (packet->stream_index == _channel_video._stream_id || 
                    packet->stream_index == _channel_audio._stream_id)
                {
                    channel& ch = packet->stream_index == _channel_video._stream_id ? _channel_video : _channel_audio;

                    int ret = 0;
                    if ((ret = avcodec_send_packet(ch._pCodecCtx, packet)) < 0)
                    {
                        std::cerr << "avcodec_send_packet() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                        do_read = false;
                    }
                    else
                    {
                        bool do_receive = true;

                        while (do_receive)
                        {
                            AVFrame* pFrame = av_frame_alloc();
                            if (pFrame == nullptr)
                                throw std::runtime_error("av_frame_alloc() failed");

                            ret = avcodec_receive_frame(ch._pCodecCtx, pFrame);

                            if (ret == AVERROR(EAGAIN))
                            {
                                //need more input, so need to read again
                                do_receive = false; 
                            }
                            else if (ret == AVERROR_EOF)
                            {
                                std::cout << "AV : EOF." << std::endl;
                                do_receive = false;
                                do_read = false;
                            }
                            else if (ret < 0)
                            {
                                std::cerr << "avcodec_receive_frame() failed : `" << ffmpeg::get_av_error(ret) << "`\n";
                                do_receive = false;
                                do_read = false;
                            }
                            else
                            {
                                //we have a frame. 
                                ok = true;
                                do_read = false;
                                //we can carry on receiving but not reading.
                                //Indeed we only really want to read 1 frame, 
                                //but we carry on receiving in case this
                                //packet has multiple frames. (audio)   

                                const bool is_video = ch._pCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO;
                                const AVRational tb = is_video ? _pFormatCtx->streams[ch._stream_id]->time_base : AVRational{1, pFrame->sample_rate};
                                const uint64_t pts  = is_video ? pFrame->pts : ch._next_pts;

                                ch._next_pts += is_video ? 1 : pFrame->nb_samples;

                                pFrame->pts = pts; //adjust, just in case
                                _src_frame_buffer.push(ffmpeg::sw_frame(&pFrame, av_rescale_q(pts, tb, {1,1000000})));
                                _last_read_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                            }

                            if (pFrame)
                                av_frame_free(&pFrame);
                        }
                    }
                }   
            }

            av_packet_free(&packet);
        }

        _connected = ok;
        return _connected;
    }
    
    inline video_demuxer::channel::~channel()
    {
        if (_pCodecCtx)
            avcodec_free_context(&_pCodecCtx);
    }
    
    inline void video_demuxer::channel::close()
    {
        channel empty;
        swap(empty);
    }

    inline bool video_demuxer::channel::is_enabled() const
    {
        return _pCodecCtx != nullptr;
    }

    inline void video_demuxer::channel::swap(channel& other)
    {
        using std::swap;
        swap(_pCodecCtx, other._pCodecCtx);
        swap(_stream_id, other._stream_id);
        swap(_next_pts,  other._next_pts);
        swap(_resizer_image, other._resizer_image);
        swap(_resizer_audio, other._resizer_audio);
    }
}

#endif //DLIB_VIDEO_DEMUXER_IMPL