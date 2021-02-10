// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifdef DLIB_ALL_SOURCE_END
#include "dlib_basic_cpp_build_tutorial.txt"
#endif

#ifndef DLIB_VIDEO_IO_ 
#define DLIB_VIDEO_IO_

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/frame.h>
#include <libavdevice/avdevice.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include <libavcodec/codec.h>
#include <libswscale/swscale.h>
}

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>
#include <thread>
#include <dlib/string.h>
#include <dlib/image_transforms/assign_image.h>

namespace dlib
{
    namespace ffmpeg_impl
    {
        template<typename PixelType>
        struct conversion_capabilities
        {
            static const AVPixelFormat format = AV_PIX_FMT_RGB24;
            using pixel_type = rgb_pixel;
        };

        template<>
        struct conversion_capabilities<bgr_pixel>
        {
            static const AVPixelFormat format = AV_PIX_FMT_BGR24;
            using pixel_type = bgr_pixel;
        };

        template<>
        struct conversion_capabilities<rgb_alpha_pixel>
        {
            static const AVPixelFormat format = AV_PIX_FMT_RGBA;
            using pixel_type = rgb_alpha_pixel;
        };
               
        template<typename PixelType>
        constexpr bool is_ffmpeg_convertible()
        {
            return std::is_same<PixelType, typename conversion_capabilities<PixelType>::pixel_type>::value;
        }
        
        std::string get_av_error(int ret)
        {
            char error_str[256] = {0};
            av_strerror(ret, error_str, sizeof(error_str));
            return std::string(error_str);
        }
        
        struct frame
        {
            uint8_t* data[4] = {nullptr};
            int linesize[4] = {0};
            int64_t timestamp_ns = 0;
            
            frame() = default;
            frame(const frame& ori) = delete;
            frame& operator=(const frame& ori) = delete;
            
            frame(frame&& ori)
            {
                swap(ori);
            }
            
            frame& operator=(frame&& ori)
            {
                reset();
                swap(ori);
                return *this;
            }
            
            ~frame()
            {
                reset();
            }
            
            void swap(frame& o)
            {
                std::swap(data, o.data);
                std::swap(linesize, o.linesize);
                std::swap(timestamp_ns, o.timestamp_ns);
            }
            
            void reset()
            {
                if (data[0])
                {
                    av_freep(&data[0]);
                    data[0] = nullptr;
                    data[1] = nullptr;
                    data[2] = nullptr;
                    data[3] = nullptr;
                    linesize[0] = 0;
                    linesize[1] = 0;
                    linesize[2] = 0;
                    linesize[3] = 0;
                    timestamp_ns = 0;
                }
            }
            
            void resize(int srcw, int srch, AVPixelFormat srcfmt)
            {
                reset();
                if (av_image_alloc(data, linesize, srcw, srch, srcfmt, 1) < 0)
                    throw std::bad_alloc();
            }
            
            void fill(AVFrame* av_frame)
            {
                timestamp_ns = av_frame->best_effort_timestamp;
                resize(av_frame->width, av_frame->height, (enum AVPixelFormat)av_frame->format);
                av_image_copy(data, linesize, (const uint8_t**)av_frame->data, av_frame->linesize, (enum AVPixelFormat)av_frame->format, av_frame->width, av_frame->height);
            }
        };
    }
    
    class video_capture
    {
    private:
        std::string         _arg;
        bool                _connected      = false;
        int                 _videoStream    = -1;  
        AVFormatContext*    _pFormatCtx     = nullptr;
        AVCodecContext*     _pCodecCtx      = nullptr;
        AVFrame*            _pFrame         = nullptr;
        SwsContext*         _imgConvertCtx  = nullptr;
        std::vector<ffmpeg_impl::frame> _src_frame_buffer;
        ffmpeg_impl::frame  _dst_frame;
        AVPixelFormat       _dst_format;
        int                 _dst_h = 0;
        int                 _dst_w = 0;
        std::map<int,std::map<std::string,std::string>> _metadata;
        
        void populate_metadata()
        {
            for (unsigned int i = 0 ; i < _pFormatCtx->nb_streams ; i++)
            {
                std::string metadata_str;
                {
                    char* charbuf = 0;
                    av_dict_get_string(_pFormatCtx->streams[i]->metadata, &charbuf, ',', ';');
                    metadata_str = std::string(charbuf);
                    free(charbuf);
                }

                std::vector<std::string> keyvals = dlib::split(metadata_str, ";");
                for (size_t kv = 0 ; kv < keyvals.size() ; kv++)
                {
                    std::vector<std::string> kv_item = dlib::split(keyvals[kv], ",");
                    assert(kv_item.size() == 2);
                    _metadata[i][kv_item[0]] = dlib::trim(kv_item[1]);
                }
            }
        }
        
        bool connect(bool is_rtsp, int nthreads)
        {
//            av_register_all();

            AVDictionary *avdic = NULL;
            if (is_rtsp)
            {
//                avformat_network_init();
                av_dict_set(&avdic,"rtsp_transport","tcp",0);
                av_dict_set(&avdic,"max_delay","5000000",0);
            }

            int ret = avformat_open_input(&_pFormatCtx, _arg.c_str(), NULL, &avdic);
            if (avdic)
                av_dict_free(&avdic);

            if (ret != 0)
            {
                std::cout << "can't open '" << _arg << "' error : " << ffmpeg_impl::get_av_error(ret) << std::endl;
                return false;
            }

            _pFormatCtx->probesize = 100000000;

            if ((ret = avformat_find_stream_info(_pFormatCtx, NULL)) < 0)
            {
                std::cout << "can't find stream information error : " << ffmpeg_impl::get_av_error(ret) << std::endl;
                return false;
            }

            /* select the video stream */
            AVCodec* pCodec = 0;
            _videoStream = av_find_best_stream(_pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, &pCodec, 0);

            if (_videoStream < 0 || _videoStream == AVERROR_STREAM_NOT_FOUND || _videoStream == AVERROR_DECODER_NOT_FOUND)
            {
                if (_videoStream == AVERROR_STREAM_NOT_FOUND)
                    std::cout << "AV : stream not found" << std::endl;
                else if (_videoStream == AVERROR_DECODER_NOT_FOUND)
                    std::cout << "AV : decoder not found" << std::endl;
                else
                    std::cout << "AV : unknown error in finding stream " << _videoStream << std::endl;
                return false;
            }

            /* create decoding context */
            _pCodecCtx = avcodec_alloc_context3(pCodec);
            if (!_pCodecCtx)
            {
                std::cout << "AV : failed to create decoding context" << std::endl;
                return false;
            }

            if (avcodec_parameters_to_context(_pCodecCtx, _pFormatCtx->streams[_videoStream]->codecpar) < 0)
                return false;
            
            populate_metadata();
            
            _pCodecCtx->thread_count = nthreads;

            /* init the video decoder */
            if (avcodec_open2(_pCodecCtx, pCodec, NULL) < 0) 
            {
                std::cout << "AV : failed to open video decoder" << std::endl;
                return false;
            }

            _pFrame = av_frame_alloc();
            
            //set dst format to src format. Maybe we won't need to convert at all.
            _dst_h      = _pCodecCtx->height;
            _dst_w      = _pCodecCtx->width;
            _dst_format = _pCodecCtx->pix_fmt;
            
            return true;
        }
        
        bool fill_decoded_buffer()
        {
            if (!_connected)
                return false;
            
            if (!_src_frame_buffer.empty())
                return true;

            bool ok = false;
            bool do_read = true;    

            //could be reading audio frames, so need a while loop
            while(do_read) 
            {
                AVPacket packet;
//                av_init_packet(&packet);
//                packet.data = NULL;
//                packet.size = 0;
    
                if (av_read_frame(_pFormatCtx, &packet) < 0)
                {
                    std::cout << "AV : failed to read packet. Probably EOF" << std::endl;
                    do_read = false;
                }
                else
                {
                    if (packet.stream_index == _videoStream)
                    {
                        if (avcodec_send_packet(_pCodecCtx, &packet) < 0)
                        {
                            std::cout << "AV : error while sending a packet to the decoder" << std::endl;
                            do_read = false;
                        }
                        else
                        {
                            bool do_receive = true;
                            
                            while (do_receive)
                            {
                                int ret = avcodec_receive_frame(_pCodecCtx, _pFrame);
                                
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
                                    std::cout << "AV : error receiving frame " << ret << std::endl;
                                    do_receive = false;
                                    do_read = false;
                                }
                                else
                                {
                                    //we have a frame. 
                                    ok = true;
                                    //we can carry on receiving but not reading.
                                    //Indeed we only really want to read 1 frame, 
                                    //but we carry on receiving in case this
                                    //packet has multiple frames.
                                    do_read = false;
                                    ffmpeg_impl::frame f;
                                    f.fill(_pFrame);
                                    _src_frame_buffer.push_back(std::move(f));
                                }
                            }
                        }
                    }

                    av_packet_unref(&packet);        
                }
            }

            _connected = ok;
            return _connected;
        }
                
        void reset_converter(int h, int w, AVPixelFormat dst_format)
        {
            /*
             * This function resets destination pixel formats if required.
             * This function gets called all the time. But, I reckon branch
             * prediction will optimise this out. Indeed, we don't expect
             * there to be a conversion all the time, probably just once, 
             * at the start.
             */
            
            if ((h > 0 && h != _dst_h) || (w > 0 && w != _dst_w) || (dst_format != _dst_format))
            {
                if (_imgConvertCtx)
                {
                    sws_freeContext(_imgConvertCtx);
                    _imgConvertCtx = nullptr;
                }

                _dst_frame.reset();

                _dst_h = h > 0 ? h : _pCodecCtx->height;
                _dst_w = w > 0 ? w : _pCodecCtx->width;
                _dst_format = dst_format;

                /*Is the new destination format different to the original frame format?*/
                if (_dst_h != _pCodecCtx->height || _dst_w != _pCodecCtx->width || _dst_format != _pCodecCtx->pix_fmt)
                {
                    _imgConvertCtx = sws_getContext(_pCodecCtx->width, _pCodecCtx->height, _pCodecCtx->pix_fmt, 
                                                   _dst_w, _dst_h, _dst_format, 
                                                   SWS_BICUBIC, NULL, NULL, NULL);

                    std::cout << "Resetting sws converter " 
                              << "(" << _pCodecCtx->height << "," << _pCodecCtx->width << "," << av_get_pix_fmt_name(_pCodecCtx->pix_fmt) << ")" 
                              << " -> " 
                              << "(" << _dst_h << "," << _dst_w << "," << av_get_pix_fmt_name(_dst_format) << ")" 
                              << std::endl;
                    
                    _dst_frame.resize(_dst_w, _dst_h, _dst_format);
                }
            }
        }
        
        template<typename ImageType>
        void assign_to_image(ImageType& img)
        {
            using pixel_type = typename ImageType::type;
            /*
             * At this stage, all the destination pixel formats and dimensions 
             * are set.
             */
            img.set_size(_dst_h, _dst_w);
            const size_t size = img.size()*pixel_traits<pixel_type>::num;
            
            ffmpeg_impl::frame f(std::move(_src_frame_buffer.back()));
            _src_frame_buffer.pop_back();
            
            if (_imgConvertCtx)
            {
                sws_scale(_imgConvertCtx, f.data, f.linesize, 0, _pCodecCtx->height, 
                          _dst_frame.data, _dst_frame.linesize);
                memcpy(img.begin(), _dst_frame.data[0], size);
            }
            else
            {
                memcpy(img.begin(), f.data[0], size);
            }
        }

    public:
        video_capture()                                       = default;
        video_capture(const video_capture& ori)               = delete;
        video_capture(video_capture&& ori)                    = delete;
        video_capture& operator=(const video_capture& ori)    = delete;
        video_capture& operator=(video_capture&& ori)         = delete;

        ~video_capture()
        {       
            close();
        }

        bool open(
            std::string arg, 
            bool is_rtsp,
            int nthreads = std::thread::hardware_concurrency())
        {
            if (!_connected)
            {
                _arg = arg;
                _connected = connect(is_rtsp, nthreads);
            } 
            else
            {
                std::cout << "Already connected to " << _arg << std::endl;
            }
            if (!_connected)
            {
                close();
            }
            return _connected;
        }

        void close()
        {
            _arg            = "";
            _connected      = false;
            _videoStream    = -1;
            _dst_h          = 0;
            _dst_w          = 0;

            if (_pFormatCtx)
            {
                avformat_close_input(&_pFormatCtx);
                avformat_free_context(_pFormatCtx);
                _pFormatCtx = nullptr;
            }

            if (_pCodecCtx)
            {
                avcodec_free_context(&_pCodecCtx);
                _pCodecCtx = nullptr;
            }

            if (_imgConvertCtx)
            {
                sws_freeContext(_imgConvertCtx);
                _imgConvertCtx = nullptr;
            }

            if (_pFrame)
            {
                av_frame_free(&_pFrame);
                _pFrame = nullptr;
            }
            
            _dst_frame.reset();
            _src_frame_buffer.clear();
        }

        std::string get_label() const
        {
            return _arg;
        }

        bool is_open() const
        {
            return _connected;
        }
        
        int frame_number() const
        {
            return _pCodecCtx ? _pCodecCtx->frame_number : -1;
        }
        
        int src_width() const
        {
            return _pCodecCtx ? _pCodecCtx->width : -1;
        }
        
        int src_height() const
        {
            return _pCodecCtx ? _pCodecCtx->height : -1;
        }
        
        std::string src_format() const
        {
            return _pCodecCtx ? av_get_pix_fmt_name(_pCodecCtx->pix_fmt) : "";
        }
        
        std::map<int,std::map<std::string,std::string>> get_all_metadata() const
        {
            const static std::map<int,std::map<std::string,std::string>> emtpy;
            return _pFormatCtx ? _metadata : emtpy;
        }
        
        std::map<std::string,std::string> get_video_metadata() const
        {
            const static std::map<std::string,std::string> empty;
            return _pFormatCtx ? _metadata.at(_videoStream) : empty;
        }
        
        float get_rotation_angle() const
        {
            if (_pFormatCtx)
            {
                const auto it = _metadata.at(_videoStream).find("rotate");
                return it != _metadata.at(_videoStream).end() ? std::stof(it->second) : 0;
            }
            return 0.0f;
        }
        
        float get_fps() const
        {
            return _pFormatCtx ? (float)_pFormatCtx->streams[_videoStream]->r_frame_rate.num / (float)_pFormatCtx->streams[_videoStream]->r_frame_rate.den : 0.0f;
        }

        template<typename ImageType,
                 typename pixel_traits<typename ImageType::type>::basic_pixel_type* = nullptr>
        bool read(ImageType& img, uint64_t& timestamp_ns)
        {
            using pixel_type = typename ImageType::type;
            
            if (!fill_decoded_buffer())
                return false;

            /*
            * libswscale will do either a full conversion
            * to the correct destination pixel format, or 
            * it will convert to RGB, and dlib will do 
            * the final conversion
            */
            reset_converter(img.nr(), img.nc(), ffmpeg_impl::conversion_capabilities<pixel_type>::format);

            /*
             * compile time value, so this will either get optimised by compiler
             * or branch prediction will get it right every time. 
             * If your compiler supported it, you could write if constexpr :)
             */
            if (ffmpeg_impl::is_ffmpeg_convertible<pixel_type>())
            {
                assign_to_image(img); 
            }
            else
            {
                array2d<typename ffmpeg_impl::conversion_capabilities<pixel_type>::pixel_type> tmp;
                assign_to_image(tmp);
                assign_image(img, tmp);
            }

 //                        timestamp_ns = pFrame->best_effort_timestamp * 1000ull; //
            timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            return true;
        }
    };
    
    class video_encoder
    {
    private:
        bool            _connected      = false;
        AVCodecContext* _pCodecCtx      = nullptr;
        SwsContext*     _imgConvertCtx  = nullptr;
        AVPacket*       _pkt            = nullptr;
        AVFrame*        _dst_frame      = nullptr;
        ffmpeg_impl::frame _tmp_frame;
        int _src_h = 0;
        int _src_w = 0;
        AVPixelFormat _src_format = AV_PIX_FMT_NONE;
        
        void reset_converter(int h, int w, AVPixelFormat src_format)
        {
            /*
             * This function resets source pixel formats if required.
             * This function gets called all the time. But, I reckon branch
             * prediction will optimise this out. Indeed, we don't expect
             * there to be a conversion all the time, probably just once, 
             * at the start.
             */
            
            if (h != _src_h || w != _src_w || src_format != _src_format)
            {
                if (_imgConvertCtx)
                {
                    sws_freeContext(_imgConvertCtx);
                    _imgConvertCtx = nullptr;
                }

                _src_h = h;
                _src_w = w;
                _src_format = src_format;

                /*Do we actually need to convert anything ? */
                if (_src_h != _pCodecCtx->height || _src_w != _pCodecCtx->width || _src_format != _pCodecCtx->pix_fmt)
                {
                    _imgConvertCtx = sws_getContext(_src_w, _src_h, _src_format, 
                                                    _pCodecCtx->width, _pCodecCtx->height, _pCodecCtx->pix_fmt, 
                                                    SWS_BICUBIC, NULL, NULL, NULL);
                    
                    _tmp_frame.resize(_src_w, _src_h, _src_format);
                    
                    std::cout << "Resetting sws converter " 
                              << "(" << _src_h << "," << _src_w << "," << av_get_pix_fmt_name(_src_format) << ")" 
                              << " -> " 
                              << "(" << _pCodecCtx->height << "," << _pCodecCtx->width << "," << av_get_pix_fmt_name(_pCodecCtx->pix_fmt) << ")" 
                              << std::endl;
                }
            }
        }
        
        template<typename ImageType>
        void image_to_frame(const ImageType& img)
        {
            /*
             * At this stage, internal state has been set. Just need to write 
             * to _dst_frame
             */
            
            using pixel_type = typename ImageType::type;
            const size_t size = img.size()*pixel_traits<pixel_type>::num;
            
            if (_imgConvertCtx)
            {
                memcpy(_tmp_frame.data[0], img.begin(), size);
                sws_scale(_imgConvertCtx, 
                          _tmp_frame.data, _tmp_frame.linesize, 0, _src_h, 
                          _dst_frame->data, _dst_frame->linesize);
            }
            else
            {
                memcpy(_dst_frame->data[0], img.begin(), size);
            }
        }
        
        bool encode(bool flush, std::vector<char>& encoded_buf)
        {
            /*
             * At this stage, internal state has been set and _dst_frame is
             * full. Ready to encode.
             */
            int ret = avcodec_send_frame(_pCodecCtx, flush ? NULL : _dst_frame);
            _dst_frame->pts++;
            
            bool ok = ret == 0;
            
            if (ret < 0)
            {
                std::cout << "AV : error : " << ffmpeg_impl::get_av_error(ret) << std::endl;
                ok = false;
            }
            else
            {
                bool keep_receiving = true;
                
                while (keep_receiving)
                {
                    ret = avcodec_receive_packet(_pCodecCtx, _pkt);

                    if (ret == AVERROR(EAGAIN))
                    {
//                        std::cout << "AV : need more input" << std::endl;
                        keep_receiving = false;
                    }
                    else if (ret == AVERROR_EOF)
                    {
//                        std::cout << "AV : flushed." << std::endl;
                        keep_receiving = false;
                    }
                    else if (ret < 0)
                    {
                        std::cout << "AV : encoder error : " << ffmpeg_impl::get_av_error(ret) << std::endl;
                        keep_receiving = false;
                        ok = false;
                    }
                    else
                    {
                        /*
                         * We have a packet so we are going to add this to our
                         * buffer. Now, for whatever reason, the vector might
                         * throw because the system has run out of memory. This
                         * is unlikely but you never know. We are avoid exceptions
                         * in this class. Instead look at the return code.
                         */
                        try
                        {
                            encoded_buf.insert(encoded_buf.end(), _pkt->data, _pkt->data + _pkt->size);
                        }
                        catch (const std::exception& e)
                        {
                            std::cout << "Failed to insert encoded data : '" << e.what() << "'. Try again next iteration" << std::endl;
                            keep_receiving = false;
                        }
                        
                        av_packet_unref(_pkt);
                    }
                }
            }
            return ok;
        }
        
    public:
        video_encoder()                                       = default;
        video_encoder(const video_encoder& ori)               = delete;
        video_encoder& operator=(const video_encoder& ori)    = delete;
        
        video_encoder(video_encoder&& ori)
        {
            swap(ori);
        }
        
        video_encoder& operator=(video_encoder&& ori)
        {
            close();
            swap(ori);
            return *this;
        }
        
        ~video_encoder()
        {
            close();
        }
        
        void swap(video_encoder& o)
        {
            std::swap(_connected,       o._connected);
            std::swap(_src_h,           o._src_h);
            std::swap(_src_w,           o._src_w);
            std::swap(_src_format,      o._src_format);
            std::swap(_pCodecCtx,       o._pCodecCtx);
            std::swap(_imgConvertCtx,   o._imgConvertCtx);
            std::swap(_pkt,             o._pkt);
            std::swap(_dst_frame,       o._dst_frame);
            std::swap(_tmp_frame,       o._tmp_frame);
        }
        
        bool is_open() const
        {
            return _connected;
        }
        
        bool open(
            AVCodecID codec_id,
            AVPixelFormat pix_fmt,
            int fps_num, int fps_denom,
            int h, int w,
            std::vector<std::pair<std::string,std::string>> codec_options, //you need to know what you are doing here
            int nthreads = std::thread::hardware_concurrency()
        )
        {
            close();
            
            AVCodec* codec = avcodec_find_encoder(codec_id);
            if (!codec)
            {
                std::cout << "Codec " << codec_id << " not found" << std::endl;
                return false;
            }
            
            _pCodecCtx = avcodec_alloc_context3(codec);
            if (!_pCodecCtx)
            {
                std::cout << "Could not allocate video context" << std::endl;
                return false;
            }
            
            _pCodecCtx->thread_count    = nthreads;
            _pCodecCtx->height          = h;
            _pCodecCtx->width           = w;
            _pCodecCtx->pix_fmt         = pix_fmt;
            _pCodecCtx->time_base       = (AVRational){fps_denom, fps_num};
            _pCodecCtx->framerate       = (AVRational){fps_num, fps_denom};
            _pCodecCtx->bit_rate        = 400000;   //not sure what a good value for this should be. Does this control lossyness ?
            _pCodecCtx->gop_size        = 10;       //not sure what to put here
            _pCodecCtx->max_b_frames    = 1;        //not sure what to put here
            for (auto it = codec_options.begin() ; it != codec_options.end() ; it++)
                av_opt_set(_pCodecCtx->priv_data, it->first.c_str(), it->second.c_str(), 0);
            
            /* init the video decoder */
            int ret = avcodec_open2(_pCodecCtx, codec, NULL);
            if (ret < 0) 
            {
                std::cout << "AV : failed to open video encoder : " << ffmpeg_impl::get_av_error(ret) << std::endl;
                return false;
            }
            
            _dst_frame = av_frame_alloc();
            if (!_dst_frame)
            {
                std::cout << "AV : failed to allocate video frame" << std::endl;
                return false;
            }
            
            _dst_frame->format  = pix_fmt;
            _dst_frame->height  = h;
            _dst_frame->width   = w;
            _dst_frame->pts     = 0;
            if (av_frame_get_buffer(_dst_frame, 0) < 0)
            {
                std::cout << "AV : failed to allocate video frame data" << std::endl;
                return false;
            }
            if (av_frame_make_writable(_dst_frame) < 0)
            {
                std::cout << "AV : failed to make video frame writeable" << std::endl;
                return false;
            }
            
            _pkt = av_packet_alloc();
            if (!_pkt)
            {
                std::cout << "AV : failed to allocate encoded packet" << std::endl;
                return false;
            }
            
            _connected = true;
            return _connected;
        }
        
        void close()
        {
            _connected  = false;
            _src_h      = 0;
            _src_w      = 0;
            _src_format = AV_PIX_FMT_NONE;
            
            if (_pCodecCtx)
            {
                avcodec_free_context(&_pCodecCtx);
                _pCodecCtx = nullptr;
            }

            if (_imgConvertCtx)
            {
                sws_freeContext(_imgConvertCtx);
                _imgConvertCtx = nullptr;
            }
            
            if (_dst_frame)
            {
                av_frame_free(&_dst_frame);
                _dst_frame = nullptr;
            }
            
            if (_pkt)
            {
                av_packet_free(&_pkt);
                _pkt = nullptr;
            }
            
            _tmp_frame.reset();
        }
        
        template<typename ImageType,
                 typename pixel_traits<typename ImageType::type>::basic_pixel_type* = nullptr>
        bool push(const ImageType& img, std::vector<char>& encoded_buf)
        {
            using pixel_type = typename ImageType::type;
            
            /*
             * libswscale will do either a full conversion to the correct 
             * destination pixel format, or dlib will first convert to RGB 
             * then swscale will do the rest. 
             */
            reset_converter(img.nr(), img.nc(), ffmpeg_impl::conversion_capabilities<pixel_type>::format);
            
            if (ffmpeg_impl::is_ffmpeg_convertible<pixel_type>())
            {
                image_to_frame(img); 
            }
            else
            {
                array2d<typename ffmpeg_impl::conversion_capabilities<pixel_type>::pixel_type> tmp;
                assign_image(tmp, img);
                image_to_frame(tmp);
            }
            
            _connected = encode(false, encoded_buf);
            return _connected;
        }
        
        bool flush(std::vector<char>& encoded_buf)
        {
            return encode(true, encoded_buf);
        }
    };

    inline void ffmpeg_list_available_protocols()
    {
        void* opaque = NULL;
        const char* name = 0;
        while ((name = avio_enum_protocols(&opaque, 0)))
            std::cout << name << std::endl;
    }
    
    struct codec_details
    {
        std::string codec_name;
        bool supports_encoding;
        bool supports_decoding;
    };
    
    inline std::vector<codec_details> ffmpeg_list_codecs()
    {
        std::vector<codec_details> details;
        void* opaque = nullptr;
        const AVCodec* codec = NULL;
        while ((codec = av_codec_iterate(&opaque)))
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
}

#endif // DLIB_VIDEO_IO_ 

