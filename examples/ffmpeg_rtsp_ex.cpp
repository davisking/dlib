// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers.
    It decodes a video, pushes frames over an RTSP client, then receives on an RTSP server.
    
    Please see all the other ffmpeg examples:
        - ffmpeg_info_ex.cpp
        - ffmpeg_video_decoding_ex.cpp
        - ffmpeg_video_demuxing_ex.cpp
        - ffmpeg_video_encoding_ex.cpp
        - ffmpeg_video_muxing_ex.cpp
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",              "input video", 1);
    parser.add_option("codec_video",    "video codec name. e.g. `h264`. Defaults to `mpeg4`", 1);
    parser.add_option("codec_audio",    "audio codec name. e.g. `aac`. Defaults to `ac3`", 1);
    parser.add_option("height",         "height of encoded stream. Defaults to whatever is in the video file", 1);
    parser.add_option("width",          "width of encoded stream. Defaults to whatever is in the video file", 1);
    parser.add_option("sample_rate",    "sample rate of encoded stream. Defaults to whatever is in the video file", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i", "codec_video", "codec_audio", "height", "width", "sample_rate"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    if (!parser.option("i"))
    {
        cout << "Missing -i" << endl;
        parser.print_options();
        return 0;
    }

    const std::string input_filepath    = get_option(parser, "i", "");
    const std::string codec_video       = get_option(parser, "codec_video", "mpeg4");
    const std::string codec_audio       = get_option(parser, "codec_audio", "aac");

    // Check your codec is available
    const auto codecs = ffmpeg::list_codecs();

    if (std::find_if(begin(codecs),   
                     end(codecs),   
                     [&](const auto& c) {return c.codec_name == codec_video && c.supports_encoding;}) == codecs.end())
    {
        cout << "Codec `" << codec_video << "` is not available as an encoder in your installation of ffmpeg." << endl;
        cout << "Either choose another codec, or build ffmpeg from source with the right dependencies installed." << endl;
        cout << "For example, if you are trying to encode to h264, hevc/h265, vp9 or avi, then your installation of ffmpeg ";
        cout << "needs to link to libx264, libx265, libvp9 or libav1" << endl;
        return EXIT_FAILURE;
    }

    if (std::find_if(begin(codecs),   
                     end(codecs),   
                     [&](const auto& c) {return c.codec_name == codec_audio && c.supports_encoding;}) == codecs.end())
    {
        cout << "Codec `" << codec_audio << "` is not available as an encoder in your installation of ffmpeg." << endl;
        cout << "Either choose another codec, or build ffmpeg from source with the right dependencies installed." << endl;
        return EXIT_FAILURE;
    }

    demuxer cap({input_filepath, video_enabled, audio_disabled});

    if (!cap.is_open())
    {
        cout << "Failed to open " << input_filepath << endl;
        return EXIT_FAILURE;
    }

    const std::string url = "rtsp://0.0.0.0:8000/stream";

    std::thread rx{[&] 
    {
        demuxer cap([&] {
            demuxer::args args;
            args.filepath = url;
            args.format_options["rtsp_flags"]       = "listen";
            args.format_options["rtsp_transport"]    = "tcp";
            return args;
        }());

        if (!cap.is_open())
        {
            cout << "Failed to open rtsp client" << endl;
            return;
        }

        image_window win;

        frame f;
        array2d<rgb_pixel> img;

        while (cap.read(f))
        {
            convert(f, img);
            win.set_image(img);
        }
    }};

    std::this_thread::sleep_for(1s);

    std::thread tx{[&] 
    {
        muxer writer([&] {
            muxer::args args;
            args.filepath       = url;
            args.output_format  = "rtsp";
            args.enable_image   = cap.video_enabled();
            args.enable_audio   = false;
            args.format_options["rtsp_transport"] = "tcp";

            if (args.enable_image)
            {
                args.args_image.codec_name  = codec_video;
                args.args_image.h           = get_option(parser, "height", cap.height());
                args.args_image.w           = get_option(parser, "width",  cap.width());
                args.args_image.fmt         = cap.pixel_fmt();
                args.args_image.framerate   = cap.fps();
            }
 
            return args;
        }());

        if (!writer.is_open())
        {
            cout << "Failed to open rtsp server" << endl;
            return;
        }

        frame f;
        while (cap.read(f))
            writer.push(std::move(f));
    }};

    tx.join();
    rx.join();

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    printf("%s\n", e.what());
    return EXIT_FAILURE;
}