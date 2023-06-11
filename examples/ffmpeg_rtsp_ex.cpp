// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers.
    It decodes a video, pushes frames over an RTSP client connection and receives on another RTSP server connection.
    
    Please see all the other ffmpeg examples:
        - ffmpeg_info_ex.cpp
        - ffmpeg_video_decoding_ex.cpp
        - ffmpeg_video_decoding2_ex.cpp
        - ffmpeg_video_demuxing_ex.cpp
        - ffmpeg_video_demuxing2_ex.cpp
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
    parser.add_option("i",      "input video", 1);
    parser.add_option("codec",  "video codec name. e.g. `h264`. Defaults to `mpeg4`", 1);
    parser.add_option("height", "height of encoded stream. Defaults to whatever is in the video file", 1);
    parser.add_option("width",  "width of encoded stream. Defaults to whatever is in the video file", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i", "codec", "height", "width"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const std::string input_filepath = parser.option("i").argument();

    // First, we open a video which we use to transmit and receive images over RTSP.
    demuxer cap({input_filepath, video_enabled, audio_disabled});

    if (!cap.is_open())
    {
        cout << "Failed to open " << input_filepath << endl;
        return EXIT_FAILURE;
    }

    if (!cap.video_enabled())
    {
        cout << "This video does not contain images." << endl;
        return EXIT_FAILURE;
    }

    const std::string url = "rtsp://0.0.0.0:8000/stream";

    // We start 2 threads:
    //  - 1 for an RTSP server that listens for an incoming RTSP client and decodes the frames
    //  - 1 for an RTSP client that connects and pushes/muxes image frames.
   
    std::thread rx{[&] 
    {
        // This is an example that show-cases the usage of demuxer::args::format_options.
        // This is used for AVFormatContext and demuxer-private options specific to the container.
        // {"rtsp_flags", "listen"} tells the RTSP demuxer that it is a server
        // {"rtsp_transport", "tcp"} configures RTSP over TCP transport. This way we don't loose any packets.
        // We set a listening timeout of 5s.

        demuxer cap([&] {
            demuxer::args args;
            args.filepath = url;
            args.format_options["rtsp_flags"]       = "listen";
            args.format_options["rtsp_transport"]   = "tcp";
            args.connect_timeout = 5s;
            return args;
        }());

        if (!cap.is_open())
        {
            cout << "Failed to receive connection from RTSP client" << endl;
            return;
        }

        image_window win;
        array2d<rgb_pixel> img;

        while (cap.read(img))
            win.set_image(img);
    }};

    std::this_thread::sleep_for(1s);

    std::thread tx{[&] 
    {
        // The muxer acts as an RTSP client, so we don't use {"rtsp_flags", "listen"}
        // When using RTSP, it is usually a good idea to specify muxer::args::output_format = "rtsp"
        // even though the URL has rtsp:// in its address. Whether or not you need to specify args.output_format = "rtsp"
        // depends on your version of ffmpeg.
        muxer writer([&] {
            muxer::args args;
            args.filepath       = url;
            args.output_format  = "rtsp";
            args.enable_image   = true;
            args.enable_audio   = false;
            args.format_options["rtsp_transport"] = "tcp";

            args.args_image.codec_name  = get_option(parser, "codec", "mpeg4");
            args.args_image.h           = get_option(parser, "height", cap.height());
            args.args_image.w           = get_option(parser, "width",  cap.width());
            args.args_image.fmt         = cap.pixel_fmt();
            args.args_image.framerate   = cap.fps();
 
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