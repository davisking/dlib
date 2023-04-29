// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the ffmpeg wrappers, in this case the muxing API.

    This is a pretty simple example. It loads a video file, extracts images and audio frames (if present) and
    re-encodes them into a video file.
    
    Please see the following examples on how to decode, demux, and get information on your installation of ffmpeg:
        - ffmpeg_info_ex.cpp
        - ffmpeg_video_decoding_ex.cpp
        - ffmpeg_video_decoding2_ex.cpp
        - ffmpeg_video_demuxing_ex.cpp
        - ffmpeg_video_demuxing2_ex.cpp
*/

#include <cstdio>
#include <dlib/media.h>
#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;
using namespace dlib::ffmpeg;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("i",              "input video", 1);
    parser.add_option("o",              "output file", 1);
    parser.add_option("codec_video",    "video codec name. e.g. `h264`", 1);
    parser.add_option("codec_audio",    "audio codec name. e.g. `aac`", 1);
    parser.add_option("height",         "height of encoded stream. Defaults to whatever is in the video file", 1);
    parser.add_option("width",          "width of encoded stream. Defaults to whatever is in the video file", 1);
    parser.add_option("sample_rate",    "sample rate of encoded stream. Defaults to whatever is in the video file", 1);

    parser.set_group_name("Help Options");
    parser.add_option("h",      "alias of --help");
    parser.add_option("help",   "display this message and exit");

    parser.parse(argc, argv);
    const char* one_time_opts[] = {"i", "o", "codec_video", "codec_audio", "height", "width", "sample_rate"};
    parser.check_one_time_options(one_time_opts);

    if (parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return 0;
    }

    const std::string input_filepath    = parser.option("i").argument();
    const std::string output_filepath   = parser.option("o").argument();

    demuxer cap(input_filepath);

    if (!cap.is_open())
    {
        cout << "Failed to open " << input_filepath << endl;
        return EXIT_FAILURE;
    }

    muxer writer([&] {
        muxer::args args;
        args.filepath     = output_filepath;
        args.enable_image = cap.video_enabled();
        args.enable_audio = cap.audio_enabled();
        if (args.enable_image)
        {
            args.args_image.codec_name  = get_option(parser, "codec_video", "");;
            args.args_image.h           = get_option(parser, "height", cap.height());
            args.args_image.w           = get_option(parser, "width",  cap.width());
            args.args_image.fmt         = cap.pixel_fmt();
            args.args_image.framerate   = cap.fps();
        }
        if (args.enable_audio)
        {
            args.args_audio.codec_name      = get_option(parser, "codec_audio", "");;
            args.args_audio.sample_rate     = get_option(parser, "sample_rate", cap.sample_rate());
            args.args_audio.channel_layout  = cap.channel_layout();
            args.args_audio.fmt             = cap.sample_fmt();
        }
        return args;
    }());

    if (!writer.is_open())
    {
        cout << "Failed to open " << output_filepath << endl;
        return EXIT_FAILURE;
    }

    frame f;
    while (cap.read(f))
        writer.push(std::move(f));

    // writer.flush(); 
    // You don't have to call flush() here because it's called in the destructor of muxer
    // If you call it more than once, it becomes a no-op basically.

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    cout << e.what() << '\n';
    return EXIT_FAILURE;
}
